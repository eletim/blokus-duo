"""Kernel-only channel ablation and phase profile; run after throughput benchmark."""
import copy,json,time
from pathlib import Path
import numpy as np
import torch
from blokus import ppo_runtime as r,ppo_native as n,ppo_runtime_reference as ref
from blokus.ppo_policy import TerminalNet,collate,observe
from blokus.value_experiment import write
ROOT=Path('artifacts/ppo-four-channel-runtime')
r.seed_all(1740200);n.tables();m=TerminalNet(6).cuda();m.load_state_dict(torch.load('artifacts/ppo-gen2/seed-1401.pt',weights_only=True)['model']);records,games,_=ref.collect(m,copy.deepcopy(m),0,1740200)
rows=[]
# Identical observations/minibatch cardinalities, only drop start planes for channel ablation.
for offset in [0,len(records)//2,len(records)-16]:
 obs=[x['obs'] for x in records[offset:offset+16]]
 for repeat_order in range(3):
  for channels in ([6,4] if repeat_order%2==0 else [4,6]):
   sliced=[(x[0][:channels].copy(),*x[1:]) for x in obs];model=TerminalNet(channels).cuda();t=time.perf_counter();batch=collate(sliced,'cuda');torch.cuda.synchronize();transfer=time.perf_counter()-t;forward=[];backward=[]
   for repeat in range(20):
    model.zero_grad(set_to_none=True);a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);c=torch.cuda.Event(enable_timing=True);d=torch.cuda.Event(enable_timing=True);a.record();logits,v=model(*batch);b.record();loss=logits.masked_fill(~batch[-1],0).sum()/batch[-1].sum()+v.square().mean();c.record();loss.backward();d.record();d.synchronize()
    if repeat>=10:forward.append(a.elapsed_time(b));backward.append(c.elapsed_time(d))
   rows.append(dict(offset=offset,channels=channels,repeat_order=repeat_order,batch=len(obs),max_candidates=batch[2].shape[1],transfer_and_collate_ms=transfer*1000,forward_gpu_ms=forward,backward_gpu_ms=backward))
# Separately measure Python legal/features/board pack, native legal-only vs fused.
from blokus.engine import State,Move,legal_moves,play
states=[]
for g in games:
 s=State()
 for i,row in enumerate(g['prefix']+g['moves']):
  if i in [4,12,24,32]:states.append(copy.deepcopy(s))
  play(s,Move(*row) if row else None,validate=False)
ops={'python_legal':lambda s:legal_moves(s),'python_features_pack_given_legal':None,'native_legal_only':lambda s:n.scan(s,False),'native_legal_features_fused':lambda s:n.scan(s,True),'native_full_observe4':lambda s:n.observe(s,4,'native')};profiles=[]
for label,fn in ops.items():
 if fn is None:
  qs=[legal_moves(s) for s in states];t=time.perf_counter()
  for _ in range(5):
   for s,q in zip(states,qs):observe(s,q,channels=4)
 else:
  for s in states:fn(s)
  t=time.perf_counter()
  for _ in range(5):
   for s in states:fn(s)
 profiles.append(dict(operation=label,ms_per_state=(time.perf_counter()-t)*1000/(5*len(states))))
write(ROOT/'phase-profile.json',dict(kernel_rows=rows,CPU_profiles=profiles,states=len(states),GPU_timing='CUDA event model forward and loss.backward separately; excludes transfer/loss construction; 10 warmups then10; three alternating channel-order repeats',CPU_timing='warm one bank, then5 repetitions; features conditional on same precomputed legal candidates',channel_ablation_same_observations=True,weights_for_channel_ablation='fresh random same architecture size except input channels; no conversion or exported checkpoint'))
