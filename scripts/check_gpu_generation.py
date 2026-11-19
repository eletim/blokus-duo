"""Identical-input numeric checks; run separately from timed games."""
import json,gzip,copy,random,time
from pathlib import Path
import numpy as np
import torch
import gpu_generation as g
from blokus.engine import State,Move,play
from blokus.ppo_policy import PPOAgent,collate
from blokus.ppo_value_leaf import value_inputs,value_only
from blokus.ppo_search_deeper import DeepSearch

def main():
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 rows=[];fixed=[];start=time.perf_counter()
 for seed in g.e.SEEDS:
  cpu=PPOAgent(seed,'cpu',backend='native');gpu=PPOAgent(seed,'cuda',backend='native')
  states=[]
  for path in sorted((g.R/'cpu-1').glob(f'{seed}-*.gz')):
   d=json.load(gzip.open(path,'rt'));s=State()
   for ply,m in enumerate(d['moves']):
    if ply in (0,8,12,20,28,32) or not g.e.legal_moves(s):states.append(copy.deepcopy(s))
    play(s,Move(*m) if m else None,validate=True)
   states.append(s)
  active=[s for s in states if not s.done]
  for k in range(0,len(active),8):
   zs=active[k:k+8];obs=[cpu._observation(s)[0] for s in zs]
   with torch.inference_mode():
    lg,vg=gpu.model(*collate(obs,'cuda'));lg=lg.cpu();vg=vg.cpu()
    b,st=zip(*(value_inputs(s,cpu.channels) for s in zs));b=torch.tensor(np.stack(b),device='cuda');st=torch.tensor(np.stack(st),device='cuda')
    vl=gpu.model.value(torch.cat((gpu.model.global_net(gpu.model.conv(b.float())),st),-1)).flatten().cpu()
    for j,(s,o) in enumerate(zip(zs,obs)):
     lc,vc=cpu.model(*collate([o],'cpu'));n=len(o[2]);a=lc[0].numpy();b=lg[j,:n].numpy()
     rows.append(dict(seed=seed,turn=s.turn,candidates=n,logit_error=float(np.max(np.abs(a-b))),value_error=float(abs(vc[0]-vg[j])),leaf_error=float(abs(vc[0]-vl[j])),top1=bool(a.argmax()==b.argmax()),top4=bool(torch.equal(lc[0].argsort(descending=True)[:4],lg[j,:n].argsort(descending=True)[:4])),gap=float(np.sort(a)[-1]-np.sort(a)[-2]) if n>1 else None))
  # Search unchanged: same-input complete trees on stage-spanning roots. GPU
  # batching is tested above; this checks backend arithmetic through search.
  for s in active[::max(1,len(active)//6)][:6]:
   ac=DeepSearch(cpu,60000);ag=DeepSearch(gpu,60000)
   m=ac.act(s,random.Random(51));n=ag.act(s,random.Random(51))
   fixed.append(dict(seed=seed,move_equal=m==n,cpu_depth=ac.stats['depth'],gpu_depth=ag.stats['depth'],root_error=max([abs(x-y) for x,y in zip(ac.stats.get('root_scores',[]),ag.stats.get('root_scores',[]))] or [0]),cpu_nodes=ac.stats['nodes'],gpu_nodes=ag.stats['nodes']))
  assert all(value_only(cpu,s)==value_only(gpu,s) for s in states if s.done)
 out=dict(tolerance=2e-5,rows=rows,fixed_trees=fixed,seconds=time.perf_counter()-start,
          max_logit=max(r['logit_error'] for r in rows),max_value=max(r['value_error'] for r in rows),max_leaf=max(r['leaf_error'] for r in rows),
          top1_agreement=np.mean([r['top1'] for r in rows]),top4_agreement=np.mean([r['top4'] for r in rows]))
 out['passed']=bool(max(out['max_logit'],out['max_value'],out['max_leaf'])<=2e-5 and out['top1_agreement']>=.99 and all(r['move_equal'] and r['root_error']<=2e-5 for r in fixed))
 g.write(g.R/'numeric-check.json',out);print({k:v for k,v in out.items() if k not in ('rows','fixed_trees')},flush=True)
if __name__=='__main__':main()
