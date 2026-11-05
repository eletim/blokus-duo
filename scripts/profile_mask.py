"""Serialized CUDA batch inference timing; no concurrent generation or audits."""
import sys,time,json,gc
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from blokus.mask_model import CandidateNet,spatial_inputs
from blokus.mask_data import ROOT,load
from blokus.engine import State,legal_moves
from blokus.value_search import encode
from blokus.ai import seed_all,features,heuristic
from blokus.value_experiment import write

def main():
 seed_all(1100);s=State(**next(r for r in load('teacher-labels') if r['bank'] and len(r['moves'])==128)['state']);qs=legal_moves(s);qs=[qs[int(i)] for i in np.argsort(-heuristic(features(s,qs)[0]),kind='stable')[:128]];b,st=encode(s);xx=spatial_inputs(s,qs);results=[]
 for rep in ['summary','cells']:
  model=CandidateNet(rep).cuda().eval();model.load_state_dict(torch.load(ROOT/f'n2400-f0-s1101-{rep}/step-1000.pt',weights_only=False)['model'])
  for n in [1,6,32,128]:
   args=[torch.as_tensor(a,device='cuda') for a in [b[None],st[None],xx[None,:n]]]
   with torch.no_grad():
    for _ in range(10):model(*args).cpu()
    torch.cuda.synchronize();baseline=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();times=[]
    for _ in range(100):
     start=time.perf_counter();model(*args).cpu();times.append((time.perf_counter()-start)*1000)
   results.append(dict(representation=rep,candidates=n,parameters=sum(p.numel() for p in model.parameters()),forward_sync_mean_ms=float(np.mean(times)),p95_ms=float(np.quantile(times,.95)),incremental_peak_allocated_bytes=torch.cuda.max_memory_allocated()-baseline))
  times=[]
  with torch.no_grad():
   for _ in range(100):
    start=time.perf_counter();q=legal_moves(s);q=[q[int(i)] for i in np.argsort(-heuristic(features(s,q)[0]),kind='stable')[:128]];x=spatial_inputs(s,q);bb,ss=encode(s);model(torch.as_tensor(bb[None],device='cuda'),torch.as_tensor(ss[None],device='cuda'),torch.as_tensor(x[None],device='cuda')).cpu();times.append((time.perf_counter()-start)*1000)
  results.append(dict(representation=rep,legal_rank_pack_transfer_forward_sync_mean_ms=float(np.mean(times)),p95_ms=float(np.quantile(times,.95)),note='Includes legal generation and all input construction; excludes 2ply value search. One diagnostic root, not whole-game latency. Both representations use shared spatial packing.'))
  del model,args;gc.collect();torch.cuda.empty_cache()
 write(ROOT/'inference.json',dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(),serialized=True,rows=results));print(results)
if __name__=='__main__':main()
