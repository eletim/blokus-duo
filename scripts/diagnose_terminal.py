"""New diagnostic games: policy change and actual CUDA kernel timing, no selection."""
import sys,json,time,random,argparse
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from blokus.terminal_ppo import ROOT,TerminalNet,observe,collate
from blokus.tournament import make_prefixes,prefix_state
from blokus.engine import legal_moves,play
from blokus.ai import features,heuristic,seed_all
from blokus.value_experiment import write

def main(root):
 root=Path(root);seed_all(1400);fixture=make_prefixes(1490000,16);obs=[]
 for p in fixture['prefixes']:
  s=prefix_state(p);ply=0
  while not s.done:
   if ply in [8,16,24]:obs.append(observe(s)[0])
   qs=legal_moves(s);m=qs[int(np.argmax(heuristic(features(s,qs)[0])))].move if qs else None;play(s,m,validate=False);ply+=1
 selection=json.loads((root/'selection.json').read_text());report=[]
 for seed,sel in selection.items():
  models=[]
  for path in [root/f'seed-{seed}/untrained.pt',Path(sel['path'])]:
   m=TerminalNet().cuda().eval();m.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model']);models.append(m)
  values=[]
  with torch.no_grad():
   for o in obs:
    data=collate([o]);a,_=models[0](*data);b,_=models[1](*data);pa=a.softmax(-1);pb=b.softmax(-1);values.append([float(a.argmax()!=b.argmax()),float((pa*(a.log_softmax(-1)-b.log_softmax(-1))).sum()),float(-(pb*b.log_softmax(-1)).sum())])
  report.append(dict(seed=seed,roots=len(obs),greedy_change_rate=float(np.mean(values,0)[0]),kl_initial_to_trained=float(np.mean(values,0)[1]),trained_entropy=float(np.mean(values,0)[2])))
 timing=[];model=models[-1]
 for batch in [1,16]:
  data=collate(obs[:batch]);
  with torch.no_grad():
   for _ in range(10):model(*data)
   torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();base=torch.cuda.memory_allocated();wall=[];gpu=[]
   for _ in range(100):
    begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start=time.perf_counter();begin.record();l,v=model(*data);end.record();torch.cuda.synchronize();wall.append((time.perf_counter()-start)*1000);gpu.append(begin.elapsed_time(end))
  timing.append(dict(batch=batch,max_candidates=int(data[-1].shape[1]),valid_candidates=int(data[-1].sum()),gpu_kernel_ms=float(np.mean(gpu)),host_forward_sync_ms=float(np.mean(wall)),incremental_peak_bytes=torch.cuda.max_memory_allocated()-base,note='Prepacked GPU inputs, excludes legal generation and H2D; see actual match timing for whole decision.'))
 write(root/'policy-diagnostic.json',dict(prefix_seed=1490000,selection_use=False,rows=report,timing=timing))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',default=str(ROOT));main(p.parse_args().root)
