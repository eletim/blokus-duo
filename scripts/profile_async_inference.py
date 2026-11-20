"""Separate post-timing operator diagnostic; never used as throughput evidence."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import gzip,json,random
from pathlib import Path
import torch
from blokus.ppo_policy import PPOAgent
from blokus.ppo_frontier import FrontierEvaluator
from blokus.generation_graph import GraphEvaluator
from blokus.generation_encoding import encode
from blokus.engine import State,Move,play
R=Path('artifacts/gpu-async512');torch.set_num_threads(1);torch.use_deterministic_algorithms(True);p=PPOAgent(1751401,'cuda',backend='native');game=json.load(gzip.open(R/'quality/cpu-1/1751401-0.json.gz','rt'));s=State();req=[]
for i,m in enumerate(game['moves']):
 if not s.done and i in (8,12,16,20,24,28):
  obs,_=p._observation(s);req.extend([('policy',obs),('value',encode(s))])
 play(s,Move(*m) if m else None,validate=False)
req=(req*16)[:128];meta=dict(torch=torch.__version__,cuda=torch.version.cuda,matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,deterministic=torch.are_deterministic_algorithms_enabled(),tensor_dtype='float32',requests=len(req),note='event intervals include stream scheduling; profiler adds overhead and is separate from throughput')
for name,ev in [('eager',FrontierEvaluator(p)),('graph',GraphEvaluator(p))]:
 for _ in range(3):ev.batch(req)
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as prof:
  for _ in range(20):ev.batch(req)
 (R/f'profile-{name}.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=35))
 prof.export_chrome_trace(str(R/f'profile-{name}.json'))
(R/'precision-environment.json').write_text(json.dumps(meta,indent=2)+'\n')
