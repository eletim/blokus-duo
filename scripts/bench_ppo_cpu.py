"""Alternating old/new portable CPU/CUDA inference benchmark; no training histories."""
import argparse,contextlib,json,random,time
from pathlib import Path
import numpy as np
import torch
from blokus import ai,ppo_policy as p,cpu_reference as ref
from blokus.engine import State,Move,play
from blokus.research import clone

@contextlib.contextmanager
def backend(old):
 targets=[(ai,'features'),(p,'features'),(p,'_candidate_inputs'),(p,'observe')];saved=[getattr(m,k) for m,k in targets]
 if old:
  for (m,k),f in zip(targets,[ref.features,ref.features,ref.inputs,ref.observe]):setattr(m,k,f)
 try:yield
 finally:
  for (m,k),f in zip(targets,saved):setattr(m,k,f)

def states():
 out=[]
 for game in json.loads(Path('artifacts/ppo-cpu-throughput/games.json').read_text()):
  s=State()
  for ply,row in enumerate(game['moves']):
   if ply in [0,4,12,24,36] or row is None:out.append(clone(s))
   play(s,Move(*row) if row else None,validate=False)
 return out

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--device',choices=['cpu','cuda'],default='cpu');ap.add_argument('--repeats',type=int,default=4);a=ap.parse_args();ai.seed_all(1730000);bank=states();agent=p.PPOAgent(1401,a.device);rows=[]
 for repeat in range(a.repeats):
  for old in ([True,False] if repeat%2==0 else [False,True]):
   with backend(old):
    for s in bank:agent.act(s,random.Random(1))
    start=time.perf_counter()
    for _ in range(3):
     for s in bank:agent.act(s,random.Random(1))
    rows.append(dict(repeat=repeat,backend='old' if old else 'new',ms_per_state=(time.perf_counter()-start)*1000/3/len(bank)))
 print(json.dumps(dict(states=len(bank),device=a.device,torch=torch.__version__,rows=rows,initialization_and_warmup_excluded=True)))
if __name__=='__main__':main()
