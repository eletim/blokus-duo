import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from run_deeper import R,INDEX,contains,state,state_hash,write
from blokus.engine import Move,play
phases={};starts={}
for phase in ('validation500','confirmation500'):
 bank=json.loads((R/f'{phase}-bank.json').read_text())['prefixes'];starts[phase]={p['state_hash'] for p in bank};seen=set()
 for f in R.glob(phase+'-*-*-*.json'):
  for g in json.loads(f.read_text())['games']:
   s=state(bank[g['prefix']]['moves']);seen.add(state_hash(s))
   for raw in g['moves']:play(s,Move(*raw) if raw else None,validate=False);seen.add(state_hash(s))
 phases[phase]=seen
idx=np.load(INDEX,mmap_mode='r');diag=json.loads((R/'diagnostic-bank.json').read_text())['prefixes'];dh={p['state_hash'] for p in diag};assert all(not contains(idx,h) for h in dh)
a=len(starts['confirmation500']&phases['validation500']);b=len(phases['confirmation500']&phases['validation500']);assert a==b==0;assert not dh&(starts['validation500']|starts['confirmation500'])
write(R/'cross-phase-audit.json',dict(pass_=True,unique_visited={k:len(v) for k,v in phases.items()},confirmation_start_vs_validation_visited=a,all_visited_cross_phase_overlap=b,diagnostic_start_training_overlap=0,diagnostic_vs_evaluation_start_overlap=0))
