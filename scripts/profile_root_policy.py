import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import gzip,json,random,numpy as np
from blokus.root_data import ROOT
from blokus.root_experiment import agent
from blokus.reply_search import reset_game
from blokus.engine import State
from blokus.research import clone
from blokus.ai import seed_all
from blokus.value_experiment import write
seed_all(901);rows=json.load(gzip.open(ROOT/'teacher-traces.json.gz','rt'));states=[r for r in rows if r['trainable'] and r['split']==1][::4][:12];out=[]
for budget in [6,25]:
 for kind,policy in [('heuristic',None),('untrained',str(ROOT/'seed-901/untrained.pt')),('student',str(ROOT/'seed-901/step-1000.pt'))]:
  a=agent(dict(method=kind,policy=policy,budget_ms=budget))
  for r in states:
   reset_game(a);a.act(clone(State(**r['state'])),random.Random(91));out.append(dict(kind=kind,budget_ms=budget,state_hash=r['state_hash'],**a.decisions[-1]))
write(ROOT/'inference-pilot.json',out)
for budget in [6,25]:
 for kind in ['heuristic','untrained','student']:
  ds=[r for r in out if r['kind']==kind and r['budget_ms']==budget];print(budget,kind,'policy ms',np.median([r['policy_ms'] for r in ds]),'roots',np.mean([r['completed_roots'] for r in ds]),flush=True)
