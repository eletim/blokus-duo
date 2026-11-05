import sys,json
from pathlib import Path
from collections import defaultdict,Counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from blokus.mask_data import ROOT,load
from blokus.engine import State,Move,play
from blokus.visited_data import state_hash
from blokus.value_experiment import write
hist=[defaultdict(list),defaultdict(list)];roots=[set(),set()]
for g in load('source-games'):
 s=State()
 for ply,m in enumerate(g['moves']):
  if ply>=8:hist[int(g['bank'])][state_hash(s)].append(ply)
  play(s,None if m is None else Move(*m),validate=False)
for r in load('teacher-labels'):roots[int(r['bank'])].add(r['state_hash'])
common=set(hist[0])&set(hist[1]);dist=Counter()
for h in common:
 p=min(hist[0][h]+hist[1][h]);dist['8-15' if p<16 else '16-23' if p<24 else '24+']+=1
write(ROOT/'trajectory-overlap.json',dict(source_state_overlap=len(common),minimum_ply_bins=dict(dist),label_root_overlap=len(roots[0]&roots[1]),evaluation_roots_in_training_source_history=len(roots[1]&set(hist[0])),training_roots_in_evaluation_source_history=len(roots[0]&set(hist[1])),note='Only selected labeled roots train the model; source trajectories are not replay-training examples.'))
