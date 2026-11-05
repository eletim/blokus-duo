"""Post-hoc exact input collisions; no changes to data, gate or model selection."""
import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from blokus.scaling_data import ROOT,load
from blokus.value_experiment import write

def main():
 size=int(sys.argv[1]);raw=np.load(ROOT/f'data-{size}.npz');d={k:raw[k] for k in raw.files};records=[r for r in load('teacher-labels') if r['bank'] or r['index']<size*3//2];groups=0;different=0;lower_sse=0.;examples=[];all_groups=0;states=0;max_range=0
 for j,r in enumerate(records):
  by={};n=len(r['moves'])
  for i in range(n):by.setdefault(d['x'][j,i].tobytes(),[]).append(i)
  collisions=[v for v in by.values() if len(v)>1];all_groups+=len(collisions);states+=bool(collisions)
  for ids in collisions:
   known=[i for i in ids if d['mask'][j,i]]
   if len(known)<2:continue
   groups+=1;y=d['target'][j,known];span=float(np.ptp(y));max_range=max(max_range,span);lower_sse+=float(((y-y.mean())**2).sum())
   if span>1e-5:
    different+=1
    if len(examples)<10:examples.append(dict(game=r['game'],state_hash=r['state_hash'],moves=[r['moves'][i] for i in known],target_delta=y.tolist(),same_candidate_input=True))
 write(ROOT/'input-collisions.json',dict(size=size,roots=len(records),roots_with_full_candidate_collision=states,full_candidate_collision_groups=all_groups,labeled_collision_groups=groups,groups_with_different_targets=different,irreducible_empirical_mse_lower_bound_from_collisions=lower_sse/int(d['mask'].sum()),max_target_range=max_range,examples=examples,note='Identical whole board/state and candidate feature bytes force identical model prediction. Candidate summary omits explicit orientation/cell mask. This lower bound covers only observed collisions and is not proof of the dominant failure cause. Diagnostic only, never used to select a checkpoint.'))
if __name__=='__main__':main()
