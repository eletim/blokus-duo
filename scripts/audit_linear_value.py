import gzip,json,hashlib
from pathlib import Path
import numpy as np
from blokus.engine import Move,play,is_legal,legal_moves
from blokus.linear_rl import prefix
from blokus.linear_train2 import state_hash
R=Path('artifacts/linear-value')
def main():
 forbidden=set(json.loads((R/'exclusions.json').read_text())['states']);seen=set();rows=[];first={}
 for seed in (3101,3102,3103):
  starts={}
  for mode in ('old','new'):
   folder=R/f'{mode}-{seed}';games=0;counts={};start_sequence=[]
   with gzip.open(folder/'games.jsonl.gz','rt') as f:
    for line in f:
     batch=json.loads(line);assert len(batch['games'])==16
     for slot,g in enumerate(batch['games']):
      s=prefix(g['prefix_seed']);assert state_hash(s) not in forbidden;start_sequence.append(g['prefix_seed']);learner=0
      expected='current' if slot//2%4<2 else 'past' if slot//2%4==2 else 'initial';assert g['seat']==slot%2 and g['kind']==expected
      for a in g['moves']:
       seen.add(state_hash(s));learner+=int(s.turn==g['seat']);m=Move(*a) if a else None;assert is_legal(s,m) if m else not legal_moves(s);play(s,m,validate=False)
      seen.add(state_hash(s));assert s.done and g['scores']==[s.score(0),s.score(1)] and learner==g['learner_states'];d=s.score(g['seat'])-s.score(1-g['seat']);assert g['reward']==float(np.sign(d));games+=1;counts[(g['kind'],g['seat'])]=counts.get((g['kind'],g['seat']),0)+1
     if batch['iteration']==1:first[f'{mode}-{seed}']=batch['games']
   starts[mode]=start_sequence;rows.append(dict(seed=seed,mode=mode,games=games,seat_opponent_counts={str(k):v for k,v in counts.items()}))
  assert starts['old']==starts['new']
 a=np.array(sorted(bytes.fromhex(x) for x in seen),dtype='S32');expected=json.loads((R/'training-summary.json').read_text()).get('main_state_sha256',json.loads((R/'training-summary.json').read_text())['state_sha256']);assert hashlib.sha256(a.tobytes()).hexdigest()==expected
 # Reproduce the discarded speed pilot to include every visited pilot state in separation.
 from blokus.linear_train2 import Trainer
 main_sha=expected;main_count=len(seen)
 for mode in ('old','new'):
  t=Trainer(9101,mode,forbidden=forbidden)
  for _ in range(32):t.step(seen)
 allstates=np.array(sorted(bytes.fromhex(x) for x in seen),dtype='S32');np.save('/tmp/linear-value-training.npy',allstates)
 with gzip.open(R/'training-states.bin.gz','wb') as stream:stream.write(allstates.tobytes())
 summary=json.loads((R/'training-summary.json').read_text());summary.update(main_state_sha256=main_sha,main_unique_states=main_count,unique_states=len(seen),state_sha256=hashlib.sha256(allstates.tobytes()).hexdigest(),pilot_states_included=True);(R/'training-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 result=dict(pass_=True,rows=rows,all_moves_replayed=True,all_outcomes_verified=True,training_hash_verified=True,unique_states=len(seen),common_external_prefixes=True,first_batch_same={str(seed):first[f'old-{seed}']==first[f'new-{seed}'] for seed in (3101,3102,3103)})
 (R/'training-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True)
if __name__=='__main__':main()
