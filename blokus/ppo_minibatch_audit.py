"""New-data-only replay, old index SHA references, and shared exogenous prefix audit."""
import gzip,hashlib,json,time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from .ppo_minibatch import R,SEEDS,write
from .ppo_gpu_train import sha
from .ppo_runtime import state_hash,pool_modes
from .engine import State,Move,play,is_legal,legal_moves
CACHE=Path('/tmp/ppo-minibatch-training.npy')
def audit():
 start=time.perf_counter();oldpath=Path('/tmp/ppo-four-gpu-all-training.npy');old=np.load(oldpath,mmap_mode='r');prior=json.loads(Path('artifacts/ppo-four-gpu/training-audit.json').read_text());assert hashlib.sha256(old.tobytes()).hexdigest()==prior['union_sha256']
 forbidden=set(json.loads((R/'excluded-starts.json').read_text())['states']);states=set();prefixes={};reports=[];checks=0
 for hp in sorted(list(R.rglob('history-*.jsonl.gz'))+[R/'benchmark-games.jsonl.gz']):
  counts=Counter();targets_count=Counter();ng=0;incidental=set();main=hp.parent.name.startswith('batch-');mb=int(hp.parent.name.split('-')[1]) if main else None;seed=int(hp.parent.name.split('-')[-1]) if main else None
  # Check checkpoint provenance and segment bytes, including pilots.
  checkpoint=hp.parent/('iteration-'+hp.stem.split('-')[-1].split('.')[0]+'.pt')
  if checkpoint.exists():
   ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
   if 'history_segment' in ck:
    assert ck['history_segment']['sha256']==sha(hp)
    if 'diagnostic-before-determinism-fix' not in str(hp):
     base=hp.parent/ck['base_checkpoint']['path'];assert sha(base)==ck['base_checkpoint']['sha256']
   if main:assert ck['minibatch']==mb and ck['seed']==seed
  with gzip.open(hp,'rt') as stream:
   for line in stream:
    row=json.loads(line);iteration=row['iteration'];assert len(row['games'])==16
    for slot,g in enumerate(row['games']):
     assert g['opponent']==pool_modes(iteration-1,16)[slot] and g['learner_seat']==(slot//4)%2
     if main:
      assert g['id']==seed*100000+(iteration-1)*16+slot
      key=(seed,iteration,slot);val=json.dumps(g['prefix']);assert key not in prefixes or prefixes[key]==val;prefixes[key]=val
     counts[g['opponent']]+=1;s=State();targets={t['ply']:t for t in g['targets']};assert len(targets)==len(g['targets'])
     for ply,move in enumerate(g['prefix']+g['moves']):
      h=state_hash(s);states.add(bytes.fromhex(h));m=Move(*move) if move else None
      if ply>=len(g['prefix']):
       assert (ply in targets)==(g['opponent']=='current' or s.turn==g['learner_seat'])
       if h in forbidden:incidental.add(h)
      if ply in targets:
       t=targets[ply];assert t['state_hash']==h and t['player']==s.turn and t['return']==float(np.sign(g['scores'][s.turn]-g['scores'][1-s.turn]));targets_count[g['opponent']]+=1
       if checks%256==0:
        qs=legal_moves(s);assert (qs[t['action']].move if qs else None)==m
       checks+=1
      assert is_legal(s,m) if m else not legal_moves(s);play(s,m,validate=False)
      if ply<len(g['prefix']):assert state_hash(s) not in forbidden
     states.add(bytes.fromhex(state_hash(s)));assert s.done and g['scores']==[s.score(0),s.score(1)];ng+=1
  reports.append(dict(path=str(hp),sha256=sha(hp),games=ng,main=main,opponent_games=dict(counts),learner_states=dict(targets_count),incidental_old_start_overlap=len(incidental)));print('AUDIT',hp,ng,flush=True)
 # Benchmark rollout is not training, but conservatively exclude its stored observations too.
 for rec in torch.load(R/'benchmark-rollout.pt',weights_only=False):states.add(bytes.fromhex(rec['state_hash']))
 delta=np.array(sorted(states),dtype='S32');union=np.unique(np.concatenate((old,delta)));np.save(CACHE,union)
 with gzip.open(R/'training-state-delta.bin.gz','wb') as f:f.write(delta.tobytes())
 raw=(R/'training-state-delta.bin.gz').read_bytes();parts=[]
 for i in range(0,len(raw),25_000_000):
  path=R/f'training-state-delta-{i//25_000_000:02d}.part';path.write_bytes(raw[i:i+25_000_000]);parts.append(dict(path=path.name,sha256=sha(path),bytes=path.stat().st_size))
 delta_sha=sha(R/'training-state-delta.bin.gz');(R/'training-state-delta.bin.gz').unlink()
 write(R/'training-audit.json',dict(delta_parts=parts,delta_encoding='concatenate parts in order, then gzip decompress into S32 array',pass_=True,rows=reports,shared_external_prefixes=len(prefixes),all_main_prefixes_identical_between_conditions=True,union_sha256=hashlib.sha256(union.tobytes()).hexdigest(),union_states=len(union),delta_count=len(delta),delta_sha256=delta_sha,prior=dict(path='artifacts/ppo-four-gpu/training-audit.json',sha256=sha('artifacts/ppo-four-gpu/training-audit.json'),index_sha256=prior['union_sha256']),seconds=time.perf_counter()-start,all_moves_legally_replayed=True,action_index_stride=256))
if __name__=='__main__':audit()
