"""Replay all new games, audit rewards, and build ephemeral union of historical/new visited hashes."""
import argparse,gzip,hashlib,json,posixpath,subprocess,time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from .engine import State,Move,play,is_legal,legal_moves
from .ppo_runtime import state_hash,pool_modes
from .ppo_gpu_train import sha
R=Path('artifacts/ppo-four-gpu');CACHE=Path('/tmp/ppo-four-gpu-all-training.npy')
def write(path,value):Path(path).write_text(json.dumps(value,indent=2))
def old_index(ref,path,seen,parts,refs):
 if path in seen:return
 seen.add(path);raw=subprocess.check_output(['git','show',ref+':'+path]);d=json.loads(raw);refs.append(dict(ref=ref,path=path,sha256=hashlib.sha256(raw).hexdigest(),count=d['count']))
 for entry in d.get('parts',[]):
  p=posixpath.normpath(posixpath.join(posixpath.dirname(path),entry['path']));raw=subprocess.check_output(['git','show',ref+':'+p]);assert hashlib.sha256(raw).hexdigest()==entry['sha256'];data=gzip.decompress(raw);parts.append(np.frombuffer(data,dtype='S32'))
 for entry in d.get('bases',[]):
  p=posixpath.normpath(posixpath.join(posixpath.dirname(path),entry['path']));raw=subprocess.check_output(['git','show',ref+':'+p]);assert hashlib.sha256(raw).hexdigest()==entry['sha256'];old_index(ref,p,seen,parts,refs)
def audit():
 start=time.perf_counter();completion=json.loads((R/'training-complete.json').read_text());end=completion['common_iteration'];forbidden=set(json.loads((R/'excluded-starts.json').read_text())['states']);new=set();reports=[];auxiliary=[]
 # Conservative exclusion also includes this task's equal-update pilot and the preceding 4ch pilot.
 prior_path='artifacts/ppo-four-channel-runtime/pilot/games.jsonl';prior_raw=subprocess.check_output(['git','show','69e4c7b:'+prior_path])
 aux_sources=[('69e4c7b:'+prior_path,prior_raw),(str(R/'optimization-pilot-history.jsonl.gz'),gzip.decompress((R/'optimization-pilot-history.jsonl.gz').read_bytes()))]
 for name,raw in aux_sources:
  count=0
  for line in raw.decode().splitlines():
   for game in json.loads(line)['games']:
    state=State();new.add(bytes.fromhex(state_hash(state)))
    for row in game['prefix']+game['moves']:
     play(state,Move(*row) if row else None);new.add(bytes.fromhex(state_hash(state)))
    assert state.done and game['scores']==[state.score(0),state.score(1)];count+=1
  auxiliary.append(dict(source=name,uncompressed_sha256=hashlib.sha256(raw).hexdigest(),games=count))
 for seed in completion['seeds']:
  p=R/f'seed-{seed}';ck=torch.load(p/f'iteration-{end}.pt',map_location='cpu',weights_only=False);assert ck['channels']==4 and ck['iteration']==end
  for filename,expected in ck['source_hashes'].items():assert sha(Path('blokus')/Path(filename).name)==expected
  assert ck['exclusion_sha256']==sha(R/'excluded-starts.json') and ck['update_mode']=='fast'
  ng=0;counts=Counter();targets_count=Counter();incidental=set();checks=0
  for entry in ck['history_chain']:
   hp=p/entry['path'];assert sha(hp)==entry['sha256'];first=ng
   with gzip.open(hp,'rt') as stream:
    for line in stream:
     row=json.loads(line);iteration=row['iteration'];games=row['games'];assert iteration==ng//16+1 and len(games)==16;log=ck['logs'][iteration-1];assert hashlib.sha256(json.dumps(games,sort_keys=True).encode()).hexdigest()==log['games_sha256']
     assert all(np.isfinite(v) for v in log['update'].values())
     for slot,g in enumerate(games):
      assert g['id']==seed*100000+(iteration-1)*16+slot and g['opponent']==pool_modes(iteration-1,16)[slot] and g['learner_seat']==(slot//4)%2;s=State();targets={t['ply']:t for t in g['targets']};assert len(targets)==len(g['targets']);counts[g['opponent']]+=1
      for ply,row in enumerate(g['prefix']+g['moves']):
       h=state_hash(s);new.add(bytes.fromhex(h));m=Move(*row) if row else None
       if ply>=len(g['prefix']):
        if h in forbidden:incidental.add(h)
        learner=g['opponent']=='current' or s.turn==g['learner_seat'];assert (ply in targets)==learner
       if ply in targets:
        t=targets[ply];assert t['player']==s.turn and t['state_hash']==h and t['return']==float(np.sign(g['scores'][s.turn]-g['scores'][1-s.turn]));targets_count[g['opponent']]+=1
        if checks%256==0:
         qs=legal_moves(s);assert (qs[t['action']].move if qs else None)==m
        checks+=1
       assert is_legal(s,m) if m else not legal_moves(s);play(s,m,validate=False)
       if ply<len(g['prefix']):assert state_hash(s) not in forbidden
      new.add(bytes.fromhex(state_hash(s)));assert s.done and g['scores']==[s.score(0),s.score(1)];ng+=1
   assert ng-first==entry['games']
  assert ng==end*16;reports.append(dict(seed=seed,games=ng,checkpoint_sha256=sha(p/f'iteration-{end}.pt'),opponent_games=dict(counts),learner_states=dict(targets_count),incidental_old_eval_start_overlap=len(incidental)));print('audited',seed,ng,flush=True)
 new_array=np.array(sorted(new),dtype='S32');del new;parts=[new_array];refs=[];old_index('a011c78','artifacts/ppo-short-search/training-states/index.json',set(),parts,refs)
 assert sum(len(part) for part in parts[1:])==refs[0]['count'], 'historical index coverage mismatch'
 union=np.unique(np.concatenate(parts));np.save(CACHE,union);write(R/'training-audit.json',dict(pass_=True,rows=reports,new_and_auxiliary_unique_states=len(new_array),auxiliary_sources=auxiliary,union_states=len(union),union_sha256=hashlib.sha256(union.tobytes()).hexdigest(),historical_indices=refs,cache=str(CACHE),seconds=time.perf_counter()-start,action_indices_sample_stride=256,all_moves_legally_replayed=True));print('AUDIT PASS',len(union),flush=True)
if __name__=='__main__':audit()
