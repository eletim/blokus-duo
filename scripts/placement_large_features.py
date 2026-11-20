"""Post-generation independent replay and matched A/C feature shards."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import concurrent.futures,collections,gzip,json,time
from pathlib import Path
import numpy as np
import placement_experiment as e
from audit_placement import replay
from blokus.engine import State,Move,play
from blokus.ppo_runtime import state_hash
from blokus.value_search import encode
from blokus.placement_value import mobility
R=Path('artifacts/value-placement-24576');OLD=Path('artifacts/value-placement-6144')
def readgz(p):return json.load(gzip.open(p,'rt'))
def writegz(p,x):
 with gzip.open(p,'wt') as h:json.dump(x,h,separators=(',',':'))
def init():
 global EX,IDX
 e.setup();EX=set(readgz(R/'exclusions.json.gz')['states']);IDX=np.load(e.INDEX,mmap_mode='r')
def worker(item):
 job,tasks=item;root=R/'feature-shards';out=root/f'{job:03d}';done=out.with_suffix('.audit.json')
 if done.exists():
  d=json.loads(done.read_text());assert d['tasks']==[(t['seed'],t['id']) for t in tasks] or d['tasks']==[[t['seed'],t['id']] for t in tasks]
  for path,h in d['files'].items():assert e.sha(path)==h
  return d
 start=time.perf_counter();rows=[];labels=[];seen=set();games=[];stage=collections.Counter();depth=collections.Counter();scores=[]
 for task in tasks:
  path=R/'games'/f"{task['seed']}-{task['id']}.json.gz";g=readgz(path);assert g['task']==task;assert replay(g['moves'])==g['scores'];z=State();local=[]
  assert len(g['stats'])==len(g['moves'])-8
  for stat in g['stats']:
   assert stat['forced_pass'] or (stat['depth']==6 and stat['stages']==[2,4,6] and not stat['fallback']);depth[str(stat['depth'])]+=1
  for ply,m in enumerate(g['moves']):
   if ply>=8:
    h=state_hash(z);assert h not in EX and not e.contains(IDX,h),(task['seed'],task['id'],ply,'excluded state')
    seen.add(h);b,st=encode(z);y=float(np.sign(g['scores'][z.turn]-g['scores'][1-z.turn]));extra=mobility(z)
    label=dict(seed=task['seed'],game=task['id'],ply=ply,turn=z.turn,hash=h,target=y,phase=task['phase'])
    local.append((b[:4],st,extra,y,label));stage[f"{task['seed']}/{z.turn}/{min(2,ply//16)}"]+=1
   play(z,Move(*m) if m else None,validate=False)
  assert z.done and [z.score(0),z.score(1)]==g['scores'];terminal=state_hash(z);assert terminal not in EX and not e.contains(IDX,terminal);seen.add(terminal)
  for b,st,extra,y,label in local:rows.append((b,st,extra,y,1/len(local)));labels.append(label)
  games.append(dict(seed=task['seed'],id=task['id'],phase=task['phase'],sha256=e.sha(path),labels=len(local)));scores.append(g['scores'])
 np.savez_compressed(out.with_suffix('.npz'),**{f'x{i}':np.stack([r[i] for r in rows]).astype(np.float32) for i in range(5)})
 writegz(out.with_suffix('.labels.json.gz'),dict(labels=labels,visited=sorted(seen)))
 d=dict(tasks=[(t['seed'],t['id']) for t in tasks],phase=tasks[0]['phase'],games=games,labels=len(labels),visited=len(seen),stage_turn_counts=dict(stage),depth_counts=dict(depth),seconds=time.perf_counter()-start,files={str(p):e.sha(p) for p in (out.with_suffix('.npz'),out.with_suffix('.labels.json.gz'))})
 e.write(done,d);print('FEATURE/AUDIT',job,len(games),len(labels),flush=True);return d

def main():
 e.setup();start=time.perf_counter();root=R/'feature-shards';root.mkdir(exist_ok=True);bank=json.loads((R/'bank.json').read_text());jobs=[]
 for phase in ('train','dev'):
  for seed in e.SEEDS:
   tasks=[t for t in bank['games'] if t['phase']==phase and t['seed']==seed]
   for i in range(0,len(tasks),256):jobs.append((len(jobs),tasks[i:i+256]))
 with concurrent.futures.ProcessPoolExecutor(max_workers=8,initializer=init) as pool:results=list(pool.map(worker,jobs))
 sets={'train':set(),'dev':set()};files={'train':[],'dev':[]};counts=collections.Counter();allgames=set()
 for job,d in enumerate(results):
  labels=readgz(root/f'{job:03d}.labels.json.gz');sets[d['phase']].update(labels['visited']);files[d['phase']].append(str(root/f'{job:03d}.npz'));counts[d['phase']]+=len(d['games'])
  for g in d['games']:
   key=(g['phase'],g['seed'],g['id']);assert key not in allgames;allgames.add(key)
 assert counts=={'train':18432,'dev':192} and not sets['train']&sets['dev']
 old=json.loads((OLD/'train-labels.json').read_text());oldseen=set(old['visited']);assert not oldseen&(sets['train']|sets['dev'])
 oldaudit=json.loads((OLD/'audit.json').read_text());assert oldaudit['pass_'];oldfeatures=OLD/'train-features.npz';data=np.load(oldfeatures);assert len(data['x0'])==len(old['labels'])
 for phase,seen in sets.items():writegz(R/f'{phase}-visited.json.gz',dict(states=sorted(seen)))
 manifest=dict(train_files=[str(oldfeatures)]+files['train'],dev_files=files['dev'],file_shas={p:e.sha(p) for p in [str(oldfeatures)]+files['train']+files['dev']},old_labels_sha=e.sha(OLD/'train-labels.json'),old_audit_sha=e.sha(OLD/'audit.json'),old_training_games=6144,new_training_games=18432,training_games=24576,dev_games=192,old_labels=len(data['x0']),new_labels=sum(d['labels'] for d in results if d['phase']=='train'),dev_labels=sum(d['labels'] for d in results if d['phase']=='dev'),source_bank_sha=e.sha(R/'bank.json'),shards=results,seconds=time.perf_counter()-start)
 e.write(R/'data-manifest.json',manifest);print('DATA READY',manifest['old_labels']+manifest['new_labels'],flush=True)
if __name__=='__main__':main()
