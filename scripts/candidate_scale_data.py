"""Nested data-volume experiment, immutable old labels and independent new games."""
import os,json,gzip,copy,random,time,hashlib,shutil,multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
import placement_experiment as e
import candidate_data as old
from blokus.gen4 import load
from blokus.engine import State,play,legal_moves
from blokus.ppo_runtime import state_hash
R=Path('artifacts/gen4-candidate-scale');OLD=Path('artifacts/gen4-candidate-policy');SEEDS=old.SEEDS;save=old.save

def prepare():
 if (R/'bank-manifest.json').exists():return
 start=time.perf_counter();torch.set_num_threads(1)
 for p,h in json.loads((OLD/'learning-manifest.json').read_text()).items():assert e.sha(p)==h,p
 ex=set(json.load(gzip.open(OLD/'excluded-states.json.gz','rt'))['states'])
 for phase in ('validation-H','validation-policy','confirmation-H'):
  for p in (OLD/phase).glob('*.json'):
   g=json.loads(p.read_text());ex.update(old.history(g['prefix_moves']+g['moves'])[0])
 initial=len(ex);idx=np.load(e.INDEX,mmap_mode='r');assert hashlib.sha256(memoryview(idx).cast('B')).hexdigest()=='cdd1f0f1ac7a0e722a670fb541ce904187949c6982cd03fda1fa7acad30f7b4c'
 prior=json.loads((OLD/'train-bank.json').read_text())['games'];games=[]
 for seed in SEEDS:
  p,_=load(seed,'native')
  for gid in range(256,4096):
   path=R/'source-games'/f'{seed}-{gid}.json'
   if path.exists():
    g=json.loads(path.read_text());ex.update(old.history(g['moves'])[0]);games.append(g);continue
   rng=random.Random(923700000000+seed*10000+gid);s=State();ms=[];positions={};visited=[]
   while not s.done:
    ply=len(ms)
    if ply>=8:visited.append(state_hash(s))
    if ply in (12,13,20,21,28,29):positions[ply]=copy.deepcopy(s)
    q=legal_moves(s);m=(rng.choice(q).move if q else None) if ply<8 else p.act(s,rng);play(s,m,validate=True);ms.append(e.raw(m))
   selected=[];omissions=[]
   for stage,j in enumerate((12,20,28)):
    zs=[positions.get(k) for k in (j,j+1)]
    if all(z is not None and len(legal_moves(z))>=2 and state_hash(z) not in ex and not e.contains(idx,state_hash(z)) for z in zs):
     assert {z.turn for z in zs}=={0,1}
     for k,z in zip((j,j+1),zs):selected.append(dict(ply=k,stage=stage,turn=z.turn,state_hash=state_hash(z)))
    else:omissions.append(stage)
   g=dict(seed=seed,id=gid,phase='train',rng_seed=923700000000+seed*10000+gid,moves=ms,scores=[s.score(0),s.score(1)],selected=selected,omitted_stages=omissions);save(path,g);games.append(g);ex.update(visited)
   if gid%256==0:print('PREPARE',seed,gid,flush=True)
 save(R/'new-bank.json',dict(games=games));allgames=prior+games
 for total,perseed in [(3072,1024),(12288,4096)]:
  dest=R/f'n{total}';dest.mkdir(exist_ok=True);subset=[g for g in allgames if g['id']<perseed];assert len(subset)==total
  save(dest/'train-bank.json',dict(games=subset));shutil.copyfile(OLD/'dev-bank.json',dest/'dev-bank.json');shutil.copyfile(R/'PLAN.md',dest/'PLAN.md');shutil.copyfile(OLD/'diagnostic-gate.json',dest/'diagnostic-gate.json')
  for phase in ['train','dev']:
   rows=json.loads((dest/f'{phase}-bank.json').read_text())['games']
   for seed in SEEDS:
    for stage in range(3):assert sum(r['turn']==0 for g in rows if g['seed']==seed for r in g['selected'] if r['stage']==stage)==sum(r['turn']==1 for g in rows if g['seed']==seed for r in g['selected'] if r['stage']==stage)
 rng=random.Random(923800000000);bank=[]
 for i in range(64):
  z=e.prefix(rng,idx,ex,8+i%3*2);bank.append(z);ex.add(z['state_hash'])
 save(R/'evaluation-bank.json',dict(prefixes=bank))
 with gzip.open(R/'excluded-states.json.gz','wt') as f:json.dump(dict(states=sorted(ex)),f)
 files=[R/'PLAN.md',R/'new-bank.json',R/'evaluation-bank.json',OLD/'learning-manifest.json',Path('scripts/candidate_scale_data.py'),Path('scripts/candidate_scale_pipeline.py'),Path('scripts/candidate_scale_eval.py')]+[R/f'n{n}'/f'{p}-bank.json' for n in [3072,12288] for p in ['train','dev']]
 manifest=dict(files={str(p):e.sha(p) for p in files},prior_excluded=initial,final_excluded=len(ex),new_games=len(games),nested_old_games=len(prior),seconds=time.perf_counter()-start)
 for n in [3072,12288]:save(R/f'n{n}'/'bank-manifest.json',manifest)
 save(R/'bank-manifest.json',manifest);print('PREPARE DONE',len(games),flush=True)

def label(g):
 old.R=R;return old.label(('train',g))

def generate(total):
 start=time.perf_counter();games=json.loads((R/f'n{total}'/'train-bank.json').read_text())['games'];jobs=[g for g in games if g['id']>=256 and not (R/'train'/f'{g["seed"]}-{g["id"]}.json').exists()]
 with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context("spawn")) as pool:
  for i,result in enumerate(pool.map(label,jobs)):
   if i%64==0:print('LABEL',total,i,len(jobs),round(time.perf_counter()-start),flush=True)
 dest=R/f'n{total}'
 for phase,gs in [('train',games),('dev',json.loads((OLD/'dev-bank.json').read_text())['games'])]:
  (dest/phase).mkdir(exist_ok=True)
  for g in gs:
   name=f'{g["seed"]}-{g["id"]}.json';src=(OLD if phase=='dev' or g['id']<256 else R)/phase/name;out=dest/phase/name
   if out.exists():assert e.sha(src)==e.sha(out)
   else:os.link(src,out)
 save(dest/'generation.json',dict(new_jobs=len(jobs),seconds=time.perf_counter()-start,teacher='unchanged root12 inner4 completed6; fixed gen4 value;4CPUworkers'))
if __name__=='__main__':prepare()
