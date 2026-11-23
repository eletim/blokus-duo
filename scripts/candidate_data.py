"""Phase-disjoint deterministic games and fixed-root teacher labels."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json,gzip,copy,random,time,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
import placement_experiment as e
from candidate_pilot import R,raw
from blokus.gen4 import load
from blokus.engine import State,play,legal_moves
from blokus.ppo_runtime import state_hash
from blokus.ppo_policy import collate
from blokus.candidate_teacher import RootWideSearch
SEEDS=e.SEEDS
OLD=Path('/home/eletim/blokus-value-placement/artifacts/value-placement-49152')
def save(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d));tmp.replace(p)
def history(ms):
 s=State();out=[]
 for i,m in enumerate(ms):
  if i>=8:out.append(state_hash(s))
  play(s,e.Move(*m) if m else None,validate=True)
 return out,s

def prepare():
 if (R/'bank-manifest.json').exists():return
 start=time.perf_counter();torch.set_num_threads(1);ex=set();sources={}
 for p in [OLD/'exclusions.json.gz',OLD/'train-visited.json.gz',OLD/'dev-visited.json.gz']:
  ex.update(json.load(gzip.open(p,'rt'))['states']);sources[str(p)]=e.sha(p)
 for stage in (36864,49152):
  for phase in ('validation','confirmation'):
   bp=OLD/f'n{stage}'/f'{phase}-bank.json'
   if not bp.exists():continue
   bank=json.loads(bp.read_text())['prefixes'];sources[str(bp)]=e.sha(bp)
   ex.update(x['state_hash'] for x in bank)
   for p in bp.parent.glob(f'{phase}-??-*.json.gz'):
    sources[str(p)]=e.sha(p)
    for g in json.load(gzip.open(p,'rt'))['games']:ex.update(history(bank[g['prefix']]['moves']+g['moves'])[0])
 for p in (R/'pilot').glob('*.json'):ex.update(history(json.loads(p.read_text())['moves'])[0])
 idx=np.load(e.INDEX,mmap_mode='r');assert hashlib.sha256(memoryview(idx).cast('B')).hexdigest()=='cdd1f0f1ac7a0e722a670fb541ce904187949c6982cd03fda1fa7acad30f7b4c'
 baseline=len(ex);counts={};excluded=0
 for phase,n,base in [('diagnosis',16,923200000000),('train',256,923300000000),('dev',32,923400000000)]:
  bank=[]
  for seed in SEEDS:
   p,_=load(seed,'native')
   for gid in range(n):
    rng=random.Random(base+seed*1000+gid);s=State();ms=[];positions={};visited=[]
    while not s.done:
     ply=len(ms)
     if ply>=8:visited.append(state_hash(s))
     if ply in (12,13,20,21,28,29):positions[ply]=copy.deepcopy(s)
     qs=legal_moves(s);m=(rng.choice(qs).move if qs else None) if ply<8 else p.act(s,rng)
     play(s,m,validate=True);ms.append(raw(m))
    selected=[];omissions=[]
    for stage,j in enumerate((12,20,28)):
     zs=[positions.get(k) for k in (j,j+1)]
     valid=all(z is not None and len(legal_moves(z))>=2 and state_hash(z) not in ex and not e.contains(idx,state_hash(z)) for z in zs)
     if valid:
      assert {z.turn for z in zs}=={0,1}
      for k,z in zip((j,j+1),zs):selected.append(dict(ply=k,stage=stage,turn=z.turn,state_hash=state_hash(z)));counts[f'{phase}/{seed}/{stage}/{z.turn}']=counts.get(f'{phase}/{seed}/{stage}/{z.turn}',0)+1
     else:omissions.append(stage);excluded+=1
    bank.append(dict(seed=seed,id=gid,phase=phase,rng_seed=base+seed*1000+gid,moves=ms,scores=[s.score(0),s.score(1)],selected=selected,omitted_stages=omissions))
    ex.update(visited)
   print('PREPARED',phase,seed,len(bank),flush=True)
  save(R/f'{phase}-bank.json',dict(games=bank))
 # New heldout starts, fixed before any model fitting or match outcome.
 for phase,n,base in [('validation',16,923500000000),('confirmation',32,923600000000)]:
  rng=random.Random(base);bank=[]
  for i in range(n):
   pr=e.prefix(rng,idx,ex,8+i%3*2);ex.add(pr['state_hash']);bank.append(pr)
  save(R/f'{phase}-bank.json',dict(prefixes=bank))
 for k,v in counts.items():
  parts=k.split('/');parts[-1]=str(1-int(parts[-1]));assert counts['/'.join(parts)]==v
 with gzip.open(R/'excluded-states.json.gz','wt') as h:json.dump(dict(states=sorted(ex)),h)
 files=[R/f'{phase}-bank.json' for phase in ('diagnosis','train','dev','validation','confirmation')]
 save(R/'bank-manifest.json',dict(files={str(p):e.sha(p) for p in files},sources=sources,prior_states=baseline,final_excluded_states=len(ex),paired_counts=counts,omitted_pairs=excluded,index_sha=hashlib.sha256(memoryview(idx).cast('B')).hexdigest(),seconds=time.perf_counter()-start))

def label(job):
 phase,g=job;torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 dest=R/phase/f'{g["seed"]}-{g["id"]}.json'
 if dest.exists():return dict(resumed=True)
 p,v=load(g['seed'],'native');start=time.perf_counter();rows=[]
 for item in g['selected']:
  z=e.state(g['moves'][:item['ply']]);assert state_hash(z)==item['state_hash'];obs,qs=p._observation(z)
  with torch.inference_mode():l,_=p.model(*collate([obs],'cpu'))
  roots=[raw(qs[i].move) for i in l[0].argsort(descending=True).tolist()[:12]];results={}
  for name,width,depth in [('wide',12,6)]+([('narrow',4,6),('shallow',12,4)] if phase=='diagnosis' else []):
   a=RootWideSearch(p,60000,width=4,max_depth=depth,value_evaluator=v,root_width=width);m=a.act(z,random.Random(0));assert a.stats['depth']==depth
   results[name]=dict(move=raw(m),**a.stats)
  top4=roots[:4];greedy=raw(p.act(z,random.Random(0)))
  if greedy not in top4:top4[-1]=greedy
  roots=results['wide']['root_moves'];assert results['wide']['move']==roots[max(range(len(roots)),key=results['wide']['root_scores'].__getitem__)]
  rows.append(dict(**item,roots=roots,top4=top4,results=results,omitted=results['wide']['move'] not in top4))
 save(dest,dict(**g,rows=rows,seconds=time.perf_counter()-start));return dict(seed=g['seed'],id=g['id'],rows=len(rows),seconds=time.perf_counter()-start)
def generate(phase):
 start=time.perf_counter();jobs=[(phase,g) for g in json.loads((R/f'{phase}-bank.json').read_text())['games']]
 with ProcessPoolExecutor(max_workers=4) as pool:
  results=[]
  for i,x in enumerate(pool.map(label,jobs)):
   results.append(x)
   if i%24==0:print('LABEL',phase,i,len(jobs),round(time.perf_counter()-start),flush=True)
 save(R/f'{phase}-generation.json',dict(wall=time.perf_counter()-start,jobs=results))
def diagnose():
 rows=[dict(seed=g['seed'],**r) for p in (R/'diagnosis').glob('*.json') for g in [json.loads(p.read_text())] for r in g['rows']]
 stable=[r for r in rows if r['omitted'] and r['results']['wide']['move']==r['results']['shallow']['move']]
 stats=dict(roots=len(rows),omitted=sum(r['omitted'] for r in rows),stable_omitted=len(stable),stable_seeds=sorted({r['seed'] for r in stable}))
 stats['passed']=stats['omitted']/max(1,len(rows))>=.05 and len(stable)>=5 and len(stats['stable_seeds'])>=2
 stats['by_stage']={str(st):dict(n=sum(r['stage']==st for r in rows),omitted=sum(r['omitted'] and r['stage']==st for r in rows),stable=sum(r['stage']==st for r in stable)) for st in range(3)}
 save(R/'diagnostic-gate.json',stats);print('DIAGNOSIS',stats,flush=True);return stats['passed']
if __name__=='__main__':
 import sys
 if sys.argv[1]=='prepare':prepare()
 else:generate(sys.argv[1]);diagnose() if sys.argv[1]=='diagnosis' else None
