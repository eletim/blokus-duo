"""Finite deterministic candidate-width diagnostic; no training or production writes."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,copy,json,time,random,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import torch
from blokus.gen4 import load
from blokus.ppo_search_deeper import DeepSearch,key
from blokus.candidate_teacher import RootWideSearch
from blokus.engine import State,play,legal_moves
from blokus.ppo_policy import collate
from blokus.ppo_runtime import state_hash
R=Path('artifacts/gen4-candidate-policy')
def raw(m):return [m.piece,m.orientation,m.row,m.col] if m is not None else None
def task(args):
 seed,gid,base,phase=args;torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 p,v=load(seed,'native');s=State();rng=random.Random(base+seed*1000+gid);ms=[];positions={};start=time.perf_counter()
 while not s.done:
  ply=len(ms)
  if ply in (12,13,20,21,28,29):positions[ply]=copy.deepcopy(s)
  qs=legal_moves(s);m=(rng.choice(qs).move if qs else None) if ply<8 else p.act(s,rng)
  play(s,m,validate=True);ms.append(raw(m))
 rows=[]
 for stage,ply in enumerate((12,20,28)):
  if any(j not in positions or len(legal_moves(positions[j]))<2 for j in (ply,ply+1)):continue
  for j in (ply,ply+1):
   z=positions[j];obs,qs=p._observation(z)
   with torch.inference_mode():logits,_=p.model(*collate([obs],'cpu'))
   order=logits[0].argsort(descending=True).tolist();roots=[raw(qs[i].move) for i in order[:12]];results={}
   for name,width,depth in [('narrow',4,6),('wide',12,6),('shallow',12,4)]:
    a=RootWideSearch(p,60000,width=4,max_depth=depth,value_evaluator=v,root_width=width)
    m=a.act(z,random.Random(0));assert a.stats['depth']==depth,(seed,gid,j,name,a.stats)
    results[name]=dict(move=raw(m),**a.stats)
   rows.append(dict(ply=j,stage=stage,turn=z.turn,state_hash=state_hash(z),roots=roots,top4=roots[:4],results=results,omitted=results['wide']['move'] not in roots[:4],stable=results['wide']['move']==results['shallow']['move']))
 out=dict(seed=seed,id=gid,phase=phase,rng_seed=base+seed*1000+gid,moves=ms,scores=[s.score(0),s.score(1)],rows=rows,seconds=time.perf_counter()-start)
 dest=R/phase/f'{seed}-{gid}.json';dest.parent.mkdir(exist_ok=True);tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(out));tmp.replace(dest)
 return dict(seed=seed,id=gid,rows=len(rows),seconds=out['seconds'],omitted=sum(x['omitted'] for x in rows),stable_omitted=sum(x['omitted'] and x['stable'] for x in rows))
def oracle():
 torch.set_num_threads(1);p,v=load(1751401,'native');s=State();rng=random.Random(78821)
 for _ in range(25):
  if s.done:break
  q=legal_moves(s);play(s,rng.choice(q).move if q else None,validate=True)
 a=DeepSearch(p,60000,width=4,max_depth=6,value_evaluator=v);b=RootWideSearch(p,60000,width=4,max_depth=6,value_evaluator=v,root_width=4)
 assert a.act(s,random.Random(0))==b.act(s,random.Random(0))
 for k in ('root_scores','depth','nodes','cutoffs'):assert a.stats[k]==b.stats[k],k
 (R/'oracle.json').write_text(json.dumps(dict(equal=True,stats=a.stats)))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--games',type=int,default=2);ap.add_argument('--phase',default='pilot');ap.add_argument('--base',type=int,default=923100000000);a=ap.parse_args();oracle();start=time.perf_counter()
 jobs=[(s,i,a.base,a.phase) for s in (1751401,1751402,1751403) for i in range(a.games) if not (R/a.phase/f'{s}-{i}.json').exists()]
 with ProcessPoolExecutor(max_workers=4) as ex:
  results=list(ex.map(task,jobs));[print(x,flush=True) for x in results]
 (R/f'{a.phase}-cost.json').write_text(json.dumps(dict(wall=time.perf_counter()-start,results=results),indent=2))
if __name__=='__main__':main()
