"""Predeclared CPU timed gen4 candidate evaluation and prefix-cluster CIs."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json,copy,random,time
import numpy as np
import torch
import placement_experiment as e
from candidate_data import R,SEEDS,save
from blokus.gen4 import load
from blokus.ppo_policy import PPOAgent
from blokus.ppo_search_deeper import DeepSearch
from blokus.ppo_runtime import state_hash

def aggregate(phase,mode):
 rows=[json.loads(p.read_text()) for p in sorted((R/f'{phase}-{mode}').glob('*.json'))];n=len(json.loads((R/f'{phase}-bank.json').read_text())['prefixes']);assert len(rows)==n*6
 pts=np.array([np.mean([g['point'] for g in rows if g['prefix']==i]) for i in range(n)]);diff=np.array([np.mean([g['difference'] for g in rows if g['prefix']==i]) for i in range(n)]);ix=np.random.default_rng(923771).integers(n,size=(20000,n))
 out=dict(games=len(rows),prefixes=n,wins=sum(g['difference']>0 for g in rows),draws=sum(g['difference']==0 for g in rows),losses=sum(g['difference']<0 for g in rows),point=float(pts.mean()),difference=float(diff.mean()),point_ci=np.quantile(pts[ix].mean(1),[.025,.975]).tolist(),difference_ci=np.quantile(diff[ix].mean(1),[.025,.975]).tolist())
 for dim,values in [('seed',SEEDS),('seat',(0,1))]:out['by_'+dim]={str(x):dict(point=float(np.mean([r['point'] for r in rows if r[dim]==x])),difference=float(np.mean([r['difference'] for r in rows if r[dim]==x]))) for x in values}
 out['timing']={}
 for who in ('candidate','frozen'):
  st=[s for r in rows for s in r['stats'] if s['who']==who and not s.get('forced_pass')];times=[s['elapsed_ms'] for s in st]
  out['timing'][who]=dict(n=len(st),mean_ms=float(np.mean(times)),max_ms=max(times),p95_ms=float(np.quantile(times,.95)),over550=float(np.mean(np.array(times)>550)),depth6=float(np.mean([s.get('depth',0)==6 for s in st])),fallback=sum(s.get('fallback',False) for s in st))
 save(R/f'{phase}-{mode}-summary.json',out);return out

def evaluate(phase):
 torch.set_num_threads(1);os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});start=time.perf_counter();bank=json.loads((R/f'{phase}-bank.json').read_text())['prefixes'];warmups=[]
 forbidden={r['state_hash'] for ph in ('train','dev','diagnosis') for g in json.loads((R/f'{ph}-bank.json').read_text())['games'] for r in g['selected']}
 for mode in (['H','policy'] if phase=='validation' else ['H']):
  for seed in SEEDS:
   frozen,value=load(seed,'native');candidate=PPOAgent(seed,'cpu',checkpoint=R/f'policy-{seed}.pt',backend='native')
   aa=DeepSearch(candidate,500,width=4,max_depth=6,value_evaluator=value) if mode=='H' else candidate
   bb=DeepSearch(frozen,500,width=4,max_depth=6,value_evaluator=value) if mode=='H' else frozen
   for a in (aa,bb):
    t=time.perf_counter();a.act(e.state(bank[0]['moves']),random.Random(1));warmups.append(time.perf_counter()-t)
   for pi,pr in enumerate(bank):
    for seat in (0,1):
     dest=R/f'{phase}-{mode}'/f'{seed}-{pi}-{seat}.json'
     if dest.exists():continue
     s=e.state(pr['moves']);ms=[];stats=[];rng=random.Random(seed*1000+pi*2+seat)
     while not s.done:
      assert state_hash(s) not in forbidden,'learning/evaluation state collision'
      a=aa if s.turn==seat else bb;t=time.perf_counter();m=a.act(s,rng);elapsed=(time.perf_counter()-t)*1000
      st=copy.deepcopy(a.stats) if mode=='H' else {};st.pop('root_scores',None);st.update(who='candidate' if s.turn==seat else 'frozen',player=s.turn,elapsed_ms=elapsed);stats.append(st);e.play(s,m,validate=True);ms.append(e.raw(m))
     scores=[s.score(0),s.score(1)];d=scores[seat]-scores[1-seat];save(dest,dict(seed=seed,prefix=pi,seat=seat,prefix_moves=pr['moves'],moves=ms,scores=scores,difference=d,point=float(d>0)+.5*(d==0),stats=stats));print('EVAL',phase,mode,seed,pi,seat,round(time.perf_counter()-start),flush=True)
  out=aggregate(phase,mode)
 h=json.loads((R/f'{phase}-H-summary.json').read_text())
 if phase=='validation':
  middle=json.loads((R/'selection.json').read_text())['middle_regression'];passed=h['point']>.5 and h['difference']>0 and middle>=-.05 and h['timing']['candidate']['over550']<=.01
 else:passed=h['point_ci'][0]>.5 and h['difference']>0
 save(R/f'{phase}-gate.json',dict(passed=bool(passed),results=h));save(R/f'{phase}-time.json',dict(seconds=time.perf_counter()-start,warmup_seconds=warmups));print('GATE',phase,passed,h['point'],flush=True)
 return passed
if __name__=='__main__':
 import sys
 evaluate(sys.argv[1])
