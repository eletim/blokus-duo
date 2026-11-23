"""Four fixed comparisons, common new prefixes, no outcome selection."""
import os,json,random,time,copy
import numpy as np
import torch
from candidate_scale_data import R,OLD,SEEDS,save
import placement_experiment as e
from blokus.gen4 import load
from blokus.ppo_policy import PPOAgent
from blokus.ppo_search_deeper import DeepSearch

def aggregate(total,control):
 folder=R/f'n{total}'/f'vs-{control}';rows=[json.loads(p.read_text()) for p in sorted(folder.glob('*.json'))];assert len(rows)==384
 point=np.array([np.mean([g['point'] for g in rows if g['prefix']==i]) for i in range(64)]);score=np.array([np.mean([g['difference'] for g in rows if g['prefix']==i]) for i in range(64)]);ix=np.random.default_rng(923881).integers(64,size=(20000,64));bp=point[ix].mean(1)
 out=dict(games=384,prefixes=64,wins=sum(g['difference']>0 for g in rows),draws=sum(g['difference']==0 for g in rows),losses=sum(g['difference']<0 for g in rows),point=float(point.mean()),difference=float(score.mean()),point_ci=np.quantile(bp,[.025,.975]).tolist(),point_family_ci=np.quantile(bp,[.00625,.99375]).tolist(),difference_ci=np.quantile(score[ix].mean(1),[.025,.975]).tolist(),prefix_point=point.tolist(),prefix_difference=score.tolist(),family=4)
 for dim,values in [('seed',SEEDS),('seat',(0,1))]:out['by_'+dim]={str(x):dict(point=float(np.mean([g['point'] for g in rows if g[dim]==x])),difference=float(np.mean([g['difference'] for g in rows if g[dim]==x]))) for x in values}
 out['timing']={}
 for who in ['candidate','control']:
  st=[s for g in rows for s in g['stats'] if s['who']==who and not s.get('forced_pass')];ts=np.array([s['elapsed_ms'] for s in st]);out['timing'][who]=dict(n=len(st),mean_ms=float(ts.mean()),p95_ms=float(np.quantile(ts,.95)),max_ms=float(ts.max()),over550=int(sum(ts>550)),depth6=float(np.mean([s.get('depth',0)==6 for s in st])),fallback=sum(s.get('fallback',False) for s in st))
 save(R/f'n{total}'/f'vs-{control}-summary.json',out);return out

def evaluate(total,control):
 torch.set_num_threads(1);os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});start=time.perf_counter();bank=json.loads((R/'evaluation-bank.json').read_text())['prefixes'];warm=[]
 for seed in SEEDS:
  p,v=load(seed,'native');c=PPOAgent(seed,'cpu',checkpoint=R/f'n{total}'/f'policy-{seed}.pt',backend='native')
  if control=='old768':p=PPOAgent(seed,'cpu',checkpoint=OLD/f'policy-{seed}.pt',backend='native')
  aa=DeepSearch(c,500,width=4,max_depth=6,value_evaluator=v);bb=DeepSearch(p,500,width=4,max_depth=6,value_evaluator=v)
  for a in [aa,bb]:
   t=time.perf_counter();a.act(e.state(bank[0]['moves']),random.Random(1));warm.append(time.perf_counter()-t)
  for pi,pr in enumerate(bank):
   for seat in [0,1]:
    dest=R/f'n{total}'/f'vs-{control}'/f'{seed}-{pi}-{seat}.json'
    if dest.exists():continue
    s=e.state(pr['moves']);rng=random.Random(seed*1000+pi*2+seat);ms=[];stats=[]
    while not s.done:
     a=aa if s.turn==seat else bb;t=time.perf_counter();m=a.act(s,rng);elapsed=(time.perf_counter()-t)*1000;st=copy.deepcopy(a.stats);st.pop('root_scores',None);st.update(who='candidate' if s.turn==seat else 'control',player=s.turn,elapsed_ms=elapsed);stats.append(st);e.play(s,m,validate=True);ms.append(e.raw(m))
    scores=[s.score(0),s.score(1)];d=scores[seat]-scores[1-seat];save(dest,dict(seed=seed,prefix=pi,seat=seat,prefix_moves=pr['moves'],moves=ms,scores=scores,difference=d,point=float(d>0)+.5*(d==0),stats=stats))
    if pi%8==0:print('EVAL',total,control,seed,pi,seat,round(time.perf_counter()-start),flush=True)
 out=aggregate(total,control);save(R/f'n{total}'/f'vs-{control}-time.json',dict(seconds=time.perf_counter()-start,warmup_seconds=warm));print('COMPARISON DONE',total,control,out['point'],flush=True)

def finish_summary():
 ix=np.random.default_rng(923881).integers(64,size=(20000,64));res={}
 for control in ['Frozen','old768']:
  a=json.loads((R/'n3072'/f'vs-{control}-summary.json').read_text());b=json.loads((R/'n12288'/f'vs-{control}-summary.json').read_text());delta=np.array(b['prefix_point'])-a['prefix_point'];res[control]=dict(mean=float(delta.mean()),paired_descriptive_ci=np.quantile(delta[ix].mean(1),[.025,.975]).tolist())
 save(R/'quantity-trend.json',res)
 for n in [3072,12288]:
  zs=[json.loads((R/f'n{n}'/f'vs-{c}-summary.json').read_text()) for c in ['Frozen','old768']];save(R/f'n{n}'/'gate.json',dict(passed=all(z['point_family_ci'][0]>.5 and z['difference']>0 for z in zs),rule='both preregistered controls corrected lower>0.5 and score>0'))
