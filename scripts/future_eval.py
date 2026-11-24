from future_common import *
from future_data import exclusions
from placement_experiment import contains,INDEX
from blokus.future_occupancy import FutureValue,FutureEvaluator
PAIRS=[('A','G'),('B','G'),('C','G'),('C','A'),('C','B')]
def model(seed,arm):
 p,v=load(seed,'native')
 if arm=='G':return p,v
 ck=torch.load(R/f'{arm}-{seed}.pt',map_location='cpu',weights_only=True);m=FutureValue(v.model,arm);m.load_state_dict(ck['model']);return p,FutureEvaluator(m)
def bank():
 if (R/'eval-bank.json').exists():return json.loads((R/'eval-bank.json').read_text())
 ex=exclusions();new=json.load(gzip.open(R/'new-exclusions.json.gz','rt'));ex.update(new['exact']);cs=set(new['canonical']);idx=np.load(INDEX,mmap_mode='r');rng=random.Random(924400000);out=[]
 while len(out)<64:
  ms=prefix(rng,8+2*(len(out)%3));s=state(ms);h=state_hash(s);c=canon(s)
  if h in ex or c in cs or contains(idx,h):continue
  out.append(dict(moves=ms,hash=h,canonical=c));ex.add(h);cs.add(c)
 save(R/'eval-bank.json',out);return out

def aggregate(pair):
 gs=[json.loads(p.read_text()) for p in (R/'evaluation'/pair).glob('*.json')];assert len(gs)==384
 points=np.array([[g['point'] for g in gs if g['prefix']==i] for i in range(64)]).mean(1);diff=np.array([[g['difference'] for g in gs if g['prefix']==i] for i in range(64)]).mean(1)
 rng=np.random.default_rng(924481);ids=rng.integers(0,64,(20000,64));bp=points[ids].mean(1);bd=diff[ids].mean(1);timing={}
 for side in ['candidate','control']:
  stats=[s for g in gs for s in g['stats'] if (s['player']==g['seat'])==(side=='candidate')];ms=[s['elapsed_ms'] for s in stats];timing[side]=dict(n=len(ms),mean_ms=float(np.mean(ms)),p95_ms=float(np.percentile(ms,95)),max_ms=max(ms),depth6=sum(s['depth']==6 for s in stats)/len(stats),fallback=sum(s['fallback'] for s in stats),over550=sum(t>550 for t in ms))
 d=dict(games=384,prefixes=64,wins=sum(g['point']==1 for g in gs),draws=sum(g['point']==.5 for g in gs),losses=sum(g['point']==0 for g in gs),point=float(points.mean()),difference=float(diff.mean()),point_ci=np.quantile(bp,[.025,.975]).tolist(),family_ci=np.quantile(bp,[.005,.995]).tolist(),difference_ci=np.quantile(bd,[.025,.975]).tolist(),prefix_point=points.tolist(),prefix_difference=diff.tolist(),timing=timing,by_seed={str(seed):float(np.mean([g['point'] for g in gs if g['seed']==seed])) for seed in SEEDS},by_seat={str(s):float(np.mean([g['point'] for g in gs if g['seat']==s])) for s in [0,1]})
 save(R/f'eval-{pair}.json',d);return d

def evaluate():
 torch.set_num_threads(1);os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});pr=bank();start=time.perf_counter()
 for left,right in PAIRS:
  pair=left+right
  if (R/f'eval-{pair}.json').exists():continue
  st=time.perf_counter()
  for seed in SEEDS:
   lp,lv=model(seed,left);rp,rv=model(seed,right);agents=[DeepSearch(lp,500,width=4,max_depth=6,value_evaluator=lv),DeepSearch(rp,500,width=4,max_depth=6,value_evaluator=rv)]
   for i,z in enumerate(pr):
    for seat in [0,1]:
     path=R/'evaluation'/pair/f'{seed}-{i}-{seat}.json'
     if path.exists():continue
     s=state(z['moves']);rng=random.Random(924500000+seed*1000+i*2+seat);moves=[];stats=[]
     while not s.done:
      agent=agents[0 if s.turn==seat else 1];m=agent.act(s,rng);stats.append(dict(agent.stats,player=s.turn));play(s,m,validate=True);moves.append(raw(m))
     scores=[s.score(0),s.score(1)];d=scores[seat]-scores[1-seat];save(path,dict(seed=seed,prefix=i,seat=seat,pair=pair,moves=moves,stats=stats,boards=s.boards,scores=scores,difference=d,point=float(d>0)+.5*(d==0)))
    if i%16==0:print('EVAL',pair,seed,i,round(time.perf_counter()-st),flush=True)
  aggregate(pair);save(R/f'eval-{pair}-time.json',dict(seconds=time.perf_counter()-st))
 save(R/'evaluation-done.json',dict(seconds=time.perf_counter()-start,comparisons=5,games=1920))
if __name__=='__main__':evaluate()
