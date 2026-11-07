"""Frozen endpoint, paired new banks; game-cluster confidence intervals."""
from h_distill import *

def prepare():
 forbidden=exclusions();idx=np.load(INDEX,mmap_mode='r');new=set()
 for phase in ('pilot','train','dev'):
  for seed in SEEDS:
   for g in json.loads((R/f'{phase}-{seed}.json').read_text())['games']:new.update(g['visited'])
 forbidden|=new
 for phase,n,sd in [('validation',12,38400100),('confirmation',32,38500100)]:
  path=R/f'{phase}-bank.json'
  if path.exists():continue
  rng=random.Random(sd);bank=[]
  while len(bank)<n:
   s=State();ms=[]
   for _ in range(rng.randint(6,10)):
    qs=legal_moves(s);m=rng.choice(qs).move if qs else None;play(s,m,validate=True);ms.append(raw(m))
   h=state_hash(s)
   if h in forbidden or contains(idx,h):continue
   forbidden.add(h);bank.append(dict(id=len(bank),moves=ms,state_hash=h))
  write(path,dict(seed=sd,prefixes=bank))
 write(R/'models.json',{str(seed):dict(original_sha=sha(f'artifacts/ppo-gen3/seed-{seed}.pt'),student_sha=sha(R/f'student-{seed}.pt')) for seed in SEEDS})
 write(R/'pre-evaluation-separation.json',dict(new_training_states=len(new),original_index_sha256='cdd1f0f1ac7a0e722a670fb541ce904187949c6982cd03fda1fa7acad30f7b4c',original_index_path=str(INDEX),new_banks_disjoint=True))

def summary(gs):
 d=np.array([g['difference'] for g in gs]);return dict(games=len(d),wins=int((d>0).sum()),draws=int((d==0).sum()),losses=int((d<0).sum()),point=float((np.sign(d)*.5+.5).mean()),difference=float(d.mean()))
def compare(phase,method,seed):
 path=R/f'{phase}-{method}-{seed}.json'
 if path.exists():return
 bank=json.loads((R/f'{phase}-bank.json').read_text())['prefixes'];p=PPOAgent(seed,'cpu');q=PPOAgent(seed,'cpu',checkpoint=R/f'student-{seed}.pt');models=[q,p];agents=[DeepSearch(x,500) for x in models] if method=='H' else models;games=[];start=time.perf_counter()
 for prefix in bank:
  for seat in (0,1):
   s=state(prefix['moves']);ms=[];times=[];rng=random.Random(38600100+prefix['id']*2+seat)
   while not s.done:
    side=int(s.turn!=seat);a=agents[side];t=time.perf_counter();m=a.act(s,rng);elapsed=(time.perf_counter()-t)*1000
    stat=copy.deepcopy(a.stats) if method=='H' else {};stat.pop('root_scores',None);times.append(dict(side=side,elapsed_ms=elapsed,**{k:v for k,v in stat.items() if k!='elapsed_ms'}));play(s,m,validate=True);ms.append(raw(m))
   games.append(dict(prefix=prefix['id'],seat=seat,moves=ms,scores=[s.score(0),s.score(1)],difference=s.score(seat)-s.score(1-seat),timings=times));print(phase,method,seed,prefix['id'],seat,flush=True)
 write(path,dict(seed=seed,games=games,summary=summary(games),wall_seconds=time.perf_counter()-start))

def aggregate(phase,method):
 rows=[json.loads((R/f'{phase}-{method}-{s}.json').read_text()) for s in SEEDS];gs=[g for r in rows for g in r['games']];groups={}
 for g in gs:groups.setdefault(g['prefix'],[]).append(g['difference'])
 assert all(len(v)==6 for v in groups.values());d=np.array([np.mean(x) for x in groups.values()]);p=np.array([np.mean(np.sign(x)*.5+.5) for x in groups.values()]);ix=np.random.default_rng(38700100).integers(len(p),size=(20000,len(p)));pm=p[ix].mean(1);dm=d[ix].mean(1);timing={}
 for side in (0,1):
  ts=[t for g in gs for t in g['timings'] if t['side']==side];ms=np.array([t['elapsed_ms'] for t in ts]);npas=[t for t in ts if not t.get('forced_pass',False)]
  timing[str(side)]=dict(mean_ms=float(ms.mean()),p99_ms=float(np.quantile(ms,.99)),max_ms=float(ms.max()),over500=int((ms>500).sum()),moves=len(ts),fallback=sum(t.get('fallback',False) for t in ts),depth6_fraction=sum(t.get('depth')==6 for t in npas)/max(1,len(npas)),depths={str(k):sum(t.get('depth',0)==k for t in ts) for k in (0,2,4,6)})
 return dict(point=float(p.mean()),point_ci95=np.quantile(pm,[.025,.975]).tolist(),point_ci_bonferroni2=np.quantile(pm,[.0125,.9875]).tolist(),difference=float(d.mean()),difference_ci95=np.quantile(dm,[.025,.975]).tolist(),by_seed={str(r['seed']):r['summary'] for r in rows},time=timing,wall_seconds=sum(r['wall_seconds'] for r in rows))

def run():
 prepare()
 for method in ('policy','H'):
  for seed in SEEDS:compare('validation',method,seed)
 a=aggregate('validation','H');b=aggregate('validation','policy');write(R/'validation-summary.json',dict(H=a,policy=b))
 gate=a['point']>.5 and a['difference']>0 and sum(v['point']>=.5 for v in a['by_seed'].values())>=2 and b['point_ci95'][1]>=.5 and a['time']['0']['p99_ms']<=750 and a['time']['0']['depth6_fraction']>=.2
 write(R/'selection.json',dict(gate=gate,epoch=8,models_sha=sha(R/'models.json'),reason='predeclared H-first gate; no seed selection'))
 if gate:
  for method in ('policy','H'):
   for seed in SEEDS:compare('confirmation',method,seed)
  write(R/'confirmation-summary.json',{m:aggregate('confirmation',m) for m in ('H','policy')})
if __name__=='__main__':setup();run()
