from h_balanced import *
import h_distill_eval as old
old.R=R
from h_distill_eval import summary
PAIRS={'B-public':('B','public'),'B-A':('B','A'),'A-public':('A','public')}
def prepare():
 forbidden=base.exclusions();idx=np.load(INDEX,mmap_mode='r');training=set()
 for phase in ('pilot','train','dev'):
  for seed in SEEDS:
   for g in json.loads((R/f'{phase}-{seed}.json').read_text())['games']:training.update(g['visited'])
 forbidden|=training
 for phase,n,sd in [('validation',12,39400100),('confirmation',32,39500100)]:
  path=R/f'{phase}-bank.json'
  if path.exists():raise RuntimeError('bank exists, preserve fixed bank')
  rng=random.Random(sd);bank=[]
  while len(bank)<n:
   s=State();ms=[]
   for _ in range(rng.randint(6,10)):
    qs=legal_moves(s);m=rng.choice(qs).move if qs else None;play(s,m,validate=True);ms.append(raw(m))
   h=state_hash(s)
   if h in forbidden or contains(idx,h):continue
   forbidden.add(h);bank.append(dict(id=len(bank),moves=ms,state_hash=h))
  write(path,dict(seed=sd,prefixes=bank))
 write(R/'models.json',{str(seed):{a:dict(path=str(R/f'{a}-{seed}.pt') if a!='public' else f'artifacts/ppo-gen3/seed-{seed}.pt',sha256=sha(R/f'{a}-{seed}.pt') if a!='public' else sha(f'artifacts/ppo-gen3/seed-{seed}.pt')) for a in ('A','B','public')} for seed in SEEDS})
 write(R/'pre-evaluation-separation.json',dict(new_training_states=len(training),original_index_sha256='cdd1f0f1ac7a0e722a670fb541ce904187949c6982cd03fda1fa7acad30f7b4c',new_banks_disjoint=True))
def compare(phase,pair,seed,search=True):
 method=pair if search else 'policy-'+pair;path=R/f'{phase}-{method}-{seed}.json'
 if path.exists():return
 bank=json.loads((R/f'{phase}-bank.json').read_text())['prefixes'];models=[PPOAgent(seed,'cpu',checkpoint=None if a=='public' else R/f'{a}-{seed}.pt') for a in PAIRS[pair]];agents=[DeepSearch(p,500,width=4,max_depth=6) for p in models] if search else models
 games=[];start=time.perf_counter()
 for prefix in bank:
  for seat in (0,1):
   s=state(prefix['moves']);ms=[];times=[];rng=random.Random(39600100+prefix['id']*2+seat)
   while not s.done:
    side=int(s.turn!=seat);a=agents[side];t=time.perf_counter();m=a.act(s,rng);elapsed=(time.perf_counter()-t)*1000
    stat=copy.deepcopy(a.stats) if search else {};stat.pop('root_scores',None);times.append(dict(side=side,elapsed_ms=elapsed,**{k:v for k,v in stat.items() if k!='elapsed_ms'}));play(s,m,validate=True);ms.append(raw(m))
   games.append(dict(prefix=prefix['id'],seat=seat,moves=ms,scores=[s.score(0),s.score(1)],difference=s.score(seat)-s.score(1-seat),timings=times));print(phase,method,seed,prefix['id'],seat,flush=True)
 write(path,dict(seed=seed,games=games,summary=summary(games),wall_seconds=time.perf_counter()-start))
def aggregate(phase,method):
 out=old.aggregate(phase,method)
 gs=[g for seed in SEEDS for g in json.loads((R/f'{phase}-{method}-{seed}.json').read_text())['games']]
 out['by_seat']={str(s):summary([g for g in gs if g['seat']==s]) for s in (0,1)}
 for side in (0,1):out['time'][str(side)]['nonpass_fallback']=sum(t.get('fallback',False) and not t.get('forced_pass',False) for g in gs for t in g['timings'] if t['side']==side)
 return out

def run():
 if not (R/'models.json').exists():prepare()
 for pair in PAIRS:
  for seed in SEEDS:compare('validation',pair,seed)
 for pair in ('B-public','A-public'):
  for seed in SEEDS:compare('validation',pair,seed,False)
 methods=list(PAIRS)+['policy-B-public','policy-A-public'];results={m:aggregate('validation',m) for m in methods};write(R/'validation-summary.json',results)
 b=results['B-public'];ba=results['B-A'];pol=results['policy-B-public']
 gate=b['point']>.5 and b['difference']>0 and sum(x['point']>=.5 for x in b['by_seed'].values())>=2 and ba['point']>=.5 and pol['point_ci95'][1]>=.5 and b['time']['0']['p99_ms']<=750 and b['time']['0']['depth6_fraction']>=.2
 write(R/'selection.json',dict(gate=gate,models_sha256=sha(R/'models.json'),checkpoint='fixed epoch8 all3seeds',reason='prospective gate, public H improvement and no point regression against A; no rescue'))
 if gate:
  for pair in ('B-public','B-A'):
   for seed in SEEDS:compare('confirmation',pair,seed)
  write(R/'confirmation-summary.json',{m:aggregate('confirmation',m) for m in ('B-public','B-A')})
if __name__=='__main__':base.setup();run()
