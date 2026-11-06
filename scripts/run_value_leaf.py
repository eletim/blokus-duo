"""Finite frozen-model search comparison; no training, no live service changes."""
import sys,json,random,hashlib,time,copy,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from blokus.engine import State,Move,play,legal_moves,is_legal
from blokus.ppo_policy import PPOAgent
from blokus.ppo_search import SearchAgent
from blokus.ppo_search_staged import ValueLeafSearch
from blokus.ppo_runtime import state_hash
R=Path('artifacts/gen3-value-leaf');SEEDS=[1751401,1751402,1751403];INDEX=Path('/tmp/ppo-lr64-training.npy')
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def contains(idx,h):
 v=np.bytes_(bytes.fromhex(h));i=np.searchsorted(idx,v);return i<len(idx) and idx[i]==v
def state(moves):
 s=State()
 for m in moves:play(s,Move(*m) if m else None,validate=False)
 return s
def prepare():
 R.mkdir(exist_ok=True);idx=np.load(INDEX,mmap_mode='r');audit=json.load(open('/home/eletim/blokus-duo-ai/artifacts/ppo-lr64/training-audit.json'));assert hashlib.sha256(idx.tobytes()).hexdigest()==audit['union_sha256']
 forbidden=set();files=[]
 for base in Path('/home/eletim').glob('blokus*'):
  root=base/'artifacts'
  if not root.is_dir() or base.resolve()==Path.cwd().resolve():continue
  found=set(root.rglob('*prefix*.json'))|set(root.rglob('*bank.json'))|set(root.rglob('excluded-starts.json'))
  for f in sorted(found):
   try:d=json.loads(f.read_text())
   except (ValueError,OSError):continue
   if not isinstance(d,dict):continue
   if isinstance(d.get('states'),list):forbidden.update(h for h in d['states'] if isinstance(h,str))
   ps=d.get('prefixes',[])
   if not isinstance(ps,list):continue
   for p in ps:
    if isinstance(p,dict) and 'moves' in p:
     try:forbidden.add(state_hash(state(p['moves'])))
     except (KeyError,TypeError,ValueError):pass
   files.append(dict(path=str(f),sha256=sha(f)))
 write(R/'separation-source.json',dict(training_index_sha256=audit['union_sha256'],training_states=len(idx),prior_starts=len(forbidden),prior_banks=files))
 for phase,n,seed in [('validation500',16,82821000),('validation2000',8,82822000),('confirmation500',32,82831000)]:
  path=R/f'{phase}-bank.json'
  if path.exists():
   forbidden.update(p['state_hash'] for p in json.loads(path.read_text())['prefixes']);continue
  rng=random.Random(seed);bank=[]
  while len(bank)<n:
   s=State();moves=[]
   for _ in range(rng.randint(4,10)):
    qs=legal_moves(s);m=rng.choice(qs).move if qs else None;play(s,m,validate=False);moves.append(None if m is None else list(m.__dict__.values()))
   h=state_hash(s)
   if h in forbidden or contains(idx,h):continue
   forbidden.add(h);bank.append(dict(id=len(bank),moves=moves,state_hash=h))
  write(path,dict(seed=seed,prefixes=bank,training_start_overlap=0))
 write(R/'models.json',{str(s):dict(path=f'artifacts/ppo-gen3/seed-{s}.pt',sha256=sha(f'artifacts/ppo-gen3/seed-{s}.pt')) for s in SEEDS})
def make(name,p,budget):
 if name=='policy':return p
 if name=='A':return SearchAgent(p,budget)
 return ValueLeafSearch(p,budget,root_stages=(8,12) if name=='D' else (),reuse=name=='D')
def summarize(games):
 d=np.array([g['difference'] for g in games]);return dict(games=len(d),wins=int((d>0).sum()),draws=int((d==0).sum()),losses=int((d<0).sum()),point=float((np.sign(d)*.5+.5).mean()),difference=float(d.mean()))
def compare(phase,left,right,seed):
 path=R/f'{phase}-{left}-{right}-{seed}.json'
 if path.exists():return json.loads(path.read_text())
 budget=2000 if phase=='validation2000' else 500;bank=json.loads((R/f'{phase}-bank.json').read_text());start=time.perf_counter();p=PPOAgent(seed,'cpu',generation='gen3',backend='python');load=time.perf_counter()-start;r=random.Random(1)
 t=time.perf_counter();p.act(State(),r);warm=time.perf_counter()-t;games=[];start=time.perf_counter()
 for prefix in bank['prefixes']:
  for seat in (0,1):
   s=state(prefix['moves']);r=random.Random(8289000+prefix['id']*2+seat);bots=[make(left,p,budget),make(right,p,budget)];history=[];timings=[]
   while not s.done:
    who=0 if s.turn==seat else 1;a=bots[who];t=time.perf_counter();m=a.act(s,r);elapsed=(time.perf_counter()-t)*1000
    stats=dict(getattr(a,'stats',{}));stats['forced_pass']=m is None;stats.pop('root_scores',None);timings.append(dict(side=who,elapsed_ms=elapsed,**{k:v for k,v in stats.items() if k!='elapsed_ms'}));play(s,m,validate=False);history.append(None if m is None else list(m.__dict__.values()))
   games.append(dict(prefix=prefix['id'],seat=seat,moves=history,scores=[s.score(0),s.score(1)],difference=s.score(seat)-s.score(1-seat),timings=timings))
   print(phase,left,right,seed,prefix['id'],seat,flush=True)
 result=dict(left=left,right=right,seed=seed,budget_ms=budget,model_sha256=sha(f'artifacts/ppo-gen3/seed-{seed}.pt'),bank_sha256=sha(R/f'{phase}-bank.json'),cold_load_seconds=load,warmup_seconds=warm,wall_seconds=time.perf_counter()-start,games=games,summary=summarize(games));write(path,result);return result
def aggregate(phase,left,right):
 rows=[json.loads((R/f'{phase}-{left}-{right}-{s}.json').read_text()) for s in SEEDS];groups={}
 for row in rows:
  for g in row['games']:groups.setdefault(g['prefix'],[]).append(g['difference'])
 assert all(len(x)==6 for x in groups.values());d=np.array([np.mean(v) for v in groups.values()]);p=np.array([np.mean(np.sign(v)*.5+.5) for v in groups.values()]);ix=np.random.default_rng(82749000).integers(len(p),size=(20000,len(p)));pm=p[ix].mean(1);dm=d[ix].mean(1)
 timing={}
 for side,name in enumerate([left,right]):
  ts=[t for row in rows for g in row['games'] for t in g['timings'] if t['side']==side];ms=np.array([t['elapsed_ms'] for t in ts]);timing[name]=dict(mean_ms=float(ms.mean()),p99_ms=float(np.quantile(ms,.99)),max_ms=float(ms.max()),over_budget=int((ms>(2000 if phase=='validation2000' else 500)).sum()),moves=len(ts),fallback=float(np.mean([t.get('fallback',False) for t in ts])),depths={str(k):sum(t.get('depth',0)==k for t in ts) for k in (0,2,4)},widths={str(k):sum(t.get('root_width',0)==k for t in ts) for k in (0,4,8,12)},nonpass_wide_fraction=sum(not t.get('forced_pass',False) and t.get('root_width',0)>=8 for t in ts)/max(1,sum(not t.get('forced_pass',False) for t in ts)),nn_policy_calls=sum(t.get('nn_policy_calls',0) for t in ts),nn_value_calls=sum(t.get('nn_value_calls',0) for t in ts),cache_hits=sum(t.get('cache_hits',0) for t in ts))
 return dict(point=float(p.mean()),point_ci95=np.quantile(pm,[.025,.975]).tolist(),point_ci_bonferroni2=np.quantile(pm,[.0125,.9875]).tolist(),difference=float(d.mean()),difference_ci95=np.quantile(dm,[.025,.975]).tolist(),by_seed={str(row['seed']):row['summary'] for row in rows},by_seat={str(seat):summarize([g for row in rows for g in row['games'] if g['seat']==seat]) for seat in (0,1)},time=timing,wall_seconds=sum(r['wall_seconds'] for r in rows))
def run():
 pairs=[('A','policy'),('E','A'),('D','A'),('D','E'),('D','policy')]
 for a,b in pairs:
  for s in SEEDS:compare('validation500',a,b,s)
 for s in SEEDS:compare('validation2000','D','A',s)
 summary={a+'-'+b:aggregate('validation500',a,b) for a,b in pairs};write(R/'validation-summary.json',summary);write(R/'aux2000-summary.json',aggregate('validation2000','D','A'))
 a=summary['D-A'];b=summary['D-policy'];e=summary['D-E']
 gate=a['point']>=.5 and b['point']>=.5 and a['difference']>=0 and b['difference']>=0 and e['point_ci95'][1]>=.5 and all(x['time']['D']['p99_ms']<=750 and x['time']['D']['nonpass_wide_fraction']>=.2 for x in (a,b))
 write(R/'selection.json',dict(candidate='D',budget_ms=500,gate=gate,models_sha256=sha(R/'models.json'),selected_before_confirmation=True))
 if gate:
  for opponent in ('A','policy'):
   for s in SEEDS:compare('confirmation500','D',opponent,s)
  write(R/'confirmation-summary.json',{'D-'+b:aggregate('confirmation500','D',b) for b in ('A','policy')})
def audit():
 idx=np.load(INDEX,mmap_mode='r');out=[]
 for phase in ('validation500','validation2000','confirmation500'):
  bank=json.loads((R/f'{phase}-bank.json').read_text())['prefixes']
  for path in R.glob(phase+'-*-*-*.json'):
   row=json.loads(path.read_text());assert row['model_sha256']==sha(f"artifacts/ppo-gen3/seed-{row['seed']}.pt");overlap=0
   for g in row['games']:
    s=State()
    for m in bank[g['prefix']]['moves']:
     move=Move(*m) if m else None;assert is_legal(s,move);play(s,move,validate=False)
    assert not contains(idx,state_hash(s))
    for m in g['moves']:
     overlap+=int(contains(idx,state_hash(s)));move=Move(*m) if m else None;assert is_legal(s,move);play(s,move,validate=False)
    overlap+=int(contains(idx,state_hash(s)))
    assert s.done and g['scores']==[s.score(0),s.score(1)] and g['difference']==s.score(g['seat'])-s.score(1-g['seat'])
   assert row['summary']==summarize(row['games']);out.append(dict(file=str(path),games=len(row['games']),later_training_overlap=overlap))
 write(R/'audit.json',dict(pass_=True,rows=out))
if __name__=='__main__':
 torch.set_num_threads(1);p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run','audit']);a=p.parse_args();globals()[a.command]()
