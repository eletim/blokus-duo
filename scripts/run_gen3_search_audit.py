"""Finite frozen-model search comparison; no training, no live service changes."""
import sys,json,random,hashlib,time,copy,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from blokus.engine import State,Move,play,legal_moves,is_legal
from blokus.ppo_policy import PPOAgent
from blokus.ppo_search import SearchAgent
from blokus.ppo_search_batch import BatchSearch
from blokus.ppo_runtime import state_hash
R=Path('artifacts/gen3-search-audit');SEEDS=[1751401,1751402,1751403];INDEX=Path('/tmp/ppo-lr64-training.npy')
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
 # All locally available prior prefix banks, plus explicit exclusion union.
 for base in ('/home/eletim/blokus-duo-ai/artifacts','/home/eletim/blokus-linear-rl/artifacts','/home/eletim/blokus-linear-confirm/artifacts'):
  if not Path(base).exists():continue
  for f in Path(base).rglob('*prefix*.json'):
   try:d=json.loads(f.read_text())
   except (ValueError,OSError):continue
   if not isinstance(d,dict):continue
   ps=d.get('prefixes',[])
   if not isinstance(ps,list):continue
   for p in ps:
    if isinstance(p,dict) and 'moves' in p:
     try:forbidden.add(state_hash(state(p['moves'])))
     except (KeyError,TypeError,ValueError):pass
   files.append(dict(path=str(f),sha256=sha(f)))
 old=Path('/home/eletim/blokus-duo-ai/artifacts/ppo-confirmation/excluded-starts.json')
 if old.exists():forbidden.update(json.loads(old.read_text())['states'])
 write(R/'separation-source.json',dict(training_index_sha256=audit['union_sha256'],training_states=len(idx),prior_starts=len(forbidden),prior_banks=files))
 for phase,n,seed in [('validation',8,82721000),('confirmation',32,82731000)]:
  path=R/f'{phase}-bank.json'
  if path.exists():continue
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
def make(name,p):
 if name=='policy':return p
 if name=='old':return SearchAgent(p,500)
 return BatchSearch(p,500,exact=name=='C')
def summarize(games):
 d=np.array([g['difference'] for g in games]);return dict(games=len(d),wins=int((d>0).sum()),draws=int((d==0).sum()),losses=int((d<0).sum()),point=float((np.sign(d)*.5+.5).mean()),difference=float(d.mean()))
def compare(phase,left,right,seed):
 path=R/f'{phase}-{left}-{right}-{seed}.json'
 if path.exists():return json.loads(path.read_text())
 bank=json.loads((R/f'{phase}-bank.json').read_text());start=time.perf_counter();p=PPOAgent(seed,'cpu',generation='gen3',backend='python');load=time.perf_counter()-start;r=random.Random(1)
 t=time.perf_counter();p.act(State(),r);warm=time.perf_counter()-t;games=[];start=time.perf_counter()
 for prefix in bank['prefixes']:
  for seat in (0,1):
   s=state(prefix['moves']);bots=[make(left,p),make(right,p)];history=[];timings=[]
   while not s.done:
    who=0 if s.turn==seat else 1;a=bots[who];t=time.perf_counter();m=a.act(s,r);elapsed=(time.perf_counter()-t)*1000
    stats=dict(getattr(a,'stats',{}));stats.pop('root_scores',None);timings.append(dict(side=who,elapsed_ms=elapsed,**{k:v for k,v in stats.items() if k!='elapsed_ms'}));play(s,m,validate=False);history.append(None if m is None else list(m.__dict__.values()))
   games.append(dict(prefix=prefix['id'],seat=seat,moves=history,scores=[s.score(0),s.score(1)],difference=s.score(seat)-s.score(1-seat),timings=timings))
   print(phase,left,right,seed,prefix['id'],seat,flush=True)
 result=dict(left=left,right=right,seed=seed,model_sha256=sha(f'artifacts/ppo-gen3/seed-{seed}.pt'),bank_sha256=sha(R/f'{phase}-bank.json'),cold_load_seconds=load,warmup_seconds=warm,wall_seconds=time.perf_counter()-start,games=games,summary=summarize(games));write(path,result);return result
def aggregate(phase,left,right):
 rows=[json.loads((R/f'{phase}-{left}-{right}-{s}.json').read_text()) for s in SEEDS];groups={}
 for row in rows:
  for g in row['games']:groups.setdefault(g['prefix'],[]).append(g['difference'])
 assert all(len(x)==6 for x in groups.values());d=np.array([np.mean(v) for v in groups.values()]);p=np.array([np.mean(np.sign(v)*.5+.5) for v in groups.values()]);ix=np.random.default_rng(82749000).integers(len(p),size=(20000,len(p)));pm=p[ix].mean(1);dm=d[ix].mean(1)
 timing={}
 for side,name in enumerate([left,right]):
  ts=[t for row in rows for g in row['games'] for t in g['timings'] if t['side']==side];ms=np.array([t['elapsed_ms'] for t in ts]);timing[name]=dict(mean_ms=float(ms.mean()),p99_ms=float(np.quantile(ms,.99)),max_ms=float(ms.max()),over500=int((ms>500).sum()),moves=len(ts),fallback=float(np.mean([t.get('fallback',False) for t in ts])),depths={str(k):sum(t.get('depth',0)==k for t in ts) for k in (0,2,4)},exact_solved=sum(t.get('exact_solved',False) for t in ts),nn_calls=sum(t.get('nn_calls',0) for t in ts))
 return dict(point=float(p.mean()),point_ci95=np.quantile(pm,[.025,.975]).tolist(),point_ci_bonferroni2=np.quantile(pm,[.0125,.9875]).tolist(),difference=float(d.mean()),difference_ci95=np.quantile(dm,[.025,.975]).tolist(),by_seed={str(row['seed']):row['summary'] for row in rows},by_seat={str(seat):summarize([g for row in rows for g in row['games'] if g['seat']==seat]) for seat in (0,1)},time=timing,wall_seconds=sum(r['wall_seconds'] for r in rows))
def run():
 pairs=[('old','policy'),('B','old'),('B','policy'),('C','old'),('C','policy')]
 for a,b in pairs:
  for s in SEEDS:compare('validation',a,b,s)
 summary={a+'-'+b:aggregate('validation',a,b) for a,b in pairs};write(R/'validation-summary.json',summary)
 name=max(('B','C'),key=lambda c:(summary[c+'-old']['point']+summary[c+'-policy']['point'],c=='B'))
 a=summary[name+'-old'];b=summary[name+'-policy'];gate=a['point']>=.5 and b['point']>=.5 and a['difference']>=0 and all(x['time'][name]['fallback']<.5 and x['time'][name]['p99_ms']<750 for x in (a,b))
 write(R/'selection.json',dict(candidate=name,gate=gate,models_sha256=sha(R/'models.json'),selected_before_confirmation=True))
 if gate:
  for opponent in ('old','policy'):
   for s in SEEDS:compare('confirmation',name,opponent,s)
  write(R/'confirmation-summary.json',{name+'-'+b:aggregate('confirmation',name,b) for b in ('old','policy')})
def audit():
 idx=np.load(INDEX,mmap_mode='r');out=[]
 for phase in ('validation','confirmation'):
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
    assert s.done and g['scores']==[s.score(0),s.score(1)] and g['difference']==s.score(g['seat'])-s.score(1-g['seat'])
   assert row['summary']==summarize(row['games']);out.append(dict(file=str(path),games=len(row['games']),later_training_overlap=overlap))
 write(R/'audit.json',dict(pass_=True,rows=out))
if __name__=='__main__':
 torch.set_num_threads(1);p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run','audit']);a=p.parse_args();globals()[a.command]()
