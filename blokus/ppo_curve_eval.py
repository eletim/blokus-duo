"""Preregistered scaling evaluation; all training states excluded before evaluation."""
import argparse,gzip,json,random,time
from pathlib import Path
import numpy as np
import torch
from . import terminal_eval as e
from .terminal_ppo import observe,collate
from .engine import State,Move,play
from .tournament import digest,prefix_state,read_prefixes,summarize
from .visited_data import state_hash
from .research import sha
from .value_experiment import write
R=Path('artifacts/ppo-learning-curve');OLD=Path('artifacts/terminal-ppo/extended')

def previous():return json.loads((OLD/'selection.json').read_text())
def paths(seed):
 d=R/f'seed-{seed}';return sorted(d.glob('iteration-*.pt'),key=lambda p:int(p.stem.split('-')[-1]))
def audit_train():
 from .engine import is_legal,legal_moves
 prefixes=set();states=set();report=[]
 for seed in (1401,1402,1403):
  path=paths(seed)[-1];c=torch.load(path,map_location='cpu',weights_only=False);from .ppo_scale import history
  games=history(c,path);ids=set();counts={};seen_targets=0
  for g in games:
   assert g['id'] not in ids;ids.add(g['id']);s=State();prefixes.add(digest(g['prefix']));targets={r['ply']:r for r in g['targets']}
   for ply,row in enumerate(g['prefix']+g['moves']):
    h=state_hash(s);states.add(h);m=Move(*row) if row is not None else None
    if ply in targets:
     t=targets[ply];assert t['player']==s.turn and t['state_hash']==h
     assert t['return']==float(np.sign(g['scores'][s.turn]-g['scores'][1-s.turn]));seen_targets+=1
     if seen_targets%128==0:
      qs=legal_moves(s);assert (qs[t['action']].move if qs else None)==m
    assert is_legal(s,m) if m else not legal_moves(s)
    play(s,m,validate=False)
   states.add(state_hash(s));assert s.done and g['scores']==[s.score(0),s.score(1)];counts[g['opponent']]=counts.get(g['opponent'],0)+1
  assert counts['current']==2*counts['past']==2*counts['heuristic'];assert len(games)==c['iteration']*16
  for l in c['logs']:
   assert all(np.isfinite(v) for v in l['update'].values());assert l['update']['grad_norm']>0
  report.append(dict(seed=seed,games=len(games),targets=seen_targets,opponents=counts,checkpoint_sha256=sha(path),history_sha256=c.get('history_sha256'),history_chain=c.get('history_chain'),prefix_rejections=sum(g.get('prefix_rejections',0) for g in games)))
  del games,c
  print('training audit',report[-1],flush=True)
 # Compact sorted binary hashes for exact reproducible membership audit.
 for name,values in [('training-states',states),('training-prefixes',prefixes)]:
  folder=R/name;folder.mkdir(exist_ok=True);ordered=sorted(values);parts=[]
  for offset in range(0,len(ordered),500000):
   path=folder/f'{offset//500000:03d}.bin.gz';path.write_bytes(gzip.compress(b''.join(bytes.fromhex(x) for x in ordered[offset:offset+500000]),mtime=0));parts.append(dict(path=path.name,sha256=sha(path)))
  write(folder/'index.json',dict(count=len(values),parts=parts))
 write(R/'training-audit.json',dict(pass_=True,seeds=report,unique_states=len(states),unique_prefixes=len(prefixes),includes_opponent_turns=True,includes_prefix_and_terminal=True))

def hashes(name):
 index=json.loads((R/name/'index.json').read_text());out=set()
 for part in index['parts']:
  path=R/name/part['path'];assert sha(path)==part['sha256'];data=gzip.decompress(path.read_bytes());out.update(data[i:i+32].hex() for i in range(0,len(data),32))
 assert len(out)==index['count'];return out
def forbidden():
 ps=hashes('training-prefixes');ss=hashes('training-states');sources=[]
 for p in Path('artifacts').rglob('*prefix*.json'):
  try:
   data=json.loads(p.read_text())
   if not isinstance(data,dict) or not isinstance(data.get('prefixes'),list):continue
   for row in data['prefixes']:
    ps.add(digest(row['moves']));ss.add(state_hash(prefix_state(row)))
   sources.append(dict(path=str(p),sha256=sha(p),count=len(data['prefixes'])))
  except (KeyError,TypeError,ValueError):continue
 return ps,ss,sources

def fixture(name,seed,count):
 path=R/f'{name}-prefixes.json'
 if path.exists():return read_prefixes(path)
 ps,ss,sources=forbidden();f=e.disjoint_fixture(seed,count,ps,ss);starts={state_hash(prefix_state(p)) for p in f['prefixes']};assert len(starts)==count and not starts&ss and not {p['id'] for p in f['prefixes']}&ps
 pool=e.make_prefixes(seed,count*4);chosen={p['id'] for p in f['prefixes']};rejected=[]
 for row in pool['prefixes']:
  if row['id'] in chosen:continue
  h=state_hash(prefix_state(row))
  if row['id'] in ps or h in ss:rejected.append(dict(id=row['id'],prefix_overlap=row['id'] in ps,state_overlap=h in ss))
 write(path,f);write(R/f'{name}-preflight.json',dict(pass_=True,seed=seed,candidate_pool_sha256=pool['sha256'],candidate_pool_count=len(pool['prefixes']),overlap_rejections_in_pool=rejected,prefix_sha256=f['sha256'],training_states_sha256=sha(R/'training-states/index.json'),forbidden_states=len(ss),forbidden_prefixes=len(ps),sources=sources,adopted_start_overlap=0,selection_sha256=sha(R/'selection.json') if (R/'selection.json').exists() else None));return f

def cached(path,left,right,f):
 if path.exists():
  r=json.loads(path.read_text());assert r['prefix_sha256']==f['sha256'] and r['left_sha256']==sha(left);return r
 r=e.match(str(left),str(right),f);write(path,r);print(path,r['summary'],flush=True);return r

def validation(prefix_seed=1530000):
 f=fixture('validation',prefix_seed,32);old=previous();selected={}
 for seed in (1401,1402,1403):
  prior=old[str(seed)];assert sha(prior['path'])==prior['sha256'];options=[Path(prior['path'])]+paths(seed);candidates=[]
  for ck in options:
   label='previous' if str(ck)==prior['path'] else ck.stem;rows=[]
   for name,spec in [('previous',prior['path']),('heuristic','heuristic'),('oldvalue','oldvalue')]:
    rows.append(cached(R/f'seed-{seed}/validation-{label}-{name}.json',ck,spec,f))
   weights=[.5,.25,.25];step=int(ck.stem.split('-')[-1]);criterion=[sum(w*r['summary']['point_rate'] for w,r in zip(weights,rows)),sum(w*r['summary']['mean_difference'] for w,r in zip(weights,rows)),-step]
   candidates.append(dict(path=str(ck),sha256=sha(ck),iteration=step,criterion=criterion,label=label))
  selected[str(seed)]=max(candidates,key=lambda c:c['criterion']);write(R/f'seed-{seed}/candidates.json',candidates)
 write(R/'selection.json',selected)

def holdout(prefix_seed=1540000,opponents=('previous','heuristic','oldvalue')):
 selected=json.loads((R/'selection.json').read_text());f=fixture('holdout',prefix_seed,96);old=previous()
 assert json.loads((R/'holdout-preflight.json').read_text())['selection_sha256']==sha(R/'selection.json')
 for seed,c in selected.items():
  assert sha(c['path'])==c['sha256']
  for name,spec in [('previous',old[seed]['path']),('heuristic','heuristic'),('oldvalue','oldvalue')]:
   if name not in opponents:continue
   cached(R/f'seed-{seed}/holdout-{name}.json',c['path'],spec,f)

class Stochastic(e.TerminalAgent):
 def act(self,s,rng):
  obs,qs=observe(s)
  with torch.no_grad():logits,_=self.model(*collate([obs],self.device));prob=logits[0].softmax(-1).cpu().numpy()
  ix=rng.choices(range(len(prob)),weights=prob,k=1)[0];return qs[ix].move if qs else None

def dev():
 f=fixture('dev',1520000,8);original=e.agent
 def factory(spec):return Stochastic(spec[6:]) if str(spec).startswith('stoch:') else original(spec)
 e.agent=factory
 for seed in (1401,1402,1403):
  options=[Path(previous()[str(seed)]['path'])]+paths(seed)
  resume512=OLD/f'seed-{seed}/iteration-512.pt'
  if resume512 not in options:options.append(resume512)
  for ck in options:
   label='previous' if str(ck)==previous()[str(seed)]['path'] else 'resume512' if ck==resume512 else ck.stem
   for stochastic in (False,True):
    dest=R/f'seed-{seed}/dev-{label}-{stochastic}.json'
    if dest.exists():
     prior=json.loads(dest.read_text());assert prior['checkpoint_sha256']==sha(ck) and prior['prefix_sha256']==f['sha256'];continue
    # match path hash is separately recorded because stochastic specification is not a file path.
    r=e.match(('stoch:' if stochastic else '')+str(ck),'heuristic',f);r['checkpoint_sha256']=sha(ck);write(dest,r)
 e.agent=original

def audit_eval(opponents=('previous','heuristic','oldvalue')):
 from .engine import is_legal,legal_moves
 training=hashes('training-states');reports={};all_states=set()
 for phase in ('dev','validation','holdout'):
  if not (R/f'{phase}-prefixes.json').exists():continue
  f=read_prefixes(R/f'{phase}-prefixes.json');assert f['count']=={'dev':8,'validation':32,'holdout':96}[phase];preflight=json.loads((R/f'{phase}-preflight.json').read_text());assert preflight['prefix_sha256']==f['sha256'] and preflight['training_states_sha256']==sha(R/'training-states/index.json');lookup={p['id']:p for p in f['prefixes']};starts={state_hash(prefix_state(p)) for p in f['prefixes']};assert not starts&training
  count=0;visited=set()
  for path in R.glob(f'seed-*/{phase}-*.json'):
   r=json.loads(path.read_text())
   if 'games' not in r:continue
   if phase=='holdout':
    selected=json.loads((R/'selection.json').read_text())[path.parent.name.split('-')[-1]];assert r['left']==selected['path'] and r['left_sha256']==selected['sha256']
   assert r['prefix_sha256']==f['sha256'] and r['summary']==summarize(r['games']);pairs={};calls=0
   if r['left_sha256'] is not None:assert sha(r['left'])==r['left_sha256']
   if 'checkpoint_sha256' in r:assert sha(r['left'].removeprefix('stoch:'))==r['checkpoint_sha256']
   assert all(np.isfinite(v) and v>=0 for t in r['timing'] for v in t['all_ms'])
   if Path(r['right']).is_file():assert sha(r['right'])==r['right_sha256']
   for g in r['games']:
    s=prefix_state(lookup[g['prefix_id']]);pairs.setdefault(g['prefix_id'],[]).append(g['seat'])
    for row in g['moves']:
     visited.add(state_hash(s));m=Move(*row) if row else None;assert is_legal(s,m) if m else not legal_moves(s);play(s,m,validate=False);calls+=1
    assert s.done and g['scores']==[s.score(0),s.score(1)] and g['difference']==s.score(g['seat'])-s.score(1-g['seat']);count+=1
   assert all(sorted(x)==[0,1] for x in pairs.values());assert calls==sum(t['calls'] for t in r['timing'])
  reports[phase]=dict(games=count,unique_decision_states=len(visited),training_actual_visited_overlap=len(visited&training),adopted_start_overlap=0);all_states|=visited
 selection=json.loads((R/'selection.json').read_text());assert sha(R/'selection.json')==json.loads((R/'holdout-preflight.json').read_text())['selection_sha256']
 for seed,c in selection.items():
  candidates=json.loads((R/f'seed-{seed}/candidates.json').read_text());assert c==max(candidates,key=lambda x:x['criterion'])
  for candidate in candidates:
   rows=[json.loads((R/f"seed-{seed}/validation-{candidate['label']}-{op}.json").read_text()) for op in ('previous','heuristic','oldvalue')]
   expected=[sum(w*r['summary'][k] for w,r in zip((.5,.25,.25),rows)) for k in ('point_rate','mean_difference')]+[-candidate['iteration']]
   assert candidate['criterion']==expected
 aggregate={}
 for opponent in opponents:
  arrays=[]
  for seed in (1401,1402,1403):
   r=json.loads((R/f'seed-{seed}/holdout-{opponent}.json').read_text());groups={}
   for g in r['games']:groups.setdefault(g['prefix_id'],[]).append(g['difference'])
   arrays.append(np.array([[np.mean([float(d>0)+.5*float(d==0) for d in ds]),np.mean(ds)] for ds in groups.values()]))
  x=np.mean(arrays,axis=0);ix=np.random.default_rng(1550000).integers(len(x),size=(4000,len(x)));ci=np.quantile(x[ix].mean(1),[.025,.975],axis=0);aggregate[opponent]=dict(point=float(x[:,0].mean()),difference=float(x[:,1].mean()),point_ci95=ci[:,0].tolist(),difference_ci95=ci[:,1].tolist(),per_seed_mean=np.mean(arrays,axis=1).tolist())
 write(R/'evaluation-audit.json',dict(pass_=True,phases=reports,aggregate=aggregate,note='Initial empty/prefix traversal states are excluded from evaluation decisions, but included in training forbidden states. Nonzero actual overlap is reported, not silently filtered. Paired seats and seeds clustered within prefix.'))
 print(aggregate,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['audit_train','validation','holdout','dev','audit_eval','match'])
 for option in ('left','right','fixture','output'):p.add_argument('--'+option)
 a=p.parse_args()
 if a.mode=='match':
  if not all((a.left,a.right,a.fixture,a.output)):p.error('match requires --left --right --fixture --output')
  if Path(a.output).exists():p.error('output exists; choose a new path to preserve recorded results')
  write(a.output,e.match(a.left,a.right,read_prefixes(a.fixture)))
 else:globals()[a.mode]()
