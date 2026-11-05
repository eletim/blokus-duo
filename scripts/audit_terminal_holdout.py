import sys,json,argparse,gzip,hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from blokus.terminal_ppo import ROOT
from blokus.tournament import read_prefixes,prefix_state,summarize,digest
from blokus.visited_data import state_hash
from blokus.research import sha
from blokus.value_experiment import write
from audit_root_distillation import replay
p=argparse.ArgumentParser();p.add_argument('--root',default=str(ROOT/'extended'));root=Path(p.parse_args().root)
assert json.loads((root/'decision.json').read_text())['advance'];fixture=read_prefixes(root/'holdout-prefixes.json');validation=read_prefixes(root/'validation-prefixes.json');assert not {p['id'] for p in fixture['prefixes']}&{p['id'] for p in validation['prefixes']};lookup={p['id']:p for p in fixture['prefixes']};selection=json.loads((root/'selection.json').read_text());games_count=0;reports={}
train_prefixes=set();train_states=set();eval_states=set()
for seed in selection:
 d=root/f'seed-{seed}';gs=json.loads((d/'games.json').read_text()) if (d/'games.json').exists() else json.load(gzip.open(d/'games.json.gz','rt'))
 for g in gs:
  train_prefixes.add(digest(g['prefix']))
  train_states.update(r['state_hash'] for r in g['targets'])
assert not train_prefixes & {digest(p['moves']) for p in fixture['prefixes']}
starts={state_hash(prefix_state(p)) for p in fixture['prefixes']};assert len(starts)==96 and not starts&train_states
plan=root/'holdout-disjoint-plan.json'
if plan.exists():assert hashlib.sha256((root/'selection.json').read_bytes()).hexdigest()==json.loads(plan.read_text())['selection_file_sha256']
for op in ['heuristic','untrained','oldvalue']:

 arrays=[]
 for seed,c in selection.items():
  r=json.loads((root/f'seed-{seed}/holdout-{op}.json').read_text());assert r['prefix_sha256']==fixture['sha256'];assert r['left_sha256']==sha(c['path'])==c['sha256'];assert r['summary']==summarize(r['games']);groups={}
  for g in r['games']:
   s=prefix_state(lookup[g['prefix_id']]);groups.setdefault(g['prefix_id'],[]).append(g['difference']);games_count+=1
   for m in g['moves']:eval_states.add(state_hash(s));replay(s,m)
   assert s.done and g['scores']==[s.score(0),s.score(1)];assert g['difference']==s.score(g['seat'])-s.score(1-g['seat'])
  arrays.append(np.array([[np.mean([float(x>0)+.5*float(x==0) for x in ds]),np.mean(ds)] for ds in groups.values()]))
 x=np.mean(arrays,0);ix=np.random.default_rng(1460000).integers(len(x),size=(4000,len(x)));ci=np.quantile(x[ix].mean(1),[.025,.975],axis=0);reports[op]=dict(point_rate=float(x[:,0].mean()),difference=float(x[:,1].mean()),point_ci95=ci[:,0].tolist(),difference_ci95=ci[:,1].tolist())
write(root/'holdout-audit.json',dict(pass_=True,games=games_count,prefixes=fixture['count'],checkpoint_reselection=False,training_prefix_overlap=0,training_visited_state_overlap=len(train_states&eval_states),aggregate=reports,note='Game pair and update seeds averaged inside each prefix, conditional on selected models.'))
