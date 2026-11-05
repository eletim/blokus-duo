"""Extra source, dev-label, outcome, and timed-evaluation separation audit."""
import json
from pathlib import Path
import numpy as np
from blokus.ppo_short_experiment import R,model,audit
from blokus.ppo_curve_eval import read_hash_index
from blokus.ppo_short_search import ShortSearch
from blokus.engine import Move,play,legal_moves,is_legal
from blokus.tournament import prefix_state,read_prefixes,digest
from blokus.visited_data import state_hash
from blokus.research import sha
from blokus.value_experiment import write
from blokus.ai import seed_all
seed_all(1690000)
lock=json.loads((R/'validation-lock.json').read_text());assert sha(R/'PLAN.md')==lock['plan_sha256']
for path,h in lock['source_sha256'].items():assert sha(path)==h
assert sha(R/'dev-cost-profile.json')==lock['dev_profile_sha256']
train=read_hash_index(R/'training-states/index.json');d=json.loads((R/'dev.json').read_text());f=read_prefixes(R/'dev-prefixes.json');prefixes={p['id']:p for p in f['prefixes']};seen=set();root_count=0;models={seed:ShortSearch(model(seed),device='cpu') for seed in [1401,1402,1403]}
for g in d['games']:
 s=prefix_state(prefixes[g['prefix_id']]);roots={r['ply']:r for r in d['roots'] if r['game']==g['game']}
 for ply,row in enumerate(g['moves'],start=len(prefixes[g['prefix_id']]['moves'])):
  h=state_hash(s);seen.add(h)
  if ply in roots:
   r=roots[ply];assert r['state_hash']==h and r['player']==s.turn
   assert abs(models[g['seed']].values([s],s.turn)[0]-r['value'])<2e-6
   assert r['return']==np.sign(g['scores'][s.turn]-g['scores'][1-s.turn]);root_count+=1
  m=Move(*row) if row else None;assert is_legal(s,m) if m else not legal_moves(s);play(s,m,validate=False)
 seen.add(state_hash(s));assert s.done and g['scores']==[s.score(0),s.score(1)]
assert not seen&train and root_count==len(d['roots'])
preflights=[]
for name in ['dev','validation','holdout']:
 path=R/f'{name}-preflight.json'
 if not path.exists():continue
 pre=json.loads(path.read_text());f=read_prefixes(R/f'{name}-prefixes.json');assert f['sha256']==pre['prefix_sha256'];assert sha(R/'training-states/index.json')==pre['training_states_sha256']
 for source in pre['sources']:assert sha(source['path'])==source['sha256']
 starts={state_hash(prefix_state(p)) for p in f['prefixes']};assert len(starts)==f['count'] and not starts&train
 if name=='holdout':assert pre['selection_sha256']==sha(R/'selection.json')
 prior=set()
 for source in pre['sources']:
  data=json.loads(Path(source['path']).read_text());prior.update(state_hash(prefix_state(x)) for x in data['prefixes'])
 actual=set()
 if name=='dev':actual=seen
 else:
  byid={x['id']:x for x in f['prefixes']}
  for result_path in R.glob(name+'-*.json'):
   result=json.loads(result_path.read_text())
   if 'games' not in result:continue
   assert result['opponent_sha256']==(json.loads((R/'lock.json').read_text())['teacher_sha256'] if result['opponent'] in ('oldvalue','teacher100') else sha(model(result['seed'])))
   for index,timing in enumerate(result['timing']):
    ts=np.array(timing['all_ms']);assert len(ts)==timing['calls'] and np.isfinite(ts).all() and (ts>=0).all()
    assert abs(ts.mean()-timing['mean_ms'])<1e-10 and abs(np.quantile(ts,.95)-timing['p95_ms'])<1e-10 and ts.max()==timing['max_ms']
    budget=result['budget_ms'] if index==0 else 6 if result['opponent']=='oldvalue' else 100 if result['opponent']=='teacher100' else None
    if budget is not None:assert float(np.mean(ts>budget))==timing['overrun_rate']
   for game in result['games']:
    state=prefix_state(byid[game['prefix_id']]);actual.add(state_hash(state))
    for row in game['moves']:
     play(state,Move(*row) if row else None,validate=False);actual.add(state_hash(state))
 assert not actual&prior
 preflights.append(dict(phase=name,starts=len(starts),training_overlap=0,prior_sources=len(pre['sources']),actual_states=len(actual),actual_prior_start_overlap=0))
write(R/'detail-audit.json',dict(pass_=True,dev_games=len(d['games']),dev_roots=root_count,dev_states=len(seen),dev_training_overlap=0,CPU_CUDA_value_agreement_tolerance=2e-6,source_hashes=True,preflights=preflights))
del train
# Main evaluator audit replays every timed game and recomputes paired CI and timings.
audit();print('detail + evaluation audit PASS')
