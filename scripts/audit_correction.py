"""Implementation-side independent arithmetic/replay checks, not external review."""
import sys,json,gzip,time,hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from blokus.correction_data import ROOT
from blokus.engine import State,legal_moves,is_legal
from blokus.research import clone,sha
from blokus.ai import seed_all,features,heuristic
from blokus.value_search import ValueEvaluator,encode
from blokus.deep_search import TreeSearch
from blokus.aligned_data import BASE
from blokus.node_diagnostic import child,row
from blokus.visited_data import state_hash
from blokus.root_policy import inputs
from blokus.correction import corrected_values
from blokus.value_experiment import write
from blokus.root_experiment import pooled
import audit_root_distillation as shared

def main():
 seed_all(1000);start=time.perf_counter();data=np.load(ROOT/'data.npz');sources=json.load(gzip.open(ROOT/'source-games.json.gz','rt'));records=json.load(gzip.open(ROOT/'teacher-labels.json.gz','rt'));states={};source_hashes=set()
 for g in sources:
  s=State()
  for ply,m in enumerate(g['moves']):states[g['game'],ply]=state_hash(s);source_hashes.add(state_hash(s));shared.replay(s,m)
  assert s.done and g['scores']==[s.score(0),s.score(1)]
 tree=TreeSearch(ValueEvaluator(BASE,'cuda',.5,24),width=3,tt_limit=0);replayed=0;splits=[set(),set(),set()];regrets=[];selected_hashes=[]
 for j,r in enumerate(records):
  s=clone(State(**r['state']));assert states[r['game'],r['ply']]==r['state_hash']==state_hash(s);splits[r['split']].add(r['state_hash'])
  qs=legal_moves(s);order=np.argsort(-heuristic(features(s,qs)[0]),kind='stable')[:128];qs=[qs[int(i)] for i in order];n=len(qs)
  assert [row(q.move) for q in qs]==r['moves'] and all(is_legal(s,q.move) for q in qs)
  x,_=inputs(s,qs);b,st=encode(s);assert np.array_equal(data['x'][j,:n],x) and np.array_equal(data['board'][j],b) and np.array_equal(data['state'][j],st)
  expected=np.array([v is not None for v in r['final']]);assert np.array_equal(data['mask'][j,:n],expected) and not data['mask'][j,n:].any()
  assert np.isnan(data['target'][j,~data['mask'][j]]).all();assert data['game'][j]==r['game'] and data['split'][j]==r['split']
  known=np.flatnonzero(expected);truth=np.array([v if v is not None else np.nan for v in r['final']]);delta=truth-np.array(r['initial'])
  assert np.allclose(data['target'][j,:n][expected],delta[expected],atol=1e-5)
  v=corrected_values(r['initial'],delta,expected);best=v.max();chosen=np.flatnonzero(v>=best-4e-6);assert all(truth[i]>=np.nanmax(truth)-4e-6 for i in chosen)
  selected_hashes.append(hashlib.sha256(json.dumps([r['moves'][i] for i in chosen]).encode()).hexdigest())
  if j%20==0:
   for i in known:
    c=child(s,qs[i].move);assert abs(-tree.search(c,1)-r['initial'][i])<1e-5;assert abs(-tree.search(c,3)-r['final'][i])<1e-5
   replayed+=1
 overlaps={f'{i}/{k}':len(splits[i]&splits[k]) for i in range(3) for k in range(i)};assert not any(overlaps.values())
 shared.ROOT=ROOT;eval_report,eval_hashes=shared.audit_matches();selection=json.loads((ROOT/'selection.json').read_text())
 best=sorted(selection['criteria'],key=lambda x:(-x['score'],x['budget'],x['alpha'],x['step']))[0];assert selection['choice']==best and selection['advance']==all(selection['gate'].values())
 validation=json.loads((ROOT/'validation-results.json').read_text())
 for c in selection['criteria']:
  rows=[r for r in validation if r['right']['spec']['method']=='old' and all(r['left']['spec'][k]==c[key] for k,key in [('step','step'),('alpha','alpha'),('budget_ms','budget')])]
  assert len(rows)==3 and abs(pooled(rows)['mean_difference']-c['score'])<1e-12
 chosen=[r for r in validation if all(r['left']['spec'][k]==best[key] for k,key in [('step','step'),('alpha','alpha'),('budget_ms','budget')])]
 old=[r for r in chosen if r['right']['spec']['method']=='old'];zero=[r for r in chosen if r['right']['spec']['method']=='zero']
 assert pooled(old)==selection['old'] and pooled(zero)==selection['zero']
 ms=[d['policy_ms'] for r in old for g in r['games'] for d in g['decisions'] if d['who']==0]
 gates=dict(positive_old=pooled(old)['mean_difference']>0,each_seed_above_minus2=all(r['summary']['mean_difference']>-2 for r in old),positive_zero=pooled(zero)['mean_difference']>0,inference_below_1ms=np.mean(ms)<1,dev_regret_improved=best['regret']<best['zero_regret'])
 assert selection['gate']==gates and abs(np.mean(ms)-selection['mean_inference_ms'])<1e-12

 for ck in selection['checkpoints'].values():assert sha(ck['path'])==ck['sha256']
 if (ROOT/'holdout-pooled.json').exists():
  results=json.loads((ROOT/'holdout-results.json').read_text());report=json.loads((ROOT/'holdout-pooled.json').read_text())
  for r in results:
   left=r['left']['spec'];assert left['step']==best['step'] and left['alpha']==best['alpha'] and left['budget_ms']==best['budget']
   assert r['left']['policy_sha256']==selection['checkpoints'][str(left['seed'])]['sha256']
   if r['right']['spec']['method']=='zero':assert r['right']['spec']['alpha']==0 and r['right']['policy_sha256']==r['left']['policy_sha256']
  for name in ['old','zero']:assert pooled([r for r in results if r['right']['spec']['method']==name])==report[name]
 overlap=len(source_hashes&eval_hashes);assert overlap==0
 write(ROOT/'audit.json',dict(pass_=True,source_games=len(sources),records=len(records),teacher_replayed_states=replayed,split_overlap=overlaps,source_eval_overlap=overlap,selection_checked=True,oracle_choice_set_sha256=hashlib.sha256(json.dumps(selected_hashes).encode()).hexdigest(),evaluation=eval_report,seconds=time.perf_counter()-start))
 print('AUDIT PASS',len(records),eval_report['games'],flush=True)
if __name__=='__main__':main()
