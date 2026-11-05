"""Replay nested source games, labels, per-game metrics and registered gates."""
import sys,json,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from blokus.scaling_data import ROOT,load,strata
from blokus.scaling_experiment import aggregate
from blokus.engine import State,Move,legal_moves,is_legal
from blokus.ai import seed_all,features,heuristic
from blokus.research import clone,sha
from blokus.visited_data import state_hash
from blokus.value_search import ValueEvaluator,encode
from blokus.deep_search import TreeSearch
from blokus.aligned_data import BASE
from blokus.node_diagnostic import child,row
from blokus.root_policy import inputs
from blokus.value_experiment import write
from audit_root_distillation import replay

def main():
 seed_all(1100);begin=time.perf_counter();sources=load('source-games');records=load('teacher-labels');states={};game_hashes=[];bygame={}
 import hashlib
 for g in sources:
  assert {k:g[k] for k in ['pool','seat','fold']}==strata(g['index']);s=State();bygame[g['game']]=g
  for ply,m in enumerate(g['moves']):states[g['game'],ply]=state_hash(s);replay(s,m)
  assert s.done and g['scores']==[s.score(0),s.score(1)];game_hashes.append(hashlib.sha256(json.dumps(g['moves']).encode()).hexdigest())
 assert len(set(g['game'] for g in sources))==len(sources)
 allstates=set();tree=TreeSearch(ValueEvaluator(BASE,'cuda',.5,24),width=3,tt_limit=0);replayed=0
 for j,r in enumerate(records):
  s=clone(State(**r['state']));assert state_hash(s)==r['state_hash']==states[r['game'],r['ply']];assert r['state_hash'] not in allstates;allstates.add(r['state_hash'])
  qs=legal_moves(s);order=np.argsort(-heuristic(features(s,qs)[0]),kind='stable')[:128];qs=[qs[int(i)] for i in order];assert [row(q.move) for q in qs]==r['moves'] and all(is_legal(s,q.move) for q in qs)
  assert set(r['top']).isdisjoint(r['random']);assert sorted(r['top'])==sorted(np.argsort(-np.array(r['initial']),kind='stable')[:4].tolist())
  if j%100==0:
   for i in r['top']+r['random']:
    c=child(s,qs[i].move);assert abs(-tree.search(c,1)-r['initial'][i])<1e-5;assert abs(-tree.search(c,3)-r['final'][i])<1e-5
   replayed+=1
 bias=json.loads((ROOT/'bias-diagnostic.json').read_text());byhash={r['state_hash']:r for r in records}
 for ck in bias['checkpoints'].values():assert sha(ck['path'])==ck['sha256']
 for r in bias['rows']:
  original=byhash[r['state_hash']];s=State(**original['state']);assert r['depth']==4
  for index,v in r['truth'].items():assert abs(-tree.search(child(s,Move(*original['moves'][int(index)])),3)-v)<1e-5
  best=max(r['truth'].values())
  for name,index in r['picks'].items():assert abs(r['regret'][name]-(best-r['truth'][str(index)]))<1e-10
 reports=[];previous=None
 for size in [240,960,2400]:
  path=ROOT/f'data-{size}.npz'
  if not path.exists():continue
  raw=np.load(path);d={k:raw[k] for k in raw.files};rs=[r for r in records if r['bank'] or r['index']<size*3//2];assert len(rs)==len(d['game'])
  for j,r in enumerate(rs):
   s=State(**r['state']);by={q.move:q for q in legal_moves(s)};qs=[by[Move(*m)] for m in r['moves']];x,_=inputs(s,qs);b,st=encode(s);n=len(qs)
   assert np.array_equal(d['board'][j],b) and np.array_equal(d['state'][j],st) and np.array_equal(d['x'][j,:n],x)
   expected=np.array([v is not None for v in r['final']]);assert np.array_equal(d['mask'][j,:n],expected) and not d['mask'][j,n:].any();assert np.isnan(d['target'][j,~d['mask'][j]]).all()
   for i in np.flatnonzero(expected):assert abs(d['target'][j,i]-(r['final'][i]-r['initial'][i]))<1e-5
   assert d['game'][j]==r['game'] and d['fold'][j]==r['fold'] and d['bank'][j]==r['bank']
  if previous is not None:
   for k in d:assert np.array_equal(previous[k],d[k][:len(previous[k])],equal_nan=True)
  previous=d
  for fold in range(3):
   tr=set(d['game'][~d['bank']&(d['fold']!=fold)].tolist());te=set(d['game'][d['bank']&(d['fold']==fold)].tolist());assert not tr&te
  curve=json.loads((ROOT/f'curve-{size}.json').read_text())
  for r in curve:
   assert aggregate(size,r['condition'],r['step'])==r
   by={}
   for f in range(3):
    for seed in [1101,1102,1103]:
     run=json.loads((ROOT/f"n{size}-f{f}-s{seed}-{r['condition']}/result.json").read_text());assert run['config']['data_sha256']==sha(path)
     item=next(t for t in run['logs'] if t['step']==r['step'])
     expected={str(g) for g in d['game'][d['bank']&(d['fold']==f)].tolist()};assert set(item['test_games'])==expected
     for g,v in item['test_games'].items():by.setdefault(g,[]).append(v)
   assert all(len(v)==3 for v in by.values())
   means=np.array([sum(np.array(v))/3 for v in by.values()]);diff=np.column_stack((means[:,1]-means[:,0],means[:,3]-means[:,2]))
   ix=np.random.default_rng(111777).integers(len(means),size=(4000,len(means)));ci=np.quantile(diff[ix].mean(axis=1),[.025,.975],axis=0)
   assert np.allclose(ci[:,0],r['mse_improvement_ci95']) and np.allclose(ci[:,1],r['regret_improvement_ci95'])
   assert np.allclose(means.mean(axis=0),[r['mean'][k] for k in ['mse','zero_mse','regret','zero_regret']])
  reports.append(dict(size=size,data_sha256=sha(path),rows=len(rs),split_hash=hashlib.sha256(json.dumps([(r['game'],r['fold'],r['bank'],r['state_hash']) for r in rs]).encode()).hexdigest()))
 init={json.loads(p.read_text())['initial_hash'] for p in ROOT.glob('n*/config.json')};assert len(init)==1
 # Gate must never be inferred from game wins; only registered heldout teacher metrics.
 gate=json.loads((ROOT/'decision.json').read_text());last=max(r['size'] for r in reports);assert gate['size']==last
 accepted=[c for c in ['plain','decay'] if aggregate(last,c)['advance']];assert gate['accepted_conditions']==accepted and gate['advance']==bool(accepted)
 result=dict(pass_=True,unique_source_game_ids=len(sources),unique_complete_move_sequences=len(set(game_hashes)),roots=len(records),root_hash_duplicates=0,teacher_replayed_states=replayed,bias_replayed_states=len(bias['rows']),datasets=reports,shared_initial_hash=next(iter(init)),seconds=time.perf_counter()-begin,note='Implementation-side replay and arithmetic checks; no external independent audit claimed.')
 write(ROOT/'audit.json',result);print(result,flush=True)
if __name__=='__main__':main()
