"""Source replay, spatial input, teacher sample, split and arithmetic checks."""
import sys,json,time,hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from blokus.mask_data import ROOT,load
from blokus.mask_experiment import aggregate
from blokus.engine import State,Move,legal_moves,is_legal
from blokus.ai import seed_all
from blokus.visited_data import state_hash
from blokus.value_search import ValueEvaluator,encode
from blokus.deep_search import TreeSearch
from blokus.aligned_data import BASE
from blokus.node_diagnostic import child
from blokus.root_policy import inputs
from blokus.value_experiment import write
from blokus.research import sha
from audit_root_distillation import replay

def main():
 seed_all(1100);begin=time.perf_counter();sources=load('source-games');records=load('teacher-labels');states={};gamehash={};allphase=[set(),set()]
 for g in sources:
  assert g['game']<3600 if not g['bank'] else 200000<=g['game']<200360
  s=State()
  for ply,m in enumerate(g['moves']):
   states[g['game'],ply]=state_hash(s)
   if ply>=8:allphase[int(g['bank'])].add(state_hash(s))
   replay(s,m)
  assert s.done and g['scores']==[s.score(0),s.score(1)];gamehash[g['game']]=hashlib.sha256(json.dumps(g['moves']).encode()).hexdigest()
 assert len(gamehash)==len(sources)==len(set(gamehash.values()))
 raw=np.load(ROOT/'compact.npz');d={k:raw[k] for k in raw.files};seen=set();tree=TreeSearch(ValueEvaluator(BASE,'cuda',.5,24),width=3,tt_limit=0);replayed=0
 for j,r in enumerate(records):
  s=State(**r['state']);assert state_hash(s)==r['state_hash']==states[r['game'],r['ply']];assert r['state_hash'] not in seen;seen.add(r['state_hash'])
  by={q.move:q for q in legal_moves(s)};ix=[i for i,v in enumerate(r['final']) if v is not None];qs=[by[Move(*r['moves'][i])] for i in ix];assert all(is_legal(s,q.move) for q in qs)
  x,_=inputs(s,qs);b,st=encode(s);n=len(ix)
  assert np.array_equal(b,d['board'][j]) and np.array_equal(st,d['state'][j]);assert np.array_equal(x,d['x'][j,:n,:39]);assert d['mask'][j].tolist()==[True]*n+[False]*(6-n)
  for k,i in enumerate(ix):
   cells=np.zeros(196)
   for y,col in qs[k].move.cells:cells[y*14+col]=1
   assert np.array_equal(cells,d['x'][j,k,39:]);assert abs(d['target'][j,k]-(r['final'][i]-r['initial'][i]))<1e-5
   if j%100==0:
    c=child(s,qs[k].move);assert abs(-tree.search(c,1)-r['initial'][i])<1e-5;assert abs(-tree.search(c,3)-r['final'][i])<1e-5
  if j%100==0:replayed+=1
 for f in range(3):assert not set(d['game'][~d['bank']&(d['fold']!=f)]) & set(d['game'][d['bank']&(d['fold']==f)])
 report=json.loads((ROOT/'comparison.json').read_text())
 for r in report:
  assert r==aggregate(2400,r['condition']);by={}
  for f in range(3):
   for seed in [1101,1102,1103]:
    run=json.loads((ROOT/f"n2400-f{f}-s{seed}-{r['condition']}/result.json").read_text());assert run['config']['data_sha256']==sha(ROOT/'compact.npz');ledger=run['logs'][0]['test_games'];assert set(ledger)=={str(g) for g in d['game'][d['bank']&(d['fold']==f)]}
    for g,v in ledger.items():by.setdefault(g,[]).append(v)
  a=np.array([np.mean(v,0) for v in by.values()]);assert all(len(v)==3 for v in by.values());assert np.allclose(a.mean(0),list(r['mean'].values()))
  idx=np.random.default_rng(111777).integers(len(a),size=(4000,len(a)));delta=a[:,[1,3]]-a[:,[0,2]];ci=np.quantile(delta[idx].mean(1),[.025,.975],axis=0);assert np.allclose(ci[:,0],r['mse_improvement_ci95']) and np.allclose(ci[:,1],r['regret_improvement_ci95'])
 decision=json.loads((ROOT/'decision.json').read_text());assert decision['advance']==report[1]['advance']
 write(ROOT/'audit.json',dict(pass_=True,source_games=len(sources),unique_game_hashes=len(set(gamehash.values())),roots=len(records),teacher_replayed_roots=replayed,train_bank_root_overlap=0,all_source_states_after_ply8_overlap=len(allphase[0]&allphase[1]),split_sha256=hashlib.sha256(json.dumps([(r['game'],r['fold'],r['bank'],r['state_hash']) for r in records]).encode()).hexdigest(),seconds=time.perf_counter()-begin,note='Implementation-side replay/arithmetic audit, not external independent audit.'))
if __name__=='__main__':main()
