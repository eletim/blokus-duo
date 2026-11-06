"""Coordinate replay, terminal target, split, deterministic update and match audit."""
import argparse,gzip,hashlib,json,copy,random
from pathlib import Path
import numpy as np
from blokus.engine import Move,play
from blokus.linear_value import value_features,Agent
from blokus.linear_train2 import state_hash
from blokus.tuple_value import codes,Learner,load_value,TupleAgent
from scripts.linear_search_audit import start,replay
from scripts import tuple_cycle as cycle
from scripts import tuple_eval as ev
from scripts import linear_value_eval as base
R=cycle.R

def training():
 rows=[];splits={};old=cycle.indices();excluded=set(json.loads((R/'exclusions.json').read_text())['states'])
 for kind in ('pilot','train','calibration'):
  splits[kind]=set()
  for seed in cycle.SEEDS:
   folder=R/f'{kind}-{seed}';data=dict(np.load(folder/'data.npz'));recorded=set();at=0
   with gzip.open(folder/'games.json.gz','rt') as f:games=json.load(f)
   for g in games:
    s,o=start(g['prefix_seed']);assert state_hash(s) not in excluded and all(not base.contains(a,state_hash(s)) for a in old)
    turns=[]
    for move in g['moves']:
     h=state_hash(s);recorded.add(h);turns.append(s.turn);assert h==g['states'][len(turns)-1]==data['hashes'][at]
     assert np.array_equal(data['phi'][at],value_features(s).reshape(-1)) and np.array_equal(data['codes'][at],codes(s));at+=1
     m=Move(*move) if move else None;o.play(m);play(s,m,validate=False)
    o.check(s);recorded.add(state_hash(s));assert s.done and o.scores()==g['scores'] and turns==g['turns']
    expected=[float(np.sign(o.scores()[p]-o.scores()[1-p])) for p in turns];assert expected==g['targets'] and np.array_equal(data['y'][at-len(turns):at],expected)
   assert at==len(data['y']) and not recorded&excluded;stored=np.load(folder/'visited.npy');assert np.array_equal(stored,np.array(sorted(bytes.fromhex(h) for h in recorded),dtype='S32'));assert not splits[kind]&recorded;splits[kind].update(recorded)
   rows.append(dict(kind=kind,seed=seed,games=len(games),states=at,visits=len(recorded),full_coordinate_replay=True))
   if kind=='train':
    for arm in ('L','T'):
     t=Learner(Agent.load(cycle.source(seed)),arm,seed+10000)
     while t.update(data['phi'],data['codes'],data['y']) is not None:pass
     saved=Learner.load(R/f'{arm}-{seed}-state.json')
     for k in ('w','m','v','visits','policy','base'):assert np.array_equal(getattr(t,k),getattr(saved,k))
     assert t.step==saved.step and t.rng.bit_generator.state==saved.rng.bit_generator.state
     a=load_value(ev.source(arm,seed));assert np.array_equal(a.weights,t.policy)
     if arm=='T':assert np.array_equal(a.table,t.w) and np.array_equal(a.value_weights,t.base)
     else:assert np.array_equal(a.value_weights,t.w.reshape(3,8))
 for a,b in [('pilot','train'),('pilot','calibration'),('train','calibration')]:assert not splits[a]&splits[b]
 union=set().union(*splits.values());assert np.array_equal(ev.index(),np.array(sorted(bytes.fromhex(h) for h in union),dtype='S32'))
 cycle.write('training-audit.json',dict(pass_=True,rows=rows,update_replay_bit_exact=True,policy_frozen=True,terminal_wdl=True,all_data_replayed=True,game_splits_disjoint=True));print('TRAIN AUDIT PASS',flush=True)

def evaluation():
 ev.setup();indices=cycle.indices()+[ev.index()];groups={};rows=[];visited_by_phase={};excluded=set(json.loads((R/'exclusions.json').read_text())['states'])
 for phase in ('validation','confirmation'):
  if not (R/f'{phase}-prefixes.json').exists():continue
  bank=json.loads((R/f'{phase}-prefixes.json').read_text());starts={r['seed']:r['hash'] for r in bank['prefixes']};assert not excluded&set(starts.values());assert bank['sha256']==hashlib.sha256(json.dumps(bank['prefixes'],sort_keys=True).encode()).hexdigest();groups[phase]={};visited_by_phase[phase]=set()
  for p in R.glob(f'{phase}-*.json.gz'):
   with gzip.open(p,'rt') as f:d=json.load(f)
   assert d['prefix_sha256']==bank['sha256'] and d['left_model_sha256']==ev.model_sha(d['left'],d['seed']) and d['right_model_sha256']==ev.model_sha(d['right'],d['seed'])
   visited=set();seats=set()
   for g in d['games']:
    s,_=start(g['prefix_seed']);assert state_hash(s)==starts[g['prefix_seed']];assert (g['prefix_seed'],g['seat']) not in seats;seats.add((g['prefix_seed'],g['seat']));replay(g,visited)
   assert len(seats)==bank['count']*2;visited_by_phase[phase].update(visited)
   keys=np.array(sorted(bytes.fromhex(h) for h in visited),dtype='S32')
   for idx in indices:
    ix=np.searchsorted(idx,keys);valid=ix<len(idx);assert not np.any(idx[ix[valid]]==keys[valid])
   for st in d['left_search']+d['right_search']:
    if st.get('forced_pass'):continue
    if st['complete']:assert len(st['values'])==st['root_candidates']==st['completed_roots']
    if st['fallback']:assert st['values']==[]
    if st['roots']:assert st['baseline'] in st['roots']
   groups[phase].setdefault(d['left']+'/'+d['right'],[]).append(d);rows.append(dict(path=str(p),games=len(d['games']),training_overlap=0))
  expected=json.loads((R/f'{phase}-summary.json').read_text())
  for k,ds in groups[phase].items():
   ds.sort(key=lambda x:x['seed']);summary=base.cluster(ds);summary['wall_seconds']=sum(d['wall'] for d in ds);assert summary==expected[k]
 if 'confirmation' in visited_by_phase:assert not visited_by_phase['validation']&visited_by_phase['confirmation']
 sel=json.loads((R/'selection.json').read_text());s=json.loads((R/'validation-summary.json').read_text());c=json.loads((R/'validation-costs.json').read_text())['T/gen3']['left'];gate=s['T/L']['point']>=.5 and s['T/F']['point']>=.5 and s['T/gen3']['point']>=s['F/gen3']['point']-.02 and c['target_complete']>=.8 and c['greedy_fallback']<.1;assert gate==sel['proceed'];assert all(base.sha(x['path'])==x['sha256'] for x in sel['hashes'])
 cycle.write('evaluation-audit.json',dict(pass_=True,rows=rows,coordinate_oracle=True,arithmetic=True,training_overlap=0,selection_recomputed=True,models_unchanged=True));print('EVAL AUDIT PASS',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['training','evaluation']);a=p.parse_args();training() if a.mode=='training' else evaluation()
