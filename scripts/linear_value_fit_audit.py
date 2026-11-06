"""Coordinate-oracle replay, terminal reward/value target and state-index audit."""
import argparse,copy,gzip,hashlib,importlib.util,json,random
from pathlib import Path
import numpy as np
from blokus.engine import State,Move,PIECES,legal_moves,play,cells_of
from blokus.linear_rl import prefix
from blokus.linear_value import Agent,value_features,outcome
from blokus.linear_train2 import state_hash,adam
from scripts.linear_value_diagnose import unpack,packed,ordered
from scripts import linear_value_fit_eval as ev
R=ev.R
spec=importlib.util.spec_from_file_location('coordinate_oracle','artifacts/secretary-review/blokus_reference.py');ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
SHAPES=[tuple(ref.orientations(p)) for p in PIECES]
class Oracle:
 def __init__(self,s):self.cells=[set(cells_of(x)) for x in s.boards];self.left=[{i for i in range(21) if b>>i&1} for b in s.remaining];self.last=s.last.copy();self.turn=s.turn;self.passes=s.passes
 def any_move(self):
  p=self.turn;own=self.cells[p];opp=self.cells[1-p];occupied=own|opp;anchors=({(x+dx,y+dy) for x,y in own for dx,dy in ref.DIAG} if own else {ref.STARTS[p]})-occupied
  anchors={c for c in anchors if 0<=c[0]<14 and 0<=c[1]<14 and not any((c[0]+dx,c[1]+dy) in own for dx,dy in ref.ORTH)}
  for piece in self.left[p]:
   for shape in SHAPES[piece]:
    for x,y in anchors:
     for a,b in shape:
      cs={(u+x-a,v+y-b) for u,v in shape}
      if ref.valid(cs,own,opp,ref.STARTS[p]):return True
  return False
 def play(self,m):
  p=self.turn;assert self.passes<2
  if m is None:assert not self.any_move();self.passes+=1
  else:
   assert m.piece in self.left[p];cs=set(m.cells);assert ref.normalize(cs) in SHAPES[m.piece];assert ref.valid(cs,self.cells[p],self.cells[1-p],ref.STARTS[p]);self.cells[p]|=cs;self.left[p].remove(m.piece);self.last[p]=m.piece;self.passes=0
  self.turn=1-p
 def scores(self):return [(20 if self.last[p]==0 else 15) if not self.left[p] else -sum(len(PIECES[i]) for i in self.left[p]) for p in (0,1)]
 def check(self,s):
  assert self.cells==[set(cells_of(x)) for x in s.boards] and self.turn==s.turn and self.passes==s.passes and self.scores()==[s.score(0),s.score(1)]
def start(seed):
 s=State();o=Oracle(s);rng=random.Random(seed)
 for _ in range(4):
  qs=legal_moves(s);m=rng.choice(qs).move if qs else None;o.play(m);play(s,m,validate=False)
 o.check(s);return s,o
def replay(g,visited=None,features=False):
 s,o=start(g['prefix_seed']);phis=[]
 for row in g['moves']:
  if visited is not None:visited.add(state_hash(s))
  if features and s.turn==g['seat']:phis.append(value_features(s))
  m=Move(*row) if row else None;o.play(m);play(s,m,validate=False)
 if visited is not None:visited.add(state_hash(s))
 o.check(s);assert s.done and o.scores()==g['scores'];d=o.scores()[g['seat']]-o.scores()[1-g['seat']];y=float((d>0)-(d<0))
 if 'reward' in g:assert g['reward']==y
 if 'difference' in g:assert g['difference']==d and g['point']==.5+.5*y
 if features:assert len(phis)==g['learner_states']
 return s,phis,y

def train():
 plan=json.loads((R/'fixed-plan.json').read_text());visited=set();rows=[];forbidden=set(json.loads((R/'exclusions.json').read_text())['states'])
 for arm in plan['arms']:
  with gzip.open(R/f'pilot-{arm}.json.gz','rt') as f:d=json.load(f)
  for g in d['games']:replay(g,visited)
  for seed in plan['seeds']:
   folder=R/f'{arm}-{seed}';source=Agent.load(ev.source('original',seed));w=source.value_weights.copy();m=np.zeros_like(w);v=m.copy();count=0;balance={};states={}
   with gzip.open(folder/'games.jsonl.gz','rt') as f:
    for line in f:
     batch=json.loads(line);phis=[];ys=[]
     for g in batch['games']:
      assert state_hash(prefix(g['prefix_seed'])) not in forbidden;_,local,y=replay(g,visited,True);phis+=local;ys += [y]*len(local);count+=1;k=f"{g['kind']}/{g['seat']}";balance[k]=balance.get(k,0)+1;states[k]=states.get(k,0)+len(local)
     phi=np.asarray(phis);pred=np.sum(phi*w,axis=(1,2));grad=np.mean((np.array(ys)-pred)[:,None,None]*phi,axis=0);adam(w,grad,m,v,batch['iteration'],.01)
   final=json.loads((folder/'state.json').read_text());assert np.array_equal(w,final['value']) and np.array_equal(m,final['m']) and np.array_equal(v,final['v']);assert np.array_equal(source.weights,final['policy']);assert count==plan['games_per_seed_arm'];rows.append(dict(seed=seed,arm=arm,games=count,balance=balance,learner_states=states,full_update_replay_exact=True))
 old=np.load('/tmp/linear-value-training.npy');new=np.array(sorted(bytes.fromhex(h) for h in visited),dtype='S32');a=np.unique(np.concatenate([old,new]));summary=json.loads((R/'training-summary.json').read_text());assert hashlib.sha256(old.tobytes()).hexdigest()==summary['old_training_sha256'];assert hashlib.sha256(new.tobytes()).hexdigest()==summary['new_sha256'];assert hashlib.sha256(a.tobytes()).hexdigest()==summary['union_sha256'];np.save('/tmp/linear-retrain-training.npy',a)
 ev.write('training-audit.json',dict(pass_=True,rows=rows,new_visited=len(new),union_states=len(a),coordinate_oracle=True,full_update_replay=True));print('TRAIN AUDIT PASS',flush=True)

def diagnostic():
 with gzip.open(R/'diagnostic.json.gz','rt') as f:d=json.load(f)
 roots=set();count=0
 for g in d['games']:
  a=Agent.load(ev.source('original',g['seed']));s,o=start(g['prefix_seed']);pred=[]
  for row in g['moves']:
   roots.add(state_hash(s))
   if s.turn==g['seat']:pred.append(a.value(s))
   m=Move(*row) if row else None;o.play(m);play(s,m,validate=False)
  o.check(s);assert s.done and o.scores()==g['scores'];assert outcome(s,g['seat'])==g['reward'];assert pred==[x['prediction'] for x in g['predictions']]
 for root in d['roots']:
  assert state_hash(unpack(root['state'])) in roots;a=Agent.load(ev.source('original',root['seed']))
  root_state=unpack(root['state']);root_moves=ordered(a,root_state,13)
  for candidate in root['candidates']:
   chosen=Move(*candidate['move']) if candidate['move'] else None;assert chosen==root_moves[candidate['rank']]
   middle=copy.deepcopy(root_state);play(middle,chosen);expected=[]
   if middle.done:expected=[middle]
   else:
    for reply in ordered(a,middle,3):child=copy.deepcopy(middle);play(child,reply);expected.append(child)
   assert len(expected)==len(candidate['leaves'])
   for actual,leaf in zip(expected,candidate['leaves']):
    assert packed(actual)==leaf['state']
    state=unpack(leaf['state']);assert a.value(state)==leaf['prediction']
    for rollout in leaf['rollouts']:
     t=copy.deepcopy(state);o=Oracle(t)
     for row in rollout['moves']:
      m=Move(*row) if row else None;o.play(m);play(t,m,validate=False)
     o.check(t);assert t.done and o.scores()==rollout['scores'];assert rollout['reward']==outcome(t,state.turn);count+=1
 ev.write('diagnostic-audit.json',dict(pass_=True,games=len(d['games']),roots=len(d['roots']),leaf_rollouts=count,coordinate_oracle=True,labels_terminal_wdl=True));print('DIAG AUDIT PASS',flush=True)

def evaluation():
 from scripts.linear_value_eval import contains
 from scripts.linear_hybrid_run import cluster
 ev.setup();a=np.load('/tmp/linear-retrain-training.npy',mmap_mode='r');rows=[]
 for phase in ('validation','confirmation'):
  if not (R/f'{phase}-summary.json').exists():continue
  bank=json.loads((R/f'{phase}-prefixes.json').read_text());assert bank['sha256']==hashlib.sha256(json.dumps(bank['prefixes'],sort_keys=True).encode()).hexdigest();starts={r['seed']:r['hash'] for r in bank['prefixes']};summary=json.loads((R/f'{phase}-summary.json').read_text())
  for pair,expected in summary.items():
   left,right=pair.split('/');matches=[]
   for seed in ev.SEEDS:
    d=json.loads((R/f'{phase}-{left}-{right}-{seed}.json').read_text());assert d['prefix_sha256']==bank['sha256'];assert d['left_model_sha256']==ev.model_sha(left,seed) and d['right_model_sha256']==ev.model_sha(right,seed);visits=set();seen=set()
    for g in d['games']:
     assert state_hash(prefix(g['prefix_seed']))==starts[g['prefix_seed']];assert (g['prefix_seed'],g['seat']) not in seen;seen.add((g['prefix_seed'],g['seat']));replay(g,visits)
    assert len(seen)==bank['count']*2;overlap=sum(contains(a,h) for h in visits);assert overlap==0
    for stat in d['left_search']+d['right_search']:
     if stat.get('forced_pass'):continue
     if stat['complete']:assert stat['completed_roots']==stat['root_candidates'] and len(stat['values'])==stat['root_candidates']
     if stat['fallback']:assert not stat['values']
    matches.append(d);rows.append(dict(phase=phase,pair=pair,seed=seed,games=len(d['games']),training_overlap=overlap))
   assert cluster(matches)==expected
 ev.write('evaluation-audit.json',dict(pass_=True,coordinate_oracle=True,rows=rows,arithmetic=True,model_hashes=True));print('EVAL AUDIT PASS',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['train','diagnostic','evaluation']);a=p.parse_args();{'train':train,'diagnostic':diagnostic,'evaluation':evaluation}[a.mode]()
