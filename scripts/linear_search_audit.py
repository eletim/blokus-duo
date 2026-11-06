"""Independent coordinate replay and exact arithmetic audit (oracle reused unchanged)."""
import copy,hashlib,importlib.util,json,random,gzip
from pathlib import Path
import numpy as np
from blokus.engine import State,Move,PIECES,legal_moves,play,cells_of
from blokus.linear_rl import prefix
from blokus.linear_train2 import state_hash
from blokus.linear_value import value_features
from scripts import linear_search_run as ev
from scripts import linear_value_eval as base
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

def audit():
 ev.setup();idx=ev.index();excluded=set(json.loads((R/'exclusions.json').read_text())['states']);allstarts=set();rows=[];allvisits=set();phasevisits={}
 with gzip.open(R/'pilot.json.gz','rt') as f:pilot=json.load(f)
 for g in pilot['games']:replay(g)
 for r in pilot['states']:
  s=State(*r['state']);assert state_hash(s)==r['hash'] and not base.contains(idx,r['hash'])
  for name,result in r['results'].items():
   w,rp=map(int,name.split('x'));a=ev.Agent.load(ev.source(r['seed']));bot=ev.CompletedSearch(a,ms=0,width=w,replies=rp,base_width=w,base_replies=rp);m=bot.act(s,random.Random(0));assert result['move']==([m.piece,m.orientation,m.row,m.col] if m else None);assert result['stats'].get('values',[])==bot.stats[-1].get('values',[])
 for phase in ('validation','confirmation'):
  if not (R/f'{phase}-prefixes.json').exists():continue
  bank=json.loads((R/f'{phase}-prefixes.json').read_text());starts={x['seed']:x['hash'] for x in bank['prefixes']};assert len(starts)==bank['count'] and not allstarts&set(starts.values());allstarts.update(starts.values());assert not excluded&set(starts.values());assert bank['sha256']==hashlib.sha256(json.dumps(bank['prefixes'],sort_keys=True).encode()).hexdigest()
  groups={};expected=json.loads((R/f'{phase}-summary.json').read_text())
  for p in R.glob(f'{phase}-*.json*'):
   d=ev.read_match(p)
   if not isinstance(d.get('games'),list):continue
   assert d['prefix_sha256']==bank['sha256'] and d['left_model_sha256']==ev.model_sha(d['left'],d['seed']) and d['right_model_sha256']==ev.model_sha(d['right'],d['seed'])
   visited=set();seats=set()
   for g in d['games']:
    assert state_hash(prefix(g['prefix_seed']))==starts[g['prefix_seed']];assert (g['prefix_seed'],g['seat']) not in seats;seats.add((g['prefix_seed'],g['seat']));replay(g,visited)
   allvisits.update(visited);phasevisits.setdefault(phase,set()).update(visited)
   overlaps=sum(base.contains(idx,h) for h in visited);assert overlaps==0;assert len(seats)==bank['count']*2
   for st in d['left_search']+d['right_search']:
    if st.get('forced_pass'):continue
    if st['fallback']:assert st['values']==[]
    if st['complete']:assert len(st['values'])==st['root_candidates']==st['completed_roots']
    if 'attempts' in st:
     comp=[a for a in st['attempts'] if a['complete']];assert all(a['roots']==a['completed_roots'] for a in comp)
     assert st['depth']==(comp[-1]['depth'] if comp else 0)
     if st['roots']:assert st['baseline'] in st['roots'] and len(set(map(tuple,st['roots'])))==len(st['roots'])
   groups.setdefault(d['left']+'/'+d['right'],[]).append(d);rows.append(dict(path=str(p),games=len(d['games']),unique_visits=len(visited),training_overlap=int(overlaps)))
  for k,ds in groups.items():assert ev.cluster(sorted(ds,key=lambda x:x['seed']))==expected[k]
 selection=json.loads((R/'selection.json').read_text());assert selection['settings_sha256']==base.sha(R/'settings.json');assert all(base.sha(x['path'])==x['sha256'] for x in selection['model_hashes'])
 costs=json.loads((R/'validation-costs.json').read_text());summary=json.loads((R/'validation-summary.json').read_text());choices=[]
 for c in ('B','C'):
  t=costs[c+'/gen3']['left'];g=summary[c+'/gen3']['point'];direct=summary[c+'/A']['point'];ok=direct>=.5 and g>=summary['A/gen3']['point']-.02 and t['target_complete']>=.8 and t['greedy_fallback']<.1;assert ok==selection['gates'][c]['pass_']
  if ok:choices.append((g,direct,-summary[c+'/gen3']['left_ms'],c=='B',c))
 assert selection['selected']==(max(choices)[-1] if choices else None)
 if 'confirmation' in phasevisits:assert not phasevisits['validation']&phasevisits['confirmation']
 extra=[];keys=np.array(sorted(bytes.fromhex(h) for h in allvisits),dtype='S32')
 for path in Path('/tmp').glob('ppo-*-training.npy'):
  a=np.load(path,mmap_mode='r');assert a.dtype==keys.dtype
  ix=np.searchsorted(a,keys);inside=ix<len(a);hits=int(np.sum(a[ix[inside]]==keys[inside]));assert hits==0
  extra.append(dict(path=str(path),states=len(a),raw_sha256=hashlib.sha256(memoryview(a).cast('B')).hexdigest(),overlap=hits))
 ev.write('additional-ppo-separation.json',dict(post_evaluation_only=True,no_prefix_resampling=True,unique_evaluation_states=len(keys),indices=extra))
 ev.write('audit.json',dict(pass_=True,coordinate_oracle=True,all_games_replayed=True,arithmetic_recomputed=True,selection_recomputed=True,training_overlap=0,models_unchanged=True,pilot_value_replay=True,rows=rows));print('AUDIT PASS',sum(x['games'] for x in rows),flush=True)
if __name__=='__main__':audit()
