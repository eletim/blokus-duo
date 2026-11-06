import copy,json,os,subprocess,sys,random
import numpy as np
from blokus.engine import State,play,legal_moves,frontier,cells_of
from blokus.linear_value import Agent,value_features
from blokus.linear_tree import CompletedSearch
from blokus.tuple_value import TupleAgent,codes,geometry,residual,table_gradient,Learner,SIZE

def position():
 s=State();a=Agent('old');rng=random.Random(17)
 for _ in range(18):play(s,a.act(s,rng))
 return s

def transform(s,transpose=False):
 t=copy.deepcopy(s)
 def bits(b):return sum(1<<((c*14+r) if transpose else ((13-r)*14+13-c)) for r,c in cells_of(b))
 t.boards=[bits(b) for b in s.boards]
 if not transpose:t.boards=t.boards[::-1];t.remaining=t.remaining[::-1];t.last=t.last[::-1];t.turn=1-t.turn
 return t

def test_codes_symmetry_boundary_and_sign():
 s=position();before=copy.deepcopy(s);ids=codes(s);assert ids.shape==(312,) and ids.min()>=0 and ids.max()<SIZE
 windows,groups,_,_,swap=geometry();assert windows.min()==0 and windows.max()==195 and set(groups)=={0,1}
 table=np.random.default_rng(2).normal(size=SIZE);a=TupleAgent(Agent('old').weights,np.ones((3,8))*.1,table)
 t=copy.deepcopy(s);t.turn=1-t.turn
 assert abs(a.value(s)+a.value(t))<1e-12
 for t in (transform(s,True),transform(s,False)):assert abs(a.value(s)-a.value(t))<1e-12
 assert s==before
 assert frontier(State(),0)[0]==1<<(4*14+4) and frontier(State(),1)[0]==1<<(9*14+9)
 # No wraparound across board edges, and every window has six distinct cells.
 assert all(len(set(w))==6 for w in windows)

def test_initial_identity_terminal_pass_and_remaining():
 base=Agent('old',value=np.ones((3,8))*.2);a=TupleAgent(base.weights,base.value_weights);s=position()
 assert a.value(s)==base.value(s)
 t=copy.deepcopy(s);t.remaining[0]=0;t.last[0]=0;u=copy.deepcopy(t);u.last[0]=1;assert not np.array_equal(value_features(t),value_features(u))
 t=State(remaining=[0,0],last=[0,1]);play(t,None);assert not t.done;play(t,None);assert a.value(t)==1;t.turn=1;assert a.value(t)==-1
 b=CompletedSearch(a,ms=.000001,width=24,replies=8);before=copy.deepcopy(s);m=b.act(s,random.Random(0));assert m==a.act(s,random.Random(0)) and s==before

def test_tuple_gradient_and_adam_formula():
 s=position();ids=np.stack([codes(s),codes(transform(s,True))]);table=np.random.default_rng(3).normal(size=SIZE)*.01;y=np.array([1.,0.]);error=y-residual(table,ids);g=table_gradient(ids,error)
 for i in np.flatnonzero(g)[:8]:
  d=np.zeros(SIZE);d[i]=1e-5
  objective=lambda w:-.5*np.mean((y-residual(w,ids))**2)
  assert abs((objective(table+d)-objective(table-d))/2e-5-g[i])<1e-8
 phi=np.zeros((2,24));l=Learner(Agent('old'),'T',1,epochs=1,batch=2);l.update(phi,ids,y)
 expected=.01*g*0 # independently first-step gradient at zero
 gg=table_gradient(ids,y);gg*=min(1,5/max(np.linalg.norm(gg),1e-12));expected=.01*gg/(np.abs(gg)+1e-8)
 assert np.allclose(l.w,expected,atol=1e-12)

def test_separate_process_resume_and_export(tmp_path):
 r=np.random.default_rng(9);phi=r.normal(size=(12,24))*.1;ids=r.integers(0,SIZE,size=(12,312),dtype=np.uint16);y=r.choice([-1.,0.,1.],12);np.savez(tmp_path/'d.npz',phi=phi,ids=ids,y=y)
 for arm in ('L','T'):
  a=Learner(Agent('old'),arm,12,epochs=4,batch=5);b=Learner(Agent('old'),arm,12,epochs=4,batch=5)
  for _ in range(4):a.update(phi,ids,y)
  for _ in range(2):b.update(phi,ids,y)
  a.save(tmp_path/'a.json');b.save(tmp_path/'b.json')
  code="from blokus.tuple_value import Learner; import numpy as np; from pathlib import Path; p=Path(__import__('sys').argv[1]); d=np.load(p/'d.npz'); b=Learner.load(p/'b.json'); [b.update(d['phi'],d['ids'],d['y']) for _ in range(2)]; b.save(p/'b.json')"
  subprocess.run([sys.executable,'-c',code,str(tmp_path)],check=True,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
  assert json.loads((tmp_path/'a.json').read_text())==json.loads((tmp_path/'b.json').read_text())
 a.agent().save(tmp_path/'model.json');loaded=TupleAgent.load(tmp_path/'model.json');assert np.array_equal(a.w,loaded.table)

def test_tuple_small_end_exact():
 from test_linear_tree import exact
 s=State(remaining=[1,1]);a=TupleAgent(Agent('old').weights,np.zeros((3,8)),np.ones(SIZE));b=CompletedSearch(a,ms=0,width=None,replies=None,depth=4,base_width=None,base_replies=None)
 v,best=exact(s);assert b.act(s,random.Random(2)) in best and max(b.stats[-1]['values'])==v

def test_independent_slow_codes():
 s=position();own=set(cells_of(s.boards[s.turn]));opp=set(cells_of(s.boards[1-s.turn]));ids=[]
 from blokus.tuple_value import phase
 for h,w in ((2,3),(3,2)):
  for r in range(15-h):
   for c in range(15-w):
    points=[(r+i,c+j) for i in range(2) for j in range(3)] if h==2 else [(r+j,c+i) for i in range(2) for j in range(3)]
    digits=[1 if q in own else 2 if q in opp else 0 for q in points];forward=sum(d*3**i for i,d in enumerate(digits));reverse=sum(d*3**i for i,d in enumerate(digits[::-1]));edge=int(r==0 or c==0 or r+h==14 or c+w==14);ids.append(min(forward,reverse)+edge*729+phase(s)*1458)
 assert np.array_equal(codes(s),ids)
