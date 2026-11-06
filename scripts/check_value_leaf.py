import sys,json,copy,random,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from blokus.engine import State,play,legal_moves
from blokus.ppo_policy import PPOAgent,collate
from blokus.ppo_value_leaf import value_inputs,value_only
from blokus.ppo_search import SearchAgent
from blokus.ppo_search_staged import ValueLeafSearch,key
R=Path('artifacts/gen3-value-leaf');torch.set_num_threads(1);r=random.Random(8280100);states=[]
for game in range(8):
 s=State();hist=[]
 while not s.done:
  hist.append(copy.deepcopy(s));qs=legal_moves(s);play(s,r.choice(qs).move if qs else None,validate=False)
 for i in np.linspace(0,len(hist)-1,12,dtype=int):states.append(hist[i])
states.extend([State(remaining=[0,0],last=[0,1]),State(remaining=[1,0],last=[None,1],turn=1),State(remaining=[0,0],last=[0,1],passes=2)])
maxdiff=0.;values=0;trees=[];t=time.perf_counter()
for gen,seeds in [('gen3',[1751401,1751402,1751403]),('gen2',[1401,1402,1403])]:
 for seed in seeds:
  for backend in ('python','native'):
   p=PPOAgent(seed,'cpu',generation=gen,backend=backend)
   for s in states:
    before=key(s)
    if not s.done:
     obs,_=p._observation(s);b,st=value_inputs(s,p.channels);assert np.array_equal(b,obs[0]) and np.array_equal(st,obs[1])
     with torch.inference_mode():_,v=p.model(*collate([obs],'cpu'))
     v=v.item()
    else:d=s.score(s.turn)-s.score(1-s.turn);v=float(np.sign(d))
    diff=abs(v-value_only(p,s));maxdiff=max(maxdiff,diff);assert diff<=1e-7,(gen,seed,backend,diff);assert key(s)==before;values+=1
   if gen=='gen3':
    for i in (0,3,6,9,11,24,27,30,33,35,60,63,66,69,71,96,97):
     s=states[i];old=SearchAgent(p,5000);e=ValueLeafSearch(p,5000);e_cached=ValueLeafSearch(p,5000,reuse=True);before=key(s)
     moves=[a.act(s,random.Random(828010)) for a in (old,e,e_cached)]
     assert len(set(moves))==1,(seed,backend,i,moves)
     assert old.stats['depth']==e.stats['depth']==e_cached.stats['depth'];assert key(s)==before
     trees.append(dict(seed=seed,backend=backend,state_index=i,old_ms=old.stats['elapsed_ms'],E_ms=e.stats['elapsed_ms'],cached_ms=e_cached.stats['elapsed_ms'],depth=e.stats['depth']))
result=dict(pass_=True,values_checked=values,max_abs_difference=maxdiff,tolerance=1e-7,trees=trees,seconds=time.perf_counter()-t,states=[[s.boards,s.remaining,s.last,s.turn,s.passes] for s in states]);(R/'equivalence.json').write_text(json.dumps(result,indent=2)+'\n');print(values,maxdiff,len(trees),result['seconds'])
