"""Development-only costs and fixed greedy-continuation WDL calibration (not minimax truth)."""
import sys,time,random,copy,json,cProfile,pstats
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from blokus.engine import State,legal_moves,play
from blokus.ppo_policy import PPOAgent,collate
from blokus.ppo_search import SearchAgent
from blokus.ppo_search_batch import BatchSearch,child,terminal
R=Path('artifacts/gen3-search-audit');torch.set_num_threads(1)
rows=[];cost=[];roots=[];p=PPOAgent(1751401,'cpu');r=random.Random(82790000)
def predict(s):
 if s.done:return terminal(s),[]
 o,qs=p._observation(s)
 with torch.inference_mode():l,v=p.model(*collate([o],'cpu'))
 return v.item(),[qs[i].move for i in l[0].argsort(descending=True).tolist()] if qs else [None]
for game in range(6):
 s=State()
 for _ in range(6):
  qs=legal_moves(s);play(s,r.choice(qs).move if qs else None,validate=False)
 for _ in range((game%3)*8):
  if s.done:break
  play(s,p.act(s,r),validate=False)
 roots.append(copy.deepcopy(s))
 v,moves=predict(s)
 for origin,z in [('root',s)]+[('leaf',child(child(s,m),n)) for m in moves[:2] for n in predict(child(s,m))[1][:2]]:
  predicted,_=predict(z);x=copy.deepcopy(z);side=x.turn
  while not x.done:play(x,p.act(x,r),validate=False)
  d=x.score(side)-x.score(1-side);target=float(np.sign(d));rows.append(dict(game=game,phase=game%3,origin=origin,value=predicted,target=target))
 for ms in (500,2000):
  _,order=predict(s)
  for label,a in [('old',SearchAgent(p,ms)),('B_same_width',BatchSearch(p,ms,root_width=4)),('B',BatchSearch(p,ms)),('C',BatchSearch(p,ms,exact=True))]:
   if '--roots-only' in sys.argv:
    p.act(s,r);continue
   m=a.act(s,r);cost.append(dict(game=game,budget=ms,method=label,policy_rank=order.index(m) if m in order else None,**a.stats))
if '--roots-only' in sys.argv:
 old=json.loads((R/'diagnostic.json').read_text());assert old['rows']==rows
 (R/'diagnostic-roots.json').write_text(json.dumps(dict(states=[[s.boards,s.remaining,s.last,s.turn,s.passes] for s in roots],calibration_exact_replay=True),indent=2)+'\n');sys.exit(0)
prof=cProfile.Profile();prof.enable()
for s in roots:SearchAgent(p,500).act(s,r)
prof.disable();ps=pstats.Stats(prof);entries=[]
for (file,line,name),(cc,nc,tt,ct,callers) in ps.stats.items():
 if name in ('legal_moves','features','encode','_candidate_inputs','collate','forward','act','evaluate'):
  entries.append(dict(file=file,line=line,function=name,calls=nc,self_seconds=tt,cumulative_seconds=ct))
summary={}
for origin in ('root','leaf'):
 a=[x for x in rows if x['origin']==origin];v=np.array([x['value'] for x in a]);t=np.array([x['target'] for x in a]);summary[origin]=dict(n=len(a),mse=float(((v-t)**2).mean()),sign_accuracy=float((np.sign(v)==t).mean()),predicted_range=[float(v.min()),float(v.max())],by_phase={str(k):dict(n=sum(x['phase']==k for x in a),mse=float(np.mean([(x['value']-x['target'])**2 for x in a if x['phase']==k]))) for k in range(3)})
(R/'diagnostic.json').write_text(json.dumps(dict(calibration=summary,rows=rows,cost=cost,profile=entries,warning='six independent games; deterministic fixed policy continuation WDL, not minimax labels or human games; development only'),indent=2)+'\n')
