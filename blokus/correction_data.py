"""New-game, equal-depth masked residual labels and teacher-information pilot."""
import argparse,gzip,json,random,time
from pathlib import Path
import numpy as np
from .engine import State,legal_moves,play
from .ai import seed_all,features,heuristic
from .research import clone,sha
from .value_search import BudgetAgent,ValueEvaluator,encode
from .reply_search import reset_game
from .aligned_data import BASE
from .deep_search import TreeSearch
from .node_diagnostic import row,child
from .visited_data import state_dict,state_hash
from .root_policy import inputs,XDIM
from .value_experiment import write
ROOT=Path('artifacts/root-value-correction')

def collect(games=24):
 seed_all(1000);ROOT.mkdir(exist_ok=True,parents=True);begin=time.perf_counter();records=[];sources=[]
 old=BudgetAgent('value',BASE,.5,24,6,'cuda');evaluator=ValueEvaluator(BASE,'cuda',.5,24)
 for g in range(games):
  rng=random.Random(1030000+g);s=State();history=[];moves=[];reset_game(old)
  while not s.done:
   history.append(clone(s));qs=legal_moves(s)
   if not qs:m=None
   elif rng.random()<.15:m=rng.choice(qs).move
   elif s.turn==g%2:m=old.act(s,rng)
   else:m=qs[int(np.argmax(heuristic(features(s,qs)[0])))].move
   play(s,m,validate=False);moves.append(row(m))
  split=0 if g<games*.8 else 1 if g<games*.9 else 2
  sources.append(dict(game=g,seed=1030000+g,split=split,moves=moves,scores=[s.score(0),s.score(1)]))
  for ply in sorted(set(int((len(history)-1)*f) for f in [.45,.65])):
   s=history[ply];qs=legal_moves(s)
   if len(qs)<2:continue
   order=np.argsort(-heuristic(features(s,qs)[0]),kind='stable')[:128];qs=[qs[int(i)] for i in order]
   tree=TreeSearch(evaluator,width=3,tt_limit=0);initial=[]
   for q in qs:initial.append(-tree.search(child(s,q.move),1))
   top=np.argsort(-np.array(initial),kind='stable')[:4].tolist();rest=[i for i in range(len(qs)) if i not in top]
   random_ids=rng.sample(rest,min(2,len(rest)));selected=top+random_ids;final=[None]*len(qs)
   for i in selected:final[i]=-tree.search(child(s,qs[i].move),3)
   records.append(dict(game=g,split=split,ply=ply,state=state_dict(s),state_hash=state_hash(s),moves=[row(q.move) for q in qs],initial=initial,final=final,top=top,random=random_ids,nodes=tree.nodes))
  if (g+1)%12==0:print('games',g+1,'rows',len(records),flush=True)
 # Reject duplicate sampled state across splits before materializing learning data.
 seen={};duplicates=[]
 for r in records:
  if r['state_hash'] in seen:duplicates.append(r['state_hash'])
  else:seen[r['state_hash']]=r
 records=list(seen.values())
 data={k:[] for k in ['board','state','x','mask','target','initial','game','split','random']}
 changes=0;regrets=[]
 for r in records:
  s=State(**r['state']);qs0={tuple(row(q.move)):q for q in legal_moves(s)};qs=[qs0[tuple(m)] for m in r['moves']];x,st=inputs(s,qs);b,st=encode(s);n=len(qs)
  mask=np.zeros(128,bool);mask[:n]=[v is not None for v in r['final']];target=np.full(128,np.nan,np.float32);initial=np.full(128,np.nan,np.float32);initial[:n]=r['initial']
  for i,v in enumerate(r['final']):
   if v is not None:target[i]=v-r['initial'][i]
  pad=np.zeros((128,XDIM),np.float32);pad[:n]=x;rand=np.zeros(128,bool);rand[r['random']]=True
  for k,v in dict(board=b,state=st,x=pad,mask=mask,target=target,initial=initial,game=r['game'],split=r['split'],random=rand).items():data[k].append(v)
  ids=np.flatnonzero(mask);best=max(r['final'][i] for i in ids);base=max(ids,key=lambda i:r['initial'][i]);regret=best-r['final'][base];regrets.append(regret);changes+=regret>4e-6
 np.savez_compressed(ROOT/'data.npz',**{k:np.array(v) for k,v in data.items()})
 for name,obj in [('teacher-labels',records),('source-games',sources)]:
  with gzip.open(ROOT/f'{name}.json.gz','wt') as f:json.dump(obj,f,separators=(',',':'))
 summary=dict(games=games,rows=len(records),duplicates_removed=len(duplicates),changed=changes,changed_fraction=changes/max(1,len(records)),base_regret=float(np.mean(regrets)),oracle_corrected_regret=0.,advance=changes/max(1,len(records))>=.1,seconds=time.perf_counter()-begin,data_sha256=sha(ROOT/'data.npz'),value_sha256=sha(BASE),label='root points: completed depth4-depth2; teacher information, NOT learned/low-budget success',selection_bias='initial top4 + uniform 2 of remaining heuristic top128; not full legal uniform',split_rows={str(i):sum(r['split']==i for r in records) for i in range(3)})
 write(ROOT/'data-summary.json',summary);print(summary,flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--games',type=int,default=24);a=p.parse_args();collect(a.games)
if __name__=='__main__':main()
