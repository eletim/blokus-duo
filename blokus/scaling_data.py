"""Nested source games, balanced opponent/seat strata and fixed evaluation bank."""
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
ROOT=Path('artifacts/correction-game-scaling')

def strata(index):return dict(pool=index%3,seat=(index//3)%2,fold=(index//6)%3)
def load(name):
 p=ROOT/f'{name}.json.gz'
 return json.load(gzip.open(p,'rt')) if p.exists() else []
def dump(name,obj):
 with gzip.open(ROOT/f'{name}.json.gz.tmp','wt') as f:json.dump(obj,f,separators=(',',':'))
 (ROOT/f'{name}.json.gz.tmp').replace(ROOT/f'{name}.json.gz')

def collect(size):
 assert size in [240,960,2400]
 if size==2400:
  allowed=json.loads((ROOT/'extension-gate.json').read_text())['extend']
  exception=ROOT/'extension-exception.json'
  if exception.exists():allowed=allowed or json.loads(exception.read_text()).get('allow_extension',False)
  assert allowed,'registered extension gate failed; explicit documented generation exception required'
 pool_games=size*3//2;seed_all(1100);ROOT.mkdir(parents=True,exist_ok=True);begin=time.perf_counter()
 sources=load('source-games');records=load('teacher-labels');done={g['game'] for g in sources};seen={r['state_hash'] for r in records};duplicates=0
 old=BudgetAgent('value',BASE,.5,24,6,'cuda');evaluator=ValueEvaluator(BASE,'cuda',.5,24)
 for bank,count in [(True,360),(False,pool_games)]:
  for index in range(count):
   gid=index+(100000 if bank else 0)
   if gid in done:continue
   st=strata(index);rng=random.Random(1130000+gid);s=State();history=[];moves=[];reset_game(old)
   while not s.done:
    history.append(clone(s));qs=legal_moves(s)
    if not qs:m=None
    elif rng.random()<.15:m=rng.choice(qs).move
    elif s.turn==st['seat'] or st['pool']==2:m=old.act(s,rng)
    elif st['pool']==0:m=rng.choice(qs).move
    else:m=qs[int(np.argmax(heuristic(features(s,qs)[0])))].move
    play(s,m,validate=False);moves.append(row(m))
   sources.append(dict(game=gid,bank=bank,index=index,seed=1130000+gid,**st,moves=moves,scores=[s.score(0),s.score(1)]))
   for phase in [.45,.65]:
    ply=int((len(history)-1)*phase);s=history[ply];qs=legal_moves(s);h=state_hash(s)
    if len(qs)<2:continue
    if h in seen:duplicates+=1;continue
    seen.add(h);order=np.argsort(-heuristic(features(s,qs)[0]),kind='stable')[:128];qs=[qs[int(i)] for i in order];tree=TreeSearch(evaluator,width=3,tt_limit=0)
    initial=[-tree.search(child(s,q.move),1) for q in qs];top=np.argsort(-np.array(initial),kind='stable')[:4].tolist();rest=[i for i in range(len(qs)) if i not in top];rand=rng.sample(rest,min(2,len(rest)));final=[None]*len(qs)
    for i in top+rand:final[i]=-tree.search(child(s,qs[i].move),3)
    records.append(dict(game=gid,bank=bank,index=index,**st,phase=phase,ply=ply,state=state_dict(s),state_hash=h,moves=[row(q.move) for q in qs],initial=initial,final=final,top=top,random=rand,nodes=tree.nodes))
   if len(sources)%60==0:dump('source-games',sources);dump('teacher-labels',records);print('games',len(sources),'roots',len(records),flush=True)
 dump('source-games',sources);dump('teacher-labels',records)
 selected=[r for r in records if r['bank'] or r['index']<pool_games];data={k:[] for k in ['board','state','x','mask','target','initial','game','fold','bank','random']}
 for r in selected:
  s=State(**r['state']);by={tuple(row(q.move)):q for q in legal_moves(s)};qs=[by[tuple(m)] for m in r['moves']];x,_=inputs(s,qs);b,st=encode(s);n=len(qs)
  mask=np.zeros(128,bool);mask[:n]=[v is not None for v in r['final']];target=np.full(128,np.nan,np.float32);initial=np.full(128,np.nan,np.float32);initial[:n]=r['initial'];pad=np.zeros((128,XDIM),np.float32);pad[:n]=x;rand=np.zeros(128,bool);rand[r['random']]=True
  for i,v in enumerate(r['final']):
   if v is not None:target[i]=v-r['initial'][i]
  for k,v in dict(board=b,state=st,x=pad,mask=mask,target=target,initial=initial,game=r['game'],fold=r['fold'],bank=r['bank'],random=rand).items():data[k].append(v)
 np.savez_compressed(ROOT/f'data-{size}.npz',**{k:np.array(v) for k,v in data.items()})
 summary=dict(size=size,pool_games=pool_games,bank_games=360,unique_generated_games=pool_games+360,rows=len(selected),seconds=time.perf_counter()-begin,duplicates_removed_this_extension=duplicates,data_sha256=sha(ROOT/f'data-{size}.npz'),value_sha256=sha(BASE),folds={str(f):dict(train_games=len({r['game'] for r in selected if not r['bank'] and r['fold']!=f}),test_games=len({r['game'] for r in selected if r['bank'] and r['fold']==f}),train_rows=sum(not r['bank'] and r['fold']!=f for r in selected),test_rows=sum(r['bank'] and r['fold']==f for r in selected)) for f in range(3)})
 write(ROOT/f'data-summary-{size}.json',summary);print(summary,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--size',type=int,required=True);collect(p.parse_args().size)
