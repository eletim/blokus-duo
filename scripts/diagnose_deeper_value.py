"""Fixed-continuation WDL diagnostic, not minimax labels, no fitting."""
import sys,json,copy,random,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from blokus.engine import State,play,legal_moves
from blokus.ppo_policy import PPOAgent,collate
from blokus.ppo_search_staged import ValueLeafSearch
from blokus.ppo_search import SearchAgent
from blokus.ppo_runtime import state_hash
R=Path('artifacts/gen3-deeper');torch.set_num_threads(1);p=PPOAgent(1751401,'cpu');records=[];bank=[];start=time.perf_counter()
for game in range(6):
 rng=random.Random(91829000+game);s=State();moves=[]
 for ply in range([12,20,28][game%3]):
  qs=legal_moves(s);m=rng.choice(qs).move if qs and rng.random()<.25 else p.act(s,rng)
  play(s,m,validate=False);moves.append(None if m is None else list(m.__dict__.values()))
 bank.append(dict(id=game,moves=moves,state_hash=state_hash(s)))
 obs,qs=p._observation(s)
 with torch.inference_mode():l,v=p.model(*collate([obs],'cpu'))
 order=l[0].argsort(descending=True).tolist();ordered=[qs[i].move for i in order];base=ordered[:4];greedy=p.act(s,random.Random(9))
 if greedy not in base:base[-1]=greedy
 roots=(base+[m for m in ordered if m not in base])[:8]
 a=ValueLeafSearch(p,5000,root_stages=(8,),reuse=True);a.act(s,random.Random(9));assert a.stats['depth']==4 and a.stats['root_width']==len(roots)
 estimates=a.stats['root_scores']
 for i,m in enumerate(roots):
  outcomes={}
  for kind in ('policy','fixed_E2'):
   results=[]
   for repeat in range(2):
    z=copy.deepcopy(s);play(z,m,validate=False);trace=[];r=random.Random(91839000+game*2+repeat);bot=p if kind=='policy' else SearchAgent(p,5000,width=4,max_depth=2)
    while not z.done:
     q=bot.act(z,r)
     if kind!='policy' and q is not None:assert bot.stats['depth']==2
     play(z,q,validate=False);trace.append(None if q is None else list(q.__dict__.values()))
    d=z.score(s.turn)-z.score(1-s.turn);results.append(dict(wdl=int(np.sign(d)),difference=d,moves=trace,scores=[z.score(0),z.score(1)]))
   outcomes[kind]=results
  records.append(dict(game=game,ply=len(moves),candidate=i,move=list(m.__dict__.values()),estimate=estimates[i],outcomes=outcomes))
 print('diagnostic game',game,flush=True)
summary={}
for kind in ('policy','fixed_E2'):
 grouped=[]
 for game in range(6):
  rs=[r for r in records if r['game']==game];pred=np.array([r['estimate'] for r in rs]);wdl=np.array([np.mean([x['wdl'] for x in r['outcomes'][kind]]) for r in rs]);old=np.argmax(pred[:4]);wide=np.argmax(pred)
  grouped.append(dict(game=game,mse=float(np.mean((pred-wdl)**2)),sign_accuracy=float(np.mean(np.sign(pred)==np.sign(wdl))),estimated_gain=float(pred[wide]-pred[old]),continuation_gain=float(wdl[wide]-wdl[old]),old_index=int(old),wide_index=int(wide),estimate_range=[float(pred.min()),float(pred.max())]))
 summary[kind]=grouped
(R/'diagnostic-bank.json').write_text(json.dumps(dict(prefixes=bank),indent=2));(R/'diagnostic.json').write_text(json.dumps(dict(records=records,summary=summary,wall_seconds=time.perf_counter()-start,independent_source_games=6,seed=1751401,continuation_repetitions=2,limit='same-game roots and repeats correlated; fixed-policy outcomes not minimax'),indent=2))
