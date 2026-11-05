"""Descriptive loss samples, not counterfactual causal attribution or training data."""
import json,random,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from blokus.engine import Move,legal_moves,play,frontier,SIZES
from blokus.research import clone
from blokus.ai import features,heuristic,seed_all
from blokus.tournament import prefix_state,read_prefixes
from blokus.visited_data import state_hash
from blokus.deep_search import TreeSearch
from blokus.value_search import ValueEvaluator
from blokus.aligned_data import BASE
R=Path('artifacts/ppo-generation2')
def metrics(s,seat):
 own=clone(s);own.turn=seat;opp=clone(s);opp.turn=1-seat
 return dict(scores=[s.score(seat),s.score(1-seat)],placed=[s.boards[seat].bit_count(),s.boards[1-seat].bit_count()],frontiers=[frontier(s,p)[0].bit_count() for p in (seat,1-seat)],legal_counts=[len(legal_moves(own)),len(legal_moves(opp))],remaining_piece_sizes=[[SIZES[i] for i in range(21) if s.remaining[p]>>i&1] for p in (seat,1-seat)])
def run():
 assert (R/'timing-complete.json').exists();seed_all(1622000);fixture=read_prefixes(R/'prefixes.json');lookup={p['id']:p for p in fixture['prefixes']};ev=ValueEvaluator(BASE,'cuda',.5,24);games=[]
 for seed in (1401,1402,1403):
  for op in ('previous','teacher100'):
   result=json.loads((R/f'{seed}-{op}.json').read_text());g=next(g for g in result['games'] if g['difference']<0);p=lookup[g['prefix_id']];s=prefix_state(p);stages={};trace=[]
   for i,row in enumerate(g['moves']):
    ply=len(p['moves'])+i;stage='opening' if ply<14 else 'middle' if ply<28 else 'ending';m=Move(*row) if row else None
    if s.turn==g['seat']:
     trace.append(dict(ply=ply,stage=stage,**metrics(s,g['seat'])))
     if stage not in stages and m is not None:
      qs=legal_moves(s);order=np.argsort(-heuristic(features(s,qs)[0]),kind='stable');chosen=next(j for j,q in enumerate(qs) if q.move==m);rank=int(np.flatnonzero(order==chosen)[0])+1;indices=list(map(int,order[:128]));indices+=[] if chosen in indices else [chosen]
      tree=TreeSearch(ev,3);values={}
      for j in indices:
       child=clone(s);play(child,qs[j].move);values[j]=-tree.search(child,1)
      best=max(values,key=values.get)
      stages[stage]=dict(ply=ply,state_hash=state_hash(s),actual_move=row,heuristic_rank=rank,legal_count=len(qs),evaluated_roots=len(indices),actual_two_ply_value=values[chosen],best_two_ply_value=values[best],reference_gap=values[best]-values[chosen],reference_choice=[qs[best].move.piece,qs[best].move.orientation,qs[best].move.row,qs[best].move.col],**metrics(s,g['seat']))
    play(s,m)
   assert s.done and [s.score(0),s.score(1)]==g['scores']
   games.append(dict(seed=seed,opponent=op,prefix_id=g['prefix_id'],seat=g['seat'],scores=g['scores'],difference=g['difference'],sample_rule='first losing game in fixed evaluation order',stages=stages,trace=trace,final=metrics(s,g['seat'])))
 summary={}
 for stage in ('opening','middle','ending'):
  rows=[g['stages'][stage] for g in games if stage in g['stages']]
  summary[stage]=dict(n=len(rows),mean_reference_gap=float(np.mean([r['reference_gap'] for r in rows])),mean_heuristic_rank=float(np.mean([r['heuristic_rank'] for r in rows])),positive_gap=sum(r['reference_gap']>1e-5 for r in rows))
 (R/'loss-analysis.json').write_text(json.dumps(dict(games=games,sample_unique_prefixes=len({g["prefix_id"] for g in games}),phase_summary=summary,reference='Fixed value alpha .5 gate24, heuristic top128 roots plus actual move, top3 opponent replies, complete depth2. Deterministic node work, not timed strength comparison, not true game value.',limitations='Six deliberately selected losses; no matched wins or counterfactual game continuations. Reference disagreement is not proof of error or causality. No training/model changes.'),indent=2)+'\n');print(summary)
if __name__=='__main__':run()
