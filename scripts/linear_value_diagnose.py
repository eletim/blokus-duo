"""Outcome calibration on visited states and the actual two-ply candidate leaves."""
import copy,gzip,json,hashlib,random,time
from pathlib import Path
import numpy as np
from blokus.engine import State,play,legal_moves
from blokus.linear_rl import prefix
from blokus.linear_value import Agent,Search,outcome,value_features
from blokus.linear_train2 import state_hash
R=Path('artifacts/linear-value-retrain');OLD=Path('artifacts/linear-hybrid/exports');SEEDS=(3101,3102,3103)
def packed(s):return dict(boards=s.boards,remaining=s.remaining,last=s.last,turn=s.turn,passes=s.passes)
def unpack(d):return State(**d)
def move_row(m):return None if m is None else [m.piece,m.orientation,m.row,m.col]
def ordered(a,s,width):
 x,qs,p=a.inputs(s)
 if not qs:return [None]
 return [qs[i].move for i in np.argsort(-(x@a.weights[p]),kind='stable')[:width]]
def finish(s,a,seed):
 t=copy.deepcopy(s);player=t.turn;rng=random.Random(seed);moves=[]
 while not t.done:
  m=a.act(t,rng);moves.append(move_row(m));play(t,m,validate=False)
 return dict(moves=moves,scores=[t.score(0),t.score(1)],reward=outcome(t,player),start_player=player)
def metrics(rows):
 if not rows:return {}
 p=np.array([r['prediction'] for r in rows]);y=np.array([r['reward'] for r in rows]);return dict(n=len(rows),mse=float(np.mean((p-y)**2)),zero_mse=float(np.mean(y*y)),sign_accuracy=float(np.mean(np.sign(p)==np.sign(y))),prediction_range=[float(p.min()),float(p.max())],mean_prediction=float(p.mean()),mean_reward=float(y.mean()),by_phase={str(k):metrics_plain([r for r in rows if r['phase']==k]) for k in range(3)})
def metrics_plain(rows):
 if not rows:return dict(n=0)
 p=np.array([r['prediction'] for r in rows]);y=np.array([r['reward'] for r in rows]);return dict(n=len(rows),mse=float(np.mean((p-y)**2)),sign_accuracy=float(np.mean(np.sign(p)==np.sign(y))),mean_prediction=float(p.mean()),mean_reward=float(y.mean()),prediction_range=[float(p.min()),float(p.max())])
def main():
 import torch
 torch.set_num_threads(1)
 from blokus.ppo_policy import PPOAgent
 R.mkdir(exist_ok=True,parents=True);start=time.perf_counter();records=[];roots=[];visit_rows=[];leaf_rows=[]
 oldindex=np.load('/tmp/linear-value-training.npy',mmap_mode='r')
 from scripts.linear_value_eval import contains
 for seed in SEEDS:
  a=Agent.load(OLD/f'seed-{seed}.json');g3=PPOAgent(model_seed=1751401+seed-3101,device='cpu',backend='native')
  for distribution in ('self','gen3'):
   for game in range(12):
    ps=2270000000+seed*1000+(0 if distribution=='self' else 100)+game;s=prefix(ps);assert not contains(oldindex,state_hash(s));seat=game%2;rng=random.Random(ps);moves=[];states=[];local=[];picked=[]
    while not s.done:
     if s.turn==seat:
      phi=value_features(s);phase=next((k for k in range(3) if np.any(phi[k])),0);local.append(dict(prediction=a.value(s),phase=phase))
      if len(picked)<2 and len(moves)+4>=12+len(picked)*12:picked.append(copy.deepcopy(s))
     actor=g3 if distribution=='gen3' and s.turn!=seat else a;m=actor.act(s,rng);moves.append(move_row(m));play(s,m,validate=False)
    y=outcome(s,seat)
    for r in local:r.update(reward=y,distribution=distribution,seed=seed)
    visit_rows.extend(local);records.append(dict(seed=seed,distribution=distribution,prefix_seed=ps,seat=seat,moves=moves,scores=[s.score(0),s.score(1)],reward=y,predictions=local))
    for j,root in enumerate(picked):
     candidates=ordered(a,root,13);ranks=list(range(min(6,len(candidates))))+[k for k in (6,12) if k<len(candidates)];values=[]
     for rank in ranks:
      t=copy.deepcopy(root);play(t,candidates[rank],validate=False);leaves=[]
      if t.done:leafstates=[t]
      else:
       leafstates=[]
       for reply in ordered(a,t,3):u=copy.deepcopy(t);play(u,reply,validate=False);leafstates.append(u)
      predicted=[];targets=[]
      for n,leaf in enumerate(leafstates):
       rollout=[finish(leaf,a,2280000000+ps*10+j*2+rep) for rep in range(2)];pred=a.value(leaf);sign=1 if leaf.turn==root.turn else -1
       phi=value_features(leaf) if not leaf.done else np.zeros((3,8));phase=next((k for k in range(3) if np.any(phi[k])),0)
       for rr in rollout:leaf_rows.append(dict(seed=seed,distribution=distribution,prediction=pred,reward=rr['reward'],phase=phase,within=rank<6))
       predicted.append(sign*pred);targets.append(sign*np.mean([rr['reward'] for rr in rollout]));leaves.append(dict(state=packed(leaf),prediction=pred,rollouts=rollout))
      values.append(dict(rank=rank,move=move_row(candidates[rank]),predicted=min(predicted),target=min(targets),leaves=leaves))
     top=[v for v in values if v['rank']<6];choice=max(top,key=lambda x:x['predicted']);best=max(top,key=lambda x:x['target']);outside=[v for v in values if v['rank']>=6]
     roots.append(dict(seed=seed,distribution=distribution,state=packed(root),candidates=values,selection_loss=best['target']-choice['target'],outside_better=bool(outside and max(x['target'] for x in outside)>best['target']),chosen_rank=choice['rank']))
  print('DIAG',seed,flush=True)
 with gzip.open(R/'diagnostic.json.gz','wt') as f:json.dump(dict(games=records,roots=roots),f)
 summary={d:dict(visited=metrics([r for r in visit_rows if r['distribution']==d]),leaves=metrics([r for r in leaf_rows if r['distribution']==d and r['within']]),mean_selection_loss=float(np.mean([r['selection_loss'] for r in roots if r['distribution']==d])),outside_better_fraction=float(np.mean([r['outside_better'] for r in roots if r['distribution']==d]))) for d in ('self','gen3')}
 summary['add_S']=summary['self']['leaves']['mse']>=1.10*summary['self']['visited']['mse'];summary['seconds']=time.perf_counter()-start;summary['games']=len(records);summary['roots']=len(roots);(R/'diagnostic-summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(summary,flush=True)
if __name__=='__main__':main()
