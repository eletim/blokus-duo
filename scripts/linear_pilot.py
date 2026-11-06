"""Fixed exploratory pilot. No checkpoint selection or adaptive stopping."""
import argparse,copy,json,random,time
from pathlib import Path
import numpy as np
from blokus.engine import State,play
from blokus.linear_rl import Trainer,LinearAgent,prefix,state_key

parser=argparse.ArgumentParser();parser.add_argument('--output',default='artifacts/linear-rl-pilot');args=parser.parse_args()
ROOT=Path(args.output)
if ROOT.exists() and any(ROOT.iterdir()):parser.error('output must be empty to preserve prior experiments')
ROOT.mkdir(parents=True,exist_ok=True)
plan=dict(seeds=[3101,3102,3103],iterations=256,games_per_iteration=16,lr=.03,initial='fixed handcrafted weights; not random actions',algorithm='terminal WDL REINFORCE; phase-specific 30 weights; Adam ascent; gradient norm cap5',opponents='50% frozen batch-start learner,25% past64iteration,25% greedy initial, balanced seats',evaluation='final only;64 fresh 4-ply prefixes x2 seats x3 models, against initial and gen3; exploratory, no model promotion',eval_prefix_seed=1800000000,backend='native',search=False)
(ROOT/'PLAN.json').write_text(json.dumps(plan,indent=2))
visited=set();models=[];train=[]
for seed in plan['seeds']:
 t=Trainer(seed,'native');start=time.perf_counter();logs=[]
 for it in range(plan['iterations']):
  logs.append(t.step(visited=visited))
  if (it+1)%64==0:print('train',seed,it+1,round(time.perf_counter()-start,2),flush=True)
 t.save(ROOT/f'state-{seed}.json');t.agent.save(ROOT/f'policy-{seed}.json',dict(seed=seed,iterations=t.iteration));models.append(t.agent)
 train.append(dict(seed=seed,seconds=time.perf_counter()-start,logs=logs))
(ROOT/'train.json').write_text(json.dumps(train))
print('training complete',len(visited),'states',flush=True)
import torch
from blokus.ppo_policy import PPOAgent
torch.set_num_threads(1)
base=PPOAgent(device='cpu',backend='native');initial=LinearAgent(backend='native')
prefixes=[];index=0
while len(prefixes)<64:
 seed=plan['eval_prefix_seed']+index;index+=1;s=prefix(seed)
 if state_key(s) not in visited:prefixes.append((seed,s))
results=[];overlap=0
for opponent,bot in [('initial',initial),('gen3',base)]:
 for i,learner in enumerate(models):
  for j,(seed,s) in enumerate(prefixes):
   for seat in [0,1]:
    state=copy.deepcopy(s);rng=random.Random(1900000000+j*2+seat);moves=[]
    while not state.done:
     overlap+=int(state_key(state) in visited)
     m=(learner if state.turn==seat else bot).act(state,rng)
     play(state,m,validate=True);moves.append(None if m is None else [m.piece,m.orientation,m.row,m.col])
    d=state.score(seat)-state.score(1-seat)
    results.append(dict(opponent=opponent,model_seed=plan['seeds'][i],prefix_seed=seed,seat=seat,difference=d,point=1 if d>0 else .5 if d==0 else 0,moves=moves))
  print('eval',opponent,i,flush=True)
(ROOT/'evaluation.json').write_text(json.dumps(results))
summary={}
for opponent in ['initial','gen3']:
 groups=[ [r for r in results if r['opponent']==opponent and r['prefix_seed']==seed] for seed,_ in prefixes]
 points=np.array([np.mean([r['point'] for r in g]) for g in groups]);diff=np.array([np.mean([r['difference'] for r in g]) for g in groups])
 ix=np.random.default_rng(12345).integers(0,64,size=(10000,64))
 summary[opponent]=dict(games=sum(map(len,groups)),point=float(points.mean()),point_ci95=np.quantile(points[ix].mean(1),[.025,.975]).tolist(),difference=float(diff.mean()),difference_ci95=np.quantile(diff[ix].mean(1),[.025,.975]).tolist())
summary['training_seconds']=sum(x['seconds'] for x in train);summary['train_unique_states']=len(visited);summary['eval_visited_overlap']=overlap;summary['ci_note']='prefix-cluster bootstrap over fixed3trainingseeds; exploratory, not external strength or population-wide seed CI'
# Fixed representative states, warm both implementations; interleave for timing.
bench=[]
for _,s in prefixes[:16]:
 for agent in [initial,models[0],base]:
  agent.act(s,random.Random(1))
for repeat in range(5):
 for _,s in prefixes[:16]:
  for name,agent in [('linear',models[0]),('gen3',base)]:
   start=time.perf_counter();agent.act(s,random.Random(1));bench.append(dict(name=name,ms=(time.perf_counter()-start)*1000))
summary['inference_cpu_ms']={name:float(np.median([r['ms'] for r in bench if r['name']==name])) for name in ['linear','gen3']}
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
