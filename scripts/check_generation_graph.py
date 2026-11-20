import json,random
import numpy as np
import torch
from blokus.ppo_policy import PPOAgent
from blokus.ppo_frontier import FrontierEvaluator
from blokus.generation_graph import GraphEvaluator
from blokus.generation_encoding import encode
from blokus.engine import State,play
import frontier_generation as f
torch.set_num_threads(1);torch.use_deterministic_algorithms(True);rows=[]
for seed in f.ref.e.SEEDS:
 cpu=PPOAgent(seed,'cpu',backend='native');gpu=PPOAgent(seed,'cuda',backend='native');s=State();r=random.Random(98);requests=[]
 for ply in range(64):
  if s.done:break
  obs,_=cpu._observation(s);requests.append(('policy',obs));requests.append(('value',encode(s)))
  play(s,cpu.act(s,r),validate=True)
 ev=GraphEvaluator(gpu);ref=FrontierEvaluator(cpu);maxlogit=0.;maxvalue=0.;agree=0;total=0
 for n in (1,8,17,64,128,256):
  req=(requests*8)[:n];a=ref.batch(req);b=ev.batch(req)
  for (kind,_),x,y in zip(req,a,b):
   if kind=='policy':
    assert np.isfinite(y[0]).all() and np.isfinite(y[1])
    maxlogit=max(maxlogit,float(np.max(np.abs(x[0]-y[0]))));maxvalue=max(maxvalue,abs(x[1]-y[1]));agree+=int(np.argmax(x[0])==np.argmax(y[0]));total+=1
   else:
    assert np.isfinite(y)
    maxvalue=max(maxvalue,abs(x-y))
 assert maxlogit<=2e-5 and maxvalue<=2e-5,(maxlogit,maxvalue)
 rows.append(dict(seed=seed,max_logit=maxlogit,max_value=maxvalue,argmax_agree=agree,total=total,graphs=len(ev.graphs),probe_requests=[1,8,17,64,128,256],note='repeated saved inputs test actual policy/value batch sizes up to128; counts are not unique states',capture_seconds=ev.capture_seconds))
f.ref.write('artifacts/gpu-async512/graph-check.json',rows)
