"""Mixed-stage CPU/CUDA input, logits, ranking and value batch comparison."""
import json,gzip
import numpy as np,torch
import frontier_generation as f
from blokus.ppo_policy import PPOAgent,collate
from blokus.ppo_frontier import FrontierEvaluator
from blokus.ppo_value_leaf import value_inputs,value_only
from blokus.engine import State,Move,play
R=f.R;torch.set_num_threads(1);torch.use_deterministic_algorithms(True);rows=[]
for seed in f.ref.e.SEEDS:
 cpu=PPOAgent(seed,'cpu',backend='native');gpu=PPOAgent(seed,'cuda',backend='native');req=[];reference=[]
 for task in json.loads((R/'bank.json').read_text())['games']:
  if task['seed']!=seed or task['id']>=4:continue
  d=json.load(gzip.open(R/'cpu-1'/f"{seed}-{task['id']}.json.gz",'rt'));s=State()
  for j,m in enumerate(d['moves']):
   if j in (0,8,16,24,32,40):
    o,_=cpu._observation(s)
    with torch.inference_mode():l,v=cpu.model(*collate([o],'cpu'))
    req.append(('policy',o));reference.append((l[0].numpy(),v.item()))
    req.append(('value',value_inputs(s,cpu.channels)));reference.append(value_only(cpu,s))
   play(s,Move(*m) if m else None)
 got=FrontierEvaluator(gpu).batch(req)
 for (kind,_),a,b in zip(req,reference,got):
  if kind=='value':rows.append(dict(seed=seed,kind=kind,value_error=abs(a-b)))
  else:
   err=float(np.abs(a[0]-b[0]).max());order=lambda x:torch.as_tensor(x).argsort(descending=True).tolist()
   rows.append(dict(seed=seed,kind=kind,value_error=abs(a[1]-b[1]),logit_error=err,top1=order(a[0])[:1]==order(b[0])[:1],top4=order(a[0])[:4]==order(b[0])[:4]))
assert max(x['value_error'] for x in rows)<=2e-5 and max(x.get('logit_error',0) for x in rows)<=2e-5
p=[x for x in rows if x['kind']=='policy'];assert sum(x['top1'] for x in p)/len(p)>=.99
f.ref.write(R/'batch-numerics.json',dict(rows=rows,max_value_error=max(x['value_error'] for x in rows),max_logit_error=max(x.get('logit_error',0) for x in rows),top1_agreement=sum(x['top1'] for x in p)/len(p),top4_agreement=sum(x['top4'] for x in p)/len(p)))
