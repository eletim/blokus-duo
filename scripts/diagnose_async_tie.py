"""Post-timing diagnosis of the one large completed-root discrepancy; no tuning."""
import gzip,json,random,hashlib
from pathlib import Path
import numpy as np
import torch
import frontier_generation as f
from blokus.engine import Move,play
from blokus.ppo_policy import PPOAgent
from blokus.ppo_frontier import FrontierSearch,FrontierEvaluator
from blokus.generation_graph import GraphEvaluator
R=Path('artifacts/gpu-async512');d=json.load(gzip.open(R/'final/cpu-1/1751401-845.json.gz','rt'));task=d['task'];s=f.ref.e.state(task['prefix']['moves']);rng=random.Random();rng.setstate(f.ref.tuples(task['rng_state']));torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
cpu=PPOAgent(task['seed'],'cpu',backend='native');gpu=PPOAgent(task['seed'],'cuda',backend='native')
for raw in d['moves'][len(task['prefix']['moves']):16]:cpu.act(s,rng);play(s,Move(*raw) if raw else None,validate=True)
initial_rng=rng.getstate();out={}
for mode in ('cpu','gpu','gpu_values_cpu_ranking'):
 ce=FrontierEvaluator(cpu);ge=GraphEvaluator(gpu);search=FrontierSearch(cpu,3600000);r=random.Random();r.setstate(initial_rng);gen=search.act(s,r);changes=[];maxlogit=0.;maxvalue=0.;calls=0
 try:
  req=next(gen)
  while True:
   a=ce.batch([req])[0];b=ge.batch([req])[0];calls+=1;kind,obs=req
   if kind=='policy':
    maxlogit=max(maxlogit,float(np.max(np.abs(a[0]-b[0]))));maxvalue=max(maxvalue,abs(a[1]-b[1]));ca=torch.as_tensor(a[0]).argsort(descending=True).tolist();cb=torch.as_tensor(b[0]).argsort(descending=True).tolist()
    if ca[:4]!=cb[:4]:
     changes.append(dict(request=calls,input_sha=hashlib.sha256(obs[0].tobytes()+obs[1].tobytes()+obs[2].tobytes()+obs[3].tobytes()).hexdigest(),same_set=set(ca[:4])==set(cb[:4]),cpu_indices=ca[:6],gpu_indices=cb[:6],cpu_scores=[float(a[0][i]) for i in ca[:6]],gpu_scores=[float(b[0][i]) for i in cb[:6]],cpu_cells=[obs[3][i].tolist() for i in ca[:4]],gpu_cells=[obs[3][i].tolist() for i in cb[:4]]))
   else:maxvalue=max(maxvalue,abs(a-b))
   answer=a if mode=='cpu' else b
   if mode=='gpu_values_cpu_ranking' and kind=='policy':answer=(a[0],b[1])
   req=gen.send(answer)
 except StopIteration as end:move=end.value
 out[mode]=dict(move=f.ref.e.raw(move),root_scores=search.stats['root_scores'],nodes=search.stats['nodes'],calls=calls,max_logit_error=maxlogit,max_value_error=maxvalue,top4_changes=changes)
 assert maxlogit<=2e-5 and maxvalue<=2e-5
assert out['cpu']['move']==d['moves'][16] and out['cpu']['root_scores']==d['stats'][16-len(task['prefix']['moves'])]['root_scores']
assert out['gpu_values_cpu_ranking']['move']==out['cpu']['move']
assert max(abs(a-b) for a,b in zip(out['gpu_values_cpu_ranking']['root_scores'],out['cpu']['root_scores']))<=2e-5
f.ref.write(R/'tie-diagnosis.json',dict(game='1751401-845',ply=16,probe='paired CPU vs CUDA graph singleton, fixed8-row bucket; not replay of arbitrary original async batch shapes',results=out))
