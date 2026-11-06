import json,time,statistics
from pathlib import Path
import torch
from blokus import ppo_runtime as r
from blokus.ppo_gpu import update_fast
from scripts.bench_ppo_runtime import records_hash
R=Path('artifacts/ppo-ragged');rows=[]
for mb in (16,64):
 for rep in range(-1,4):
  for compact in ([False,True] if rep%2==0 else [True,False]):
   r.seed_all(0);m,p,o,ck=r.restore('artifacts/ppo-four-gpu/seed-1751401/iteration-4096.pt','cuda');r.seed_all(178400);torch.cuda.reset_peak_memory_stats();start=time.perf_counter();rec,games,t=r.collect(m,p,4096,178400,backend='native');u=update_fast(m,o,rec,packed=not compact,minibatch=mb,compact=compact);wall=time.perf_counter()-start
   if rep>=0:rows.append(dict(rep=rep,minibatch=mb,compact=compact,wall=wall,games=16,learner_states=len(rec),input_hash=records_hash(rec),collection=t,update=u,peak_bytes=torch.cuda.max_memory_allocated(),games_per_second=16/wall,learner_states_per_second=len(rec)/wall))
assert len({x['input_hash'] for x in rows})==1
(R/'e2e-benchmark.json').write_text(json.dumps(dict(rows=rows,warmup=1,repeats=4,initialization_excluded=True,same_rollout_all_runs=True,summary={f'{mb}-{c}':dict(wall=statistics.median(x['wall'] for x in rows if x['minibatch']==mb and x['compact']==c),collect=statistics.median(x['collection']['seconds'] for x in rows if x['minibatch']==mb and x['compact']==c),update=statistics.median(x['update']['seconds'] for x in rows if x['minibatch']==mb and x['compact']==c)) for mb in (16,64) for c in (False,True)}),indent=2)+'\n')
print('same-rollout end-to-end PASS')
