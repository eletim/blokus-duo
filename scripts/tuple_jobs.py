"""Finite three-process independent-game collection, fixed endpoint, no agents."""
import concurrent.futures,json,os,subprocess,sys,time
from pathlib import Path
R=Path('artifacts/linear-tuple');start=time.perf_counter();plan=json.loads((R/'fixed-plan.json').read_text());deadline=start+5400

def run(seed):
 for mode,extra in [('generate',['--kind','train','--games',str(plan['games_per_seed'])]),('train',[]),('generate',['--kind','calibration','--games','32'])]:
  tag=extra[1] if extra else 'fit';cmd=[sys.executable,'-m','scripts.tuple_cycle',mode,'--seed',str(seed)]+extra
  with (R/f'job-{seed}-{tag}.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=max(.1,deadline-time.perf_counter()),env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
 return seed
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(run,plan['seeds']))
cpu=0
for seed in rows:
 for p in [R/f'train-{seed}/summary.json',R/f'calibration-{seed}/summary.json',R/f'train-summary-{seed}.json']:cpu+=json.loads(p.read_text())['cpu']
(R/'jobs-summary.json').write_text(json.dumps(dict(complete=True,seeds=rows,wall=time.perf_counter()-start,cumulative_cpu=cpu,workers=3,timeout_seconds=5400),indent=2)+'\n')
print('JOBS COMPLETE',rows,time.perf_counter()-start,flush=True)
