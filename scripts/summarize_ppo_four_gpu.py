"""Final tables from saved data only; no model selection or extra games."""
import json,statistics,hashlib
from pathlib import Path
R=Path('artifacts/ppo-four-gpu')
def load(name):return json.loads((R/name).read_text())
bench=load('update-benchmark.json');pilot=load('pilot-comparison.json');rows=[]
for mode in ('baseline','fast'):
 logs=pilot['results'][mode]['logs'][1:];wall=[x['generation']['seconds']+x['update']['seconds'] for x in logs]
 rows.append(dict(mode=mode,iterations=len(logs),games_per_iteration=16,median_iteration_wall=statistics.median(wall),games_per_second=16/statistics.median(wall),learner_states_per_second=sum(x['update']['records'] for x in logs)/sum(wall),optimizer_steps_per_second=sum(x['update']['updates'] for x in logs)/sum(wall),total_optimizer_steps=sum(x['update']['updates'] for x in logs),median_collection=statistics.median(x['generation']['seconds'] for x in logs),median_update=statistics.median(x['update']['seconds'] for x in logs)))
curves=[]
for folder in sorted(R.glob('seed-*')):
 cumulative_wall=0.
 for p in sorted(folder.glob('result-*.json'),key=lambda p:int(p.stem.split('-')[-1])):
  d=json.loads(p.read_text());logs=d['logs'];cumulative_wall+=d['wall_with_save'];curves.append(dict(seed=d['seed'],iteration=d['iteration'],games=d['iteration']*16,chunk_wall=d['wall_with_save'],cumulative_wall=cumulative_wall,entropy=statistics.mean(x['update']['entropy'] for x in logs),kl=statistics.mean(x['update']['approx_kl'] for x in logs),value_mse=statistics.mean(x['update']['value_mse'] for x in logs),optimizer_steps=sum(x['update']['updates'] for x in logs),learner_states=sum(x['update']['records'] for x in logs),collection_cpu=sum(x['generation']['cpu_generation_pack'] for x in logs),collection_inference_sync=sum(x['generation']['gpu_inference_sync'] for x in logs),update_wall=sum(x['update']['seconds'] for x in logs),history_write=sum(x['history_write_seconds'] for x in logs),opponent_wdl={k:statistics.mean(x['opponent_wdl'][k] for x in logs) for k in ('current','past','heuristic')}))
(R/'summary.json').write_text(json.dumps(dict(update=bench['summary'],end_to_end=rows,training_curve=curves),indent=2))
