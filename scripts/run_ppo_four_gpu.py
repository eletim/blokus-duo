"""Finite equal-experience three-seed schedule. Never extends after validation."""
import json,time
from pathlib import Path
from blokus.ppo_gpu_train import run
R=Path('artifacts/ppo-four-gpu');seeds=[1751401,1751402,1751403];start=time.perf_counter();total=0.;results=[];previous=0
for end in [512,1024,2048,3072,4096]:
 if previous and total+(total/(previous*3))*(end-previous)*3>5300:
  print('budget stops at common',previous,flush=True);break
 for seed in seeds:
  p=R/f'seed-{seed}';res=run(p,seed,end,resume=str(p/f'iteration-{previous}.pt') if previous else None,cap=5400-total,exclusions=str(R/'excluded-starts.json'));total+=res['wall_with_save'];results.append({k:v for k,v in res.items() if k!='logs'});(R/'training-progress.json').write_text(json.dumps(dict(total_wall=total,results=results),indent=2));assert res['complete'],'incomplete chunk: report common checkpoint, do not extend'
 previous=end
(R/'training-complete.json').write_text(json.dumps(dict(complete=previous==4096,common_iteration=previous,seconds=total,wall=time.perf_counter()-start,seeds=seeds,games_per_seed=previous*16,results=results),indent=2));print('TRAINING FINISHED',previous,total,flush=True)
