import json,time
from blokus.ppo_minibatch import R,SEEDS,run,write
start=time.perf_counter();total=0.;results=[];previous=4096
for end in (4608,5120):
 if total and total+(total/((previous-4096)*6))*(end-previous)*6>3500:break
 for seed in SEEDS:
  for mb in (16,64):
   folder=R/f'batch-{mb}-seed-{seed}';base=f'artifacts/ppo-four-gpu/seed-{seed}/iteration-4096.pt' if previous==4096 else str(folder/f'iteration-{previous}.pt')
   result=run(folder,base,end,mb,R/'excluded-starts.json',cap=3600-total);total+=result['wall'];results.append({k:v for k,v in result.items() if k!='logs'});write(R/'training-progress.json',dict(seconds=total,results=results));assert result['complete']
 previous=end
write(R/'training-complete.json',dict(common_iteration=previous,complete=previous==5120,seconds=total,wall=time.perf_counter()-start,results=results));print('COMPLETE',previous,total,flush=True)
