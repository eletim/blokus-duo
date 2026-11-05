"""Separate CUDA processes, past-policy exchange at iteration 516, external history."""
import sys,tempfile,subprocess,shutil
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from check_correction_resume import equal
from blokus.ppo_scale import history
from blokus.value_experiment import write
r=Path(tempfile.mkdtemp(prefix='ppo-scale-resume-'));base=Path('artifacts/terminal-ppo/extended/seed-1401/iteration-512.pt')
try:
 def run(name,step,resume):
  out=r/name;subprocess.run([sys.executable,'-m','blokus.ppo_scale','--seed','1401','--resume',str(resume),'--output',str(out),'--iterations',str(step),'--cap','120'],check=True);return out/f'iteration-{step}.pt'
 full=run('full',518,base);part=run('part',514,base);end=run('end',518,part)
 a=torch.load(full,map_location='cpu',weights_only=False);b=torch.load(end,map_location='cpu',weights_only=False)
 checks={k:equal(a[k],b[k]) for k in ['model','past','optimizer','iteration','python_rng','numpy_rng','torch_rng','cuda_rng']};checks['history']=equal(history(a,full),history(b,end));assert all(checks.values())
 write('artifacts/ppo-learning-curve/cuda-resume.json',dict(pass_=True,separate_process=True,base_iteration=512,continuous_additional=6,first_additional=2,resumed_additional=4,past_exchange_iteration=516,checks=checks))
finally:shutil.rmtree(r)
