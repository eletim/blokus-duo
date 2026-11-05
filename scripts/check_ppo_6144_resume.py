import sys,tempfile,subprocess,shutil,json
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from check_correction_resume import equal
from blokus.ppo_scale import history
from blokus.value_experiment import write
r=Path(tempfile.mkdtemp(prefix='ppo6144-resume-'));base=Path('artifacts/ppo-4096/seed-1401/iteration-4096.pt')
try:
 def run(name,step,resume):
  out=r/name;subprocess.run([sys.executable,'-m','blokus.ppo_scale','--seed','1401','--resume',str(resume),'--output',str(out),'--iterations',str(step),'--cap','120','--forbidden-prefixes','artifacts/ppo-6144/forbidden-prefixes.json'],check=True);return out/f'iteration-{step}.pt'
 full=run('full',4102,base);part=run('part',4098,base);end=run('end',4102,part)
 a=torch.load(full,map_location='cpu',weights_only=False);b=torch.load(end,map_location='cpu',weights_only=False)
 checks={k:equal(a[k],b[k]) for k in ['model','past','optimizer','iteration','python_rng','numpy_rng','torch_rng','cuda_rng']};checks['history']=equal(history(a,full),history(b,end));assert all(checks.values())
 write('artifacts/ppo-6144/cuda-resume.json',dict(pass_=True,separate_process=True,base_iteration=4096,continuous_additional=6,first_additional=2,resumed_additional=4,past_exchange_iteration=4100,checks=checks))
finally:shutil.rmtree(r)
