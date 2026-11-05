import sys,tempfile,subprocess,shutil,json
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from check_correction_resume import equal
from blokus.ppo_scale import history
from blokus.value_experiment import write
r=Path(tempfile.mkdtemp(prefix='ppopool-resume-'));base=Path('artifacts/ppo-6144/seed-1401/iteration-6144.pt')
try:
 def run(name,step,resume):
  out=r/name;subprocess.run([sys.executable,'-m','blokus.ppo_scale','--seed','1401','--resume',str(resume),'--output',str(out),'--iterations',str(step),'--cap','120','--pool-config','artifacts/ppo-generation-pool/pool-B.json','--forbidden-prefixes','artifacts/ppo-generation-pool/forbidden-prefixes.json'],check=True);return out/f'iteration-{step}.pt'
 full=run('full',6150,base);part=run('part',6146,base);end=run('end',6150,part)
 a=torch.load(full,map_location='cpu',weights_only=False);b=torch.load(end,map_location='cpu',weights_only=False)
 checks={k:equal(a[k],b[k]) for k in ['model','past','optimizer','iteration','python_rng','numpy_rng','torch_rng','cuda_rng','pool']};checks['history']=equal(history(a,full),history(b,end));assert all(checks.values())
 write('artifacts/ppo-generation-pool/cuda-resume.json',dict(pass_=True,separate_process=True,base_iteration=6144,continuous_additional=6,first_additional=2,resumed_additional=4,past_exchange_iteration=6148,checks=checks))
finally:shutil.rmtree(r)
