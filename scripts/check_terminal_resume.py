import sys,subprocess,tempfile,shutil,json
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from check_correction_resume import equal
from blokus.value_experiment import write
from blokus.terminal_ppo import ROOT
root=Path(tempfile.mkdtemp(prefix='terminal-resume-'))
try:
 def run(name,n,resume=None):
  out=root/name;cmd=[sys.executable,'-m','blokus.terminal_ppo','--iterations',str(n),'--batch','8','--output',str(out)]
  if resume:cmd+=['--resume',str(resume)]
  subprocess.run(cmd,check=True);return out/'latest.pt'
 full=run('full',6);part=run('part',4);end=run('end',6,part);a=torch.load(full,weights_only=False,map_location='cpu');b=torch.load(end,weights_only=False,map_location='cpu');checks={k:equal(a[k],b[k]) for k in ['model','past','optimizer','iteration','games_gzip','python_rng','numpy_rng','torch_rng','cuda_rng']};assert all(checks.values())
 init=torch.load(full.parent/'untrained.pt',weights_only=False,map_location='cpu');assert any(not torch.equal(v,init['model'][k]) for k,v in a['model'].items());assert not torch.equal(a['model']['policy.2.weight'],init['model']['policy.2.weight']);assert all(l['update']['grad_norm']>0 for l in a['logs']);assert all(l['update']['approx_kl']>=-1e-6 for l in a['logs'])
 write(ROOT/'cuda-resume.json',dict(pass_=True,separate_process=True,continuous_iterations=6,first_iterations=4,resumed_additional_iterations=2,games=48,checks=checks,nonzero_gradient=True,parameters_changed=True,policy_head_changed=True))
finally:shutil.rmtree(root)
