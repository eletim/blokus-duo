"""40 uninterrupted updates vs 20 + additional20, separate CUDA processes."""
import sys,tempfile,subprocess,json,shutil
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from check_correction_resume import equal
from blokus.scaling_data import ROOT
from blokus.value_experiment import write

def main():
 root=Path(tempfile.mkdtemp(prefix='scaling-resume-'));reports=[]
 try:
  for condition in ['plain','decay']:
   def run(name,target,resume=None):
    out=root/(condition+name);cmd=[sys.executable,'-m','blokus.scaling_train','--size','240','--condition',condition,'--steps',str(target),'--output',str(out)]
    if resume:cmd+=['--resume',str(resume)]
    subprocess.run(cmd,check=True);return out/'latest.pt'
   full=run('full',40);part=run('part',20);end=run('end',40,part);a=torch.load(full,map_location='cpu',weights_only=False);b=torch.load(end,map_location='cpu',weights_only=False)
   checks={k:equal(a[k],b[k]) for k in ['model','optimizer','step','python_rng','numpy_rng','torch_rng','cuda_rng','data_sha256']};assert all(checks.values());reports.append(dict(condition=condition,uninterrupted_updates=40,first_segment_updates=20,resumed_additional_updates=20,resumed_total_updates=40,checks=checks))
  write(ROOT/'cuda-resume.json',dict(pass_=True,separate_process=True,reports=reports,previous_report_correction='40 continuous versus 20 + additional 20 = 40; --steps is total target, not additional updates'))
 finally:shutil.rmtree(root)
if __name__=='__main__':main()
