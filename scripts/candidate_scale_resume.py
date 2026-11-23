"""New-stage separate-process checkpoint check; does not update exports."""
import os,argparse,subprocess,sys,tempfile,json
from pathlib import Path
import torch
import candidate_resume_check as check
from candidate_scale_data import R,SEEDS,save

def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',type=int,choices=[3072,12288],required=True);p.add_argument('--child',action='store_true');p.add_argument('--seed',type=int);p.add_argument('--device',choices=['cpu','cuda']);p.add_argument('--out');a=p.parse_args();check.R=R/f'n{a.stage}'
 if a.child:check.child(a.seed,a.device,a.out);return
 result=[]
 with tempfile.TemporaryDirectory() as tmp:
  for seed in SEEDS:
   for device in ['cpu','cuda']:
    files=[Path(tmp)/f'{seed}-{device}-{i}.pt' for i in range(2)]
    for f in files:subprocess.run([sys.executable,__file__,'--stage',str(a.stage),'--child','--seed',str(seed),'--device',device,'--out',str(f)],check=True)
    a1,a2=[torch.load(f,map_location='cpu',weights_only=False) for f in files];assert check.equal(a1,a2);result.append(dict(seed=seed,device=device,exact=True,loss=a1['loss']))
 save(check.R/'resume-audit.json',dict(results=result,note='Same-device separate-process Adam/RNG next step exact; no CPU/CUDA bit equivalence claim. Synthetic check never changes selected exports.'))
if __name__=='__main__':main()
