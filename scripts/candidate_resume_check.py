"""Separate-process deterministic next-update check, not extra training."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,copy,json,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
import torch
from candidate_data import R,SEEDS,save
from candidate_train import objective
from blokus.gen4 import load
from blokus.engine import State
from blokus.ppo_policy import collate

def child(seed,device,out):
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 ck=torch.load(R/f'models/{seed}/epoch-8.pt',map_location=device,weights_only=False)
 p,_=load(seed,'native');obs,qs=p._observation(State())
 with torch.inference_mode():logits,_=p.model(*collate([obs],'cpu'))
 rows=[dict(obs=obs,target=torch.softmax(logits[0],0),original=torch.log_softmax(logits[0],0))]
 model=p.model.to(device);model.load_state_dict(ck['model'])
 for n,x in model.named_parameters():x.requires_grad_(n.startswith('policy.'))
 opt=torch.optim.Adam([x for x in model.parameters() if x.requires_grad],lr=1e-4);opt.load_state_dict(ck['optimizer']);rng=np.random.default_rng();rng.bit_generator.state=ck['numpy_rng'];torch.set_rng_state(ck['torch_rng'].cpu())
 if device=='cuda':torch.cuda.set_rng_state(ck['cuda_rng'].cpu())
 opt.zero_grad();loss=objective(model,rows,device);loss.backward();torch.nn.utils.clip_grad_norm_([x for x in model.parameters() if x.requires_grad],1.);opt.step()
 torch.save(dict(model={k:v.cpu() for k,v in model.state_dict().items()},optimizer=opt.state_dict(),loss=float(loss),rng_draw=rng.integers(2**31,size=8).tolist()),out)

def equal(a,b):
 if torch.is_tensor(a):return torch.equal(a.cpu(),b.cpu())
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return a==b

def main():
 p=argparse.ArgumentParser();p.add_argument('--child',action='store_true');p.add_argument('--seed',type=int);p.add_argument('--device',choices=['cpu','cuda']);p.add_argument('--out');a=p.parse_args()
 if a.child:child(a.seed,a.device,a.out);return
 result=[]
 with tempfile.TemporaryDirectory() as tmp:
  for seed in SEEDS:
   for device in ['cpu','cuda']:
    files=[Path(tmp)/f'{seed}-{device}-{i}.pt' for i in range(2)]
    for f in files:subprocess.run([sys.executable,__file__,'--child','--seed',str(seed),'--device',device,'--out',str(f)],check=True)
    left,right=[torch.load(f,map_location='cpu',weights_only=False) for f in files];assert equal(left,right)
    result.append(dict(seed=seed,device=device,separate_process_next_update_exact=True,loss=left['loss']))
 save(R/'resume-audit.json',dict(results=result,note='Same-device separate-process restored Adam/RNG next-update equality; CPU/CUDA bit equality not claimed. Uses synthetic initial-state objective only, never updates research exports.'))
if __name__=='__main__':main()
