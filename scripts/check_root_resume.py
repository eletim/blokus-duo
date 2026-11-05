"""Separate-process CUDA continuation; retain equality result, remove duplicate ckpts."""
import sys,json,subprocess,tempfile,shutil
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from blokus.root_data import ROOT
from blokus.research import sha
from blokus.value_experiment import write

def equal(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

def main():
    root=Path(tempfile.mkdtemp(prefix='root-resume-'))
    try:
        def run(name,steps,resume=None):
            cmd=[sys.executable,'-m','blokus.root_train','--output',str(root/name),'--seed','919','--steps',str(steps)]
            if resume:cmd+=['--resume',str(resume)]
            subprocess.run(cmd,check=True);return root/name/'latest.pt'
        full=run('full',40);part=run('part',20);resumed=run('resumed',40,part)
        a=torch.load(full,map_location='cpu',weights_only=False);b=torch.load(resumed,map_location='cpu',weights_only=False)
        checks={k:equal(a[k],b[k]) for k in ['model','optimizer','step','python_rng','numpy_rng','torch_rng','cuda_rng','data_sha256']};assert all(checks.values())
        write(ROOT/'cuda-resume.json',dict(pass_=True,separate_process=True,checks=checks,full_sha256=sha(full),resumed_sha256=sha(resumed),steps=40,note='file hashes differ because timings/paths metadata differ; tensors/optimizer/RNG bit equal'))
    finally:shutil.rmtree(root)
if __name__=='__main__':main()
