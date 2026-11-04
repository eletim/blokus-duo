"""Actual CUDA gradients and separate-process exact-resume checks for A/B/C."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import json
import subprocess
import sys
import tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from blokus.ai import seed_all
from blokus.engine import State,legal_moves
from blokus.research import PolicyValue,pack,to_device


def same(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b


def main():
    output=Path(sys.argv[1]);result=dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),methods={})
    seed_all(545)
    for method in ['A','B','C']:
        model=PolicyValue(method).cuda();s=State();logits,value=model(*to_device(pack(s,legal_moves(s)),'cuda'))
        (-logits.log_softmax(0)[0]+value.square()).backward();torch.cuda.synchronize()
        finite=all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        assert finite
        with tempfile.TemporaryDirectory(prefix='blokus-research-resume-') as d:
            root=Path(d)
            def run(name,games,resume=None):
                cmd=[sys.executable,'-m','blokus.experiment','train','--method',method,'--seed','889','--games',str(games),'--seconds','60','--output',str(root/name)]
                if resume:cmd+=['--resume',str(resume)]
                subprocess.run(cmd,check=True,capture_output=True,text=True)
                return root/name/'latest.pt'
            whole=run('whole',4);split=run('split',2);resumed=run('resumed',2,split)
            a=torch.load(whole,weights_only=False);b=torch.load(resumed,weights_only=False)
            checks={key:same(a[key],b[key]) for key in ['model','optimizer','episode','python_rng','numpy_rng','torch_rng','cuda_rng']}
            assert all(checks.values()),checks
        result['methods'][method]=dict(finite_cuda_gradients=finite,separate_process_resume=checks)
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
