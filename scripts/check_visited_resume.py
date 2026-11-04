"""Dual checkpoint selectors: separate-process exact CUDA resume."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tests.test_visited import fixture
from blokus.value_experiment import write


def main():
    results={}
    with tempfile.TemporaryDirectory(prefix='blokus-aligned-resume-') as tmp:
        root=Path(tmp);data=root/'data.npz';fixture(data)
        for mode in ['absolute']:
            common=[sys.executable,'-m','blokus.visited_experiment','train','--data',str(data),'--dev',str(data),'--seconds','30','--seed','601']
            for name,steps,resume in [('full',4,None),('part',2,None),('resumed',2,root/f'{mode}-part/latest.pt')]:
                cmd=common+['--output',str(root/f'{mode}-{name}'),'--steps',str(steps)]
                if resume:cmd+=['--resume',str(resume)]
                subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL)
            a=torch.load(root/f'{mode}-full/latest.pt',weights_only=False);b=torch.load(root/f'{mode}-resumed/latest.pt',weights_only=False)
            assert a['step']==b['step']==4 and all(torch.equal(v,b['model'][k]) for k,v in a['model'].items())
            assert torch.equal(a['torch_rng'],b['torch_rng']) and a['python_rng']==b['python_rng']
            assert a['numpy_rng'][0]==b['numpy_rng'][0] and np.array_equal(a['numpy_rng'][1],b['numpy_rng'][1]) and a['numpy_rng'][2:]==b['numpy_rng'][2:]
            assert all(torch.equal(x,y) for x,y in zip(a['cuda_rng'],b['cuda_rng']))
            for k,vs in a['optimizer']['state'].items():assert all(torch.equal(v,b['optimizer']['state'][k][n]) for n,v in vs.items())
            results[mode]='PASS: model/optimizer/Python/NumPy/Torch/CUDA RNG bit equal'
    write('artifacts/visited-diversity/resume-check.json',dict(results=results,torch=torch.__version__,gpu=torch.cuda.get_device_name()));print(results)
if __name__=='__main__':main()
