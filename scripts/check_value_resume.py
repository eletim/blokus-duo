"""Separate-process strict CUDA continuation check for both outcome architectures."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from blokus.engine import State,legal_moves,play
from blokus.value_search import encode


def main():
    results={}
    with tempfile.TemporaryDirectory(prefix='blokus-value-resume-') as tmp:
        root=Path(tmp);s=State();boards=[];states=[]
        for i in range(20):
            b,x=encode(s);boards.append(np.packbits(b.reshape(-1)));states.append(x)
            qs=legal_moves(s);play(s,qs[0].move if qs else None)
        data=root/'data.npz';np.savez(data,board=boards,state=states,target=np.linspace(-.2,.2,20,dtype=np.float32),kind=np.zeros(20,dtype=np.uint8),game=np.arange(1,21))
        for arch in ['flat','pooled']:
            common=[sys.executable,'-m','blokus.value_experiment','train','--data',str(data),'--arch',arch,'--augment','--seed','88','--seconds','30']
            for label,steps,resume in [('full',4,None),('part',2,None),('resumed',2,root/f'{arch}-part/latest.pt')]:
                cmd=common+['--steps',str(steps),'--output',str(root/f'{arch}-{label}')]
                if resume:cmd+=['--resume',str(resume)]
                subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL)
            a=torch.load(root/f'{arch}-full/latest.pt',weights_only=False);b=torch.load(root/f'{arch}-resumed/latest.pt',weights_only=False)
            assert a['step']==b['step']==4
            for name in ['model']:
                assert all(torch.equal(v,b[name][k]) for k,v in a[name].items())
            assert torch.equal(a['torch_rng'],b['torch_rng'])
            assert all(torch.equal(x,y) for x,y in zip(a['cuda_rng'],b['cuda_rng']))
            for k,vs in a['optimizer']['state'].items():
                assert all(torch.equal(v,b['optimizer']['state'][k][n]) for n,v in vs.items())
            results[arch]='PASS: model/optimizer/CPU and CUDA RNG bit identical; augmentation enabled'
    out=Path('artifacts/value-search/resume-check.json');out.write_text(json.dumps(dict(results=results,torch=torch.__version__,gpu=torch.cuda.get_device_name()),indent=2));print(out.read_text())
if __name__=='__main__':main()
