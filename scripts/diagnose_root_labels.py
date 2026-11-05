"""Post-evaluation tie-label diagnosis only; never select or modify models."""
import sys,gzip,json
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from blokus.root_data import ROOT
from blokus.root_policy import RootPolicy
from blokus.ai import seed_all
from blokus.value_experiment import write

def main():
    seed_all(901);rs=[r for r in json.load(gzip.open(ROOT/'teacher-traces.json.gz','rt')) if r['trainable']];data=np.load(ROOT/'data.npz');counts=[]
    for r in rs:
        t=r['target'];r['ties']=[i for i,(d,v) in enumerate(zip(r['depths'],r['final'])) if d==r['depths'][t] and abs(v-r['final'][t])<=4e-6];counts.append(len(r['ties']))
    ids=np.where(data['split']==1)[0];x=torch.as_tensor(data['x'][ids],device='cuda');st=torch.as_tensor(data['state'][ids],device='cuda');out=[]
    for step in [0,1000,3000]:
        for seed in [901,902,903]:
            path=ROOT/f'seed-{seed}'/('untrained.pt' if step==0 else f'step-{step}.pt');m=RootPolicy().cuda().eval();m.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model'])
            with torch.no_grad():p=m(x,st).cpu().numpy()+data['prior'][ids]
            p[~data['legal_mask'][ids]]=-1e9;pred=p.argmax(1)
            out.append(dict(seed=seed,step=step,dev_exact=float(np.mean(pred==data['target'][ids])),dev_same_depth_teacher_tie=float(np.mean([int(q) in rs[int(i)]['ties'] for i,q in zip(ids,pred)]))))
    result=dict(eligible=len(rs),multiple_tie_rows=sum(c>1 for c in counts),dev_multiple_tie_rows=sum(counts[int(i)]>1 for i in ids),dev_rows=len(ids),mean_same_depth_ties=float(np.mean(counts)),metrics=out,note='Post-evaluation diagnosis only; no model/checkpoint change. Equal same-depth teacher values are not proven equal true game values.')
    write(ROOT/'label-noise.json',result);print(result)
if __name__=='__main__':main()
