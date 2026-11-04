"""Decision/ordering metrics using actual alpha and exact-terminal overrides."""
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from blokus.ai import seed_all
from blokus.engine import State
from blokus.aligned_data import replies_group,BASE
from blokus.value_search import ValueEvaluator
from blokus.value_experiment import write


def main():
    seed_all(501);root=Path('artifacts/search-aligned');d=np.load(root/'groups.npz');metadata=json.loads((root/'groups.groups.json').read_text())
    ids=np.flatnonzero(d['game']%10==0);leaves=[];ranges=[]
    for i in ids:
        group,_,_=replies_group(State(**metadata[i]['root']));row=[]
        for branch in group:
            begin=len(leaves);leaves.extend(branch);row.append((begin,len(leaves)))
        ranges.append(row)
    target=np.where(d['mask'][ids],d['returns'][ids].mean(-1),np.inf).min(-1);i,j=np.triu_indices(4,1);td=target[:,i]-target[:,j];significant=np.abs(td)>.025
    models={'prior':BASE}
    for mode in ['absolute','difference','ranking']:
        for seed in [501,502,503]:models[f'{mode}-{seed}']=str(root/mode/f'seed-{seed}/selected.pt')
    output={}
    for name,path in models.items():
        evaluator=ValueEvaluator(path,'cpu',.5,24);v=np.concatenate([evaluator.batch(leaves[k:k+256]) for k in range(0,len(leaves),256)])
        pred=np.array([[v[a:b].min() for a,b in row] for row in ranges]);choice=pred.argmax(-1);regret=target.max(-1)-target[np.arange(len(pred)),choice];pd=pred[:,i]-pred[:,j]
        output[name]=dict(decision_regret_points=float(regret.mean()*40),optimal_choice_rate=float((regret<1e-7).mean()),ordering_accuracy=float(((pd>0)==(td>0))[significant].mean()),pair_mse=float(np.square(pd-td).mean()),groups=len(ids),alpha=.5,terminal_override=True)
    write(root/'actual-inference-diagnostics.json',output);print(json.dumps(output,indent=2))
if __name__=='__main__':main()
