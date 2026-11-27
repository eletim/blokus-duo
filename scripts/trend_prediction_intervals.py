"""Paired held-out prediction comparisons; no training or checkpoint selection."""
import numpy as np
import torch
from scripts.komi4_trend import R,SEEDS,read,write,predict,baseline
from blokus.margin_network import NetworkEvaluator

def run():
    torch.set_num_threads(1)
    selection=read(R/'selection.json')
    data=dict(np.load(R/'features/test.npz'))
    groups=np.unique(data['group']); rng=np.random.default_rng(20261127)
    samples=rng.integers(0,len(groups),(4000,len(groups)))
    group_indices=[np.flatnonzero(data['group']==g) for g in groups]
    def aggregate(errors):
        return np.array([np.average(errors[ix],weights=data['w'][ix]) for ix in group_indices])
    def interval(values):return np.quantile(values[samples].mean(axis=1),[.025,.975]).tolist()
    baselines={}
    for seed in SEEDS:
        p=predict(NetworkEvaluator(baseline(seed)).model,data,'cpu')
        baselines[f'baseline-{seed}']=aggregate(abs(p-data['y']))
    out={}
    for seed in SEEDS:
        p=predict(NetworkEvaluator(selection['models'][str(seed)]['checkpoint']).model,data,'cpu');errors=p-data['y'];ae=aggregate(abs(errors));bias=aggregate(errors)
        out[str(seed)]=dict(mae_ci95=interval(ae),bias_ci95=interval(bias),
            paired_mae_difference={name:dict(mean=float((ae-b).mean()),ci95=interval(ae-b)) for name,b in baselines.items() if name == f'baseline-{seed}'})
    write(R/'prediction-intervals.json',dict(groups=len(groups),bootstrap_replicates=4000,models=out,meaning='Negative paired MAE difference favors new model; fixed held-out mixed-policy outcomes, not optimal-play truth. Group resampling, no multiple-comparison correction.'))
    print(out)
if __name__=='__main__':run()
