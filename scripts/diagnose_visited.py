"""Post-run diagnostics only: never changes checkpoint selection."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from blokus.visited_train import load,metrics
from blokus.value_search import OutcomeNet
from blokus.value_experiment import write


def candidate(a,mask):return np.where(mask,a,np.inf).min(-1)


def main():
    root=Path('artifacts/visited-diversity');dev=load(root/'dev.npz','cuda');model=OutcomeNet('pooled').cuda();result={}
    for stage in ['pilot','full']:
        folder=root/stage
        if not (folder/'summary.json').exists():continue
        diagnostics=[]
        for condition in ['wide','dense']:
            for seed in [601,602,603]:
                for selector in ['raw','live']:
                    ck=torch.load(folder/condition/f'seed-{seed}'/f'{selector}.pt',weights_only=False)
                    model.load_state_dict(ck['model']);diagnostics.append(dict(condition=condition,seed=seed,selector=selector,step=ck['step'],metrics=metrics(model,dev)))
        d=np.load(folder/'dense.npz');a=d['returns'];m=d['mask'];v16=candidate(a.mean(-1),m);stats={}
        for n in [4,8]:
            v=candidate(a[...,:n].mean(-1),m);chosen=v.argmax(-1)
            stats[str(n)]=dict(best_choice_agreement_with_16=float((chosen==v16.argmax(-1)).mean()),regret_to_16_points=float((v16.max(-1)-v16[np.arange(len(v)),chosen]).mean()*40),candidate_rmse_points=float(np.sqrt(((v-v16)**2).mean())*40))
        first=candidate(a[...,:8].mean(-1),m);second=candidate(a[...,8:].mean(-1),m)
        stats['disjoint_8_vs_8']=dict(best_choice_agreement=float((first.argmax(-1)==second.argmax(-1)).mean()),candidate_rmse_points=float(np.sqrt(((first-second)**2).mean())*40))
        rows=json.loads((folder/'wide.records.json').read_text());remaining=[sum(int(x).bit_count() for x in row['root']['remaining']) for row in rows]
        coverage=dict(roots=len(rows),source_games=len({r['game'] for r in rows}),dense_source_games=len(set(d['game'].tolist())),remaining_piece_quantiles=np.quantile(remaining,[0,.25,.5,.75,1]).tolist())
        result[stage]=dict(coverage=coverage,checkpoint_diagnostics=diagnostics,label_stability=stats,label_stability_caveat='4/8 vs16 share samples; independent halves reported separately; not ground truth')
    write(root/'diagnostics.json',result);print(json.dumps({s:d['label_stability'] for s,d in result.items()},indent=2))
if __name__=='__main__':main()
