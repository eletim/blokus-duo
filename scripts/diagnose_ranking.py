"""Held-out-group diagnostics, including continuation disagreement; no model selection."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from blokus.ai import seed_all
from blokus.ranking_train import dataset,metrics
from blokus.value_search import OutcomeNet
from blokus.value_experiment import write

seed_all(301);root=Path('artifacts/paired-ranking');b,x,y,ti,vi=dataset(root/'groups.npz','cpu');results={}
paths={'prior':'artifacts/value-search/compact/seed-201/predictive-best.pt'}
paths.update({f'ranking-{s}':str(root/f'seed-{s}/selected.pt') for s in [301,302,303]})
for name,path in paths.items():
    ck=torch.load(path,map_location='cpu',weights_only=False);model=OutcomeNet('pooled');model.load_state_dict(ck['model']);model.eval()
    results[name]=metrics(model,b,x,y,vi)
    if name!='prior':
        log=[json.loads(line) for line in (Path(path).parent/'train.jsonl').read_text().splitlines()]
        assert ck['step']==min(log,key=lambda r:r['pair_mse'])['step']
d=np.load(root/'groups.npz');a=d['returns'][d['game']%10==0];i,j=np.triu_indices(4,1);diff=a[:,i]-a[:,j]
results['continuation_noise']=dict(pair_difference_disagreement_mse=float(np.square(diff[:,:,0]-diff[:,:,1]).mean()),
    continuation_order_agreement=float(((diff[:,:,0]>0)==(diff[:,:,1]>0)).mean()),
    note='two continuation policies/seeds differ; includes policy effects, not a pure sampling variance estimate')
write(root/'predictive-diagnostics.json',results);print(json.dumps(results,indent=2))
