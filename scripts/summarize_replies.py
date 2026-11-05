"""Game-unit diagnostic uncertainty; no checkpoint or parameter selection."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from blokus.value_experiment import write

root=Path('artifacts/reply-diagnostics');data=json.loads((root/'diagnostic.json').read_text())['states'];out={}
def ci(values):
    a=np.asarray(values,float);idx=np.random.default_rng(804).integers(len(a),size=(4000,len(a)));return dict(mean=float(a.mean()),ci95=np.quantile(a[idx].mean(1),[.025,.975]).tolist(),unit='independent source game',games=len(a))
for kind in ['mid','end']:
    ds=[d for d in data if d['kind']==kind];z={k:ci([d['metrics'][k] for d in ds]) for k in ds[0]['metrics']}
    z['root6_omission_rate']=ci([d['metrics']['root6_loss']>1e-6 for d in ds])
    z['old_selected_reply_omission_rate']=ci([min(max(d['records'],key=lambda r:min(r['shallow'][:3]))['reference'][:3])>min(max(d['records'],key=lambda r:min(r['shallow'][:3]))['reference'])+1e-6 for d in ds])
    z['configs']=[]
    for i in range(8):
        cs=[d['fixed_node'][i] for d in ds];z['configs'].append(dict(root_width=cs[0]['root_width'],reply_width=cs[0]['reply_width'],depth=cs[0]['depth'],regret=ci([c['regret'] for c in cs]),mean_nodes=float(np.mean([c['nodes'] for c in cs])),mean_completed_roots=float(np.mean([len(c['values']) for c in cs])),interrupted=sum(c['interrupted'] for c in cs)))
    z['widening_regret_change']=ci([d['fixed_node'][6]['regret']-d['fixed_node'][4]['regret'] for d in ds])
    out[kind]=z
write(root/'diagnostic-uncertainty.json',out)
print(json.dumps(out,indent=2))
# Raw wins/game (draws count as zero), clustered by the paired prefix.
if (root/'holdout-results.json').exists():
    report=json.loads((root/'holdout-results.json').read_text());groups={}
    for g in report['games']:groups.setdefault(g['prefix_id'],[]).append(g)
    a=np.asarray([np.mean([g['difference']>0 for g in gs]) for gs in groups.values()]);idx=np.random.default_rng(805).integers(len(a),size=(4000,len(a)))
    write(root/'evaluation-uncertainty.json',dict(raw_win_rate=float(a.mean()),raw_win_rate_ci95=np.quantile(a[idx].mean(1),[.025,.975]).tolist(),bootstrap_unit='prefix pair',prefixes=len(a),games=len(report['games']),seed=805,draws=4000))
