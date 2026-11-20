"""Render measured throughput and per-move latency; not a strength curve."""
import json,gzip
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
R=Path('artifacts/gpu-generation')
a=json.loads((R/'audit.json').read_text());fig,axes=plt.subplots(1,3,figsize=(13,4))
labels=['cpu-1','gpu-1','cpu-2','gpu-2'];vals=[a['results'][k]['games_per_second'] for k in labels]
axes[0].bar(labels,vals,color=['#3977aa','#dc7940']*2);axes[0].set_ylabel('Completed games / second');axes[0].set_title('24 games, 8 CPU rule workers')
for mode,color in [('cpu','#3977aa'),('gpu','#dc7940')]:
    values=[s['elapsed_ms'] for i in (1,2) for p in (R/f'{mode}-{i}').glob('*.gz') for s in json.load(gzip.open(p,'rt'))['stats'] if not s['forced_pass']]
    values=np.sort(values);axes[1].plot(values,np.arange(1,len(values)+1)/len(values),label=mode,color=color)
axes[1].set_xscale('log');axes[1].set_xlabel('Nonpass move latency (ms, parallel generation)');axes[1].set_ylabel('Cumulative fraction');axes[1].legend();axes[1].set_title('Every nonpass move completed 6 ply')
for kind,color in [('policy','#3977aa'),('value','#dc7940')]:
    counts=np.zeros(8)
    for i in (1,2):
        h=json.loads((R/f'gpu-{i}.json').read_text())['server']['batch_histogram']
        for k,v in h.items():
            t,n=k.split(':')
            if t==kind:counts[int(n)-1]+=v
    axes[2].plot(range(1,9),np.cumsum(counts)/counts.sum(),marker='o',label=kind,color=color)
axes[2].set_xlabel('Actual GPU batch size');axes[2].set_ylabel('Cumulative fraction of batches');axes[2].legend();axes[2].set_title('Split by model and request type')
fig.suptitle('Frozen gen3, width 4 / completed depth 6 — GPU broker did not speed up generation')
fig.tight_layout();fig.savefig(R/'throughput.png',dpi=150)
