import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('artifacts/gpu-async512');points=[];series='controlled' if (R/'controlled-64/timing.json').exists() else 'curve'
for n in (64,128,256,512):
 x=json.loads((R/f'{series}-{n}/timing.json').read_text());t=json.loads((R/f'{series}-{n}-telemetry.json').read_text());g=x['groups'][0];h=g['batch_hist'];batch=sum(int(k)*v for k,v in h.items())/sum(h.values());util=np.mean([float(s['gpu'].split(',')[0]) for s in t['samples']]);padded=sum(max(8,1<<(int(k)-1).bit_length())*v for k,v in h.items())/sum(h.values());points.append((n,x['games']/x['seconds'],util,batch,padded))
a=np.array(points);fig,axes=plt.subplots(1,3,figsize=(13,3.7));axes[0].plot(a[:,0],a[:,1],'o-');axes[0].set_ylabel('Completed games / second');axes[1].plot(a[:,0],a[:,2],'o-',color='tab:green');axes[1].set_ylabel('GPU utilization (%)');axes[2].plot(a[:,0],a[:,3],'o-',color='tab:orange');axes[2].plot(a[:,0],a[:,4],'--',color='gray',label='Graph padded rows');axes[2].lines[0].set_label('Valid NN requests');axes[2].legend();axes[2].set_ylabel('Mean NN batch rows')
for ax in axes:ax.set_xlabel('Active-game capacity');ax.set_xticks(a[:,0]);ax.grid(alpha=.25)
fig.suptitle('Same-source development curve: seed1751401,512 games, one run/setting');fig.tight_layout();fig.savefig(R/'concurrency.png',dpi=160);plt.close(fig)
if (R/'final/async-1/timing.json').exists():
 x=json.loads((R/'final/async-1/timing.json').read_text());fig,axes=plt.subplots(2,1,figsize=(10,6));offset=0
 for g in x['groups']:
  tr=np.array(g['occupancy_trace']);axes[0].plot(offset+tr[:,0],tr[:,1],label=str(g['seed']));axes[1].plot(offset+tr[:,0],tr[:,3]);offset+=g['seconds']
 axes[0].set_ylabel('Actually active games');axes[0].legend();axes[1].set_ylabel('Completed games in seed group');axes[1].set_xlabel('Work wall seconds');fig.suptitle('Async512 replenishment and tail (formal run1)');fig.tight_layout();fig.savefig(R/'occupancy.png',dpi=160);plt.close(fig)
 names=['cpu-1','async-1','parallel-1','async-2','cpu-2'];rates=[]
 for name in names:
  t=json.loads((R/f'final/{name}-telemetry.json').read_text());rates.append(3072/t['outer_seconds'])
 fig,ax=plt.subplots(figsize=(9,4));ax.bar(names,rates,color=['gray','tab:blue','tab:orange','tab:blue','gray']);ax.set_ylabel('Games / second (outer wall incl. initialization)');ax.set_title('New3072-game identical bank: frozen width4 / complete6ply');fig.tight_layout();fig.savefig(R/'formal-throughput.png',dpi=160)
