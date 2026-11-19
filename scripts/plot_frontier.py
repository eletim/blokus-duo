import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
R=Path('artifacts/gpu-frontier');a=json.loads((R/'audit.json').read_text());fig,ax=plt.subplots(1,3,figsize=(14,4))
labels=[];rates=[]
for mode in ('cpu','parallel'):
 for i in (1,2):
  r=a['results'][f'{mode}-{i}'];labels.append(f'{mode}\nrepeat {i}');rates.append(192/(r['seconds']+r['initialization_seconds']))
ax[0].bar(labels,rates,color=['#5278ad']*2+['#cc7049']*2);ax[0].set_ylabel('Completed games / second (including init)');ax[0].set_title('Same 192 games, width 4, completed depth 6')
for i in (1,2):
 r=json.loads((R/f'parallel-{i}.json').read_text());hist={};parts=np.zeros(5)
 for g in r['groups']:
  for n,c in g['batches'].items():hist[int(n)]=hist.get(int(n),0)+c
  parts+=np.array([g['pack_seconds'],g['forward_seconds'],g['return_seconds'],g['frontier_wait_receive_seconds'],g['reply_send_seconds']])
 ns=sorted(hist);ax[1].step(ns,np.cumsum([hist[n] for n in ns])/sum(hist.values()),where='post',label=f'repeat {i}')
 if i==1:
  ax[2].barh(['pack + H2D','forward + sync','D2H return','CPU frontier + receive','reply send'],parts)
ax[1].set_xlabel('Actual NN batch size');ax[1].set_ylabel('Cumulative fraction of calls');ax[1].legend();ax[1].set_title('Same model grouped, policy/value separate')
ax[2].set_xlabel('Server wall seconds (repeat 1)');ax[2].set_title('Server wall breakdown\n(includes dispatch and synchronization)')
fig.suptitle('GPU frontier generation: throughput is separate from playing strength');fig.tight_layout();fig.savefig(R/'throughput.png',dpi=150,bbox_inches='tight')
