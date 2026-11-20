"""Standalone learning and new strength figures, no extrapolated strength."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('artifacts/value-placement-24576');seeds=(1751401,1751402,1751403)
selection=json.loads((R/'checkpoint-selection.json').read_text());fig,axes=plt.subplots(1,2,figsize=(11,4.2))
for variant,color in [('A','#1469a8'),('C','#d66b21')]:
 curves=[json.loads((R/f'learning-{variant}-{s}.json').read_text())['curve'] for s in seeds]
 for ax,key in zip(axes,['train_mse','dev_mse']):
  ys=np.array([[r[key] for r in c] for c in curves]);ax.plot(range(17),ys.mean(0),label=variant,color=color);ax.fill_between(range(17),ys.min(0),ys.max(0),alpha=.15,color=color);ax.axvline(selection['epoch'],color='black',ls=':',alpha=.5);ax.set_xlabel('Epoch (matched examples / optimizer steps)');ax.set_ylabel('Game-weighted terminal WDL MSE');ax.grid(alpha=.2);ax.legend()
axes[0].set_title('24,576 training games');axes[1].set_title('192 new development games; shade = seed range');fig.suptitle('Real CUDA updates; one common epoch selected only by development MSE');fig.tight_layout();fig.savefig(R/'learning-curves.png',dpi=160);plt.close(fig)
phases=[p for p in ('validation','confirmation') if (R/f'{p}-summary.json').exists()];fig,axes=plt.subplots(1,len(phases),figsize=(6*len(phases),4.8),squeeze=False)
for ax,phase in zip(axes[0],phases):
 d=json.loads((R/f'{phase}-summary.json').read_text())
 for i,p in enumerate(('AQ','AF','AC')):
  v=d[p];center=100*v['point'];lo,hi=np.array(v['point_adjusted_ci'])*100;ax.errorbar(center,i,xerr=[[center-lo],[hi-center]],fmt='o',capsize=5,color='#1469a8');ax.text(center,i+.18,f"{center:.2f}% | score {v['difference']:+.2f}",ha='center',fontsize=9)
 ax.set_yticks(range(3),['new A vs old 3,072 A','new A vs Frozen gen3+H','new A vs matched-data C']);ax.axvline(50,color='black',ls='--');ax.set_xlim(0,100);ax.set_ylim(-.5,2.6);ax.set_xlabel('Win=1, draw=0.5 (%)');ax.set_title(phase+'; 3-seed prefix clusters\nBonferroni 98.33% intervals (3 primary pairs)');ax.grid(axis='x',alpha=.2)
fig.tight_layout();fig.savefig(R/'strength-points.png',dpi=160);plt.close(fig)
print('FIGURES SAVED')
