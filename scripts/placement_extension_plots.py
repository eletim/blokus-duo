"""Render actual saved learning and fixed independent point estimates, no new trials."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('artifacts/value-placement-49152');SEEDS=(1751401,1751402,1751403)
def main():
 fig,axes=plt.subplots(2,2,figsize=(12,8))
 for row,n in enumerate((36864,49152)):
  root=R/f'n{n}';sel=json.loads((root/'checkpoint-selection.json').read_text())['epoch']
  for v,color in [('A','tab:blue'),('C','tab:orange')]:
   curves=[json.loads((root/f'learning-{v}-{s}.json').read_text())['curve'] for s in SEEDS]
   for col,key in enumerate(('train_mse','dev_mse')):
    vals=np.array([[c[key] for c in curve] for curve in curves]);axes[row,col].plot(range(17),vals.mean(0),label=v,color=color);axes[row,col].fill_between(range(17),vals.min(0),vals.max(0),alpha=.15,color=color)
  for col,key in enumerate(('train','new development')):
   ax=axes[row,col];ax.axvline(sel,color='black',ls='--',label=f'selected epoch {sel}');ax.set(title=f'{n:,} games: {key}',xlabel='Epoch',ylabel='Actual terminal WDL MSE');ax.grid(alpha=.2);ax.legend()
 fig.suptitle('Two matched-data CUDA learning stages (bands: fixed three seeds)\nDev selects one common epoch per stage; loss is not playing strength');fig.tight_layout();fig.savefig(R/'learning-curves.png',dpi=160);plt.close(fig)
 final=json.loads((R/'n49152/confirmation-summary.json').read_text());pairs=['AQ','AR','AF','AC'];names=['Old 3,072-game A','Previous 24,576-game A','Frozen gen3 + H','Same-data 49,152-game C'];ys=np.array([final[p]['point']*100 for p in pairs]);ci=np.array([final[p]['point_adjusted_ci'] for p in pairs])*100
 fig,ax=plt.subplots(figsize=(9,5));ax.errorbar(ys,range(4),xerr=np.array([ys-ci[:,0],ci[:,1]-ys]),fmt='o',capsize=5);ax.axvline(50,color='grey',ls='--');ax.set(yticks=range(4),yticklabels=names,xlabel='49,152-game A points (%) : win1 / draw0.5',title='Independent 32 prefixes x two seats x three seeds\nFour-comparison Bonferroni intervals (98.75% each)');ax.invert_yaxis();ax.grid(axis='x',alpha=.2);fig.tight_layout();fig.savefig(R/'confirmation-points.png',dpi=160);plt.close(fig)
 fig,ax=plt.subplots(figsize=(8,4));labels=[];points=[];cis=[]
 for n in (36864,49152):
  d=json.loads((R/f'n{n}/validation-summary.json').read_text())['AP'];labels.append(f'{n:,} vs '+('24,576' if n==36864 else '36,864'));points.append(100*(d['point']-.5));cis.append(100*(np.array(d['point_ci95'])-.5))
 points=np.array(points);cis=np.array(cis);ax.errorbar(range(2),points,yerr=np.array([points-cis[:,0],cis[:,1]-points]),fmt='o',capsize=5);ax.axhline(0,color='grey',ls='--');ax.axhline(3,color='orange',ls=':',label='predeclared practical +3pp');ax.set(xticks=range(2),xticklabels=labels,ylabel='Point gain over 50% (percentage points)',title='Exploratory adjacent-size trends: distinct new 8-prefix banks\nOrdinary95% cluster CI; no general ceiling claim');ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(R/'stage-trends.png',dpi=160);plt.close(fig)
if __name__=='__main__':main()
