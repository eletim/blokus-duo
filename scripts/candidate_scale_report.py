"""Render fixed scaling results; no fitting or model selection."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('artifacts/gen4-candidate-scale'); OLD=Path('artifacts/gen4-candidate-policy')
def read(p):return json.loads(p.read_text())
def main():
 fig,axs=plt.subplots(1,2,figsize=(11,4))
 for n,p in [(768,OLD),(3072,R/'n3072'),(12288,R/'n12288')]:
  d=read(p/'selection.json');axs[0].plot(range(9),d['mean_dev_loss'],marker='o',label=str(n))
 axs[0].set(xlabel='Epoch',ylabel='Reused development objective');axs[0].legend(title='Total training games')
 for i,c in enumerate(['Frozen','old768']):
  ds=[read(R/f'n{n}'/f'vs-{c}-summary.json') for n in [3072,12288]]
  y=[100*d['point'] for d in ds];ci=[d['point_family_ci'] for d in ds]
  axs[1].errorbar([0+i*.08,1+i*.08],y,yerr=[[v-100*a[0] for v,a in zip(y,ci)],[100*a[1]-v for v,a in zip(y,ci)]],fmt='o-',capsize=4,label='vs '+c)
 axs[1].axhline(50,color='gray',ls='--');axs[1].set(xticks=[.04,1.04],xticklabels=['3072','12288'],xlabel='Total training games',ylabel='Win points (%) / family CI');axs[1].legend()
 fig.tight_layout();fig.savefig(R/'learning.png',dpi=160);plt.close(fig)
 if all((p/'dev-turnover.json').exists() for p in [OLD,R/'n3072',R/'n12288']):
  rows=[]
  for n,p in [(768,OLD),(3072,R/'n3072'),(12288,R/'n12288')]:
   d=read(p/'dev-turnover.json')['results']
   for s in range(3):
    vals=[x['stages'][str(s)] for x in d];a={k:sum(v[k] for v in vals) for k in vals[0]}
    rows.append(dict(total=n,stage=s,**a))
  (R/'proxy-curve.json').write_text(json.dumps(rows,indent=2)+'\n')
  fig,ax=plt.subplots(figsize=(7,4))
  for s,label in enumerate(['Early','Middle','Late']):
   rs=[x for x in rows if x['stage']==s];ax.plot([x['total'] for x in rs],[100*(x['teacher_new']-x['teacher_old'])/x['n'] for x in rs],marker='o',label=label)
  ax.axhline(0,color='gray',ls='--');ax.set(xscale='log',xlabel='Total training games',ylabel='Teacher top-4 retention change (pp)');ax.legend();fig.tight_layout();fig.savefig(R/'proxy.png',dpi=160);plt.close(fig)
if __name__=='__main__':main()
