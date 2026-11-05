"""Standalone learning-curve plot and computational accounting."""
import sys,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from blokus.scaling_data import ROOT
from blokus.value_experiment import write

def main():
 curves=[]
 for size in [240,960,2400]:
  p=ROOT/f'curve-{size}.json'
  if p.exists():curves.extend(json.loads(p.read_text()))
 fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
 for j,metric in enumerate(['mse','regret']):
  for i,step in enumerate([1000,3000]):
   ax=axes[i,j]
   for c,color in [('plain','#285ea8'),('decay','#d36625')]:
    rows=sorted([r for r in curves if r['step']==step and r['condition']==c],key=lambda r:r['size']);x=[r['size'] for r in rows];y=[r['mean'][metric]/r['mean']['zero_'+metric] for r in rows];lo=[min(v[metric]/v['zero_'+metric] for v in r['folds'].values()) for r in rows];hi=[max(v[metric]/v['zero_'+metric] for v in r['folds'].values()) for r in rows]
    ax.plot(x,y,'o-',color=color,label=c+' heldout');ax.fill_between(x,lo,hi,color=color,alpha=.12);ax.plot(x,[r['train'][metric]/r['train']['zero_'+metric] for r in rows],'--',color=color,label=c+' train')
   ax.axhline(1,color='black',linestyle=':');ax.set_xscale('log');ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator());ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter());ax.set_xticks(sorted({r['size'] for r in curves}),[str(s) for s in sorted({r['size'] for r in curves})]);ax.set_title(f'{"Raw residual MSE" if metric=="mse" else "Selection regret (alpha .5)"} / zero, {step} updates');ax.set_xlabel('Nominal independent training games / fold');ax.set_ylabel('Ratio (lower is better)');ax.grid(alpha=.2)
 axes[0,0].legend(fontsize=8);fig.suptitle('Fixed heldout games; shading = fold range, not confidence interval')
 fig.savefig(ROOT/'learning-curves.png',dpi=160);fig.savefig(ROOT/'learning-curves.svg');plt.close(fig)
 results=[json.loads(p.read_text()) for p in ROOT.glob('n*/result.json')];sizes=sorted({r['config']['size'] for r in results});generation=[json.loads((ROOT/f'data-summary-{s}.json').read_text()) for s in sizes]
 write(ROOT/'compute-summary.json',dict(models=len(results),updates=sum(r['step'] for r in results),gpu_main_seconds=sum(r['seconds'] for r in results),generation_seconds=sum(r['seconds'] for r in generation),datasets=generation,matplotlib=matplotlib.__version__,note='Fixed update budget, effective epochs in per-run logs. Each data-generation timing is incremental. Bank+pool game IDs unique; independent sequence count in audit.'))
if __name__=='__main__':main()
