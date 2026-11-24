"""Descriptive plots and fixed-ID occupancy examples after all comparisons."""
from future_common import *
from future_eval import model,PAIRS
from blokus.future_occupancy import terminal_labels
from blokus.value_inputs import encode
from blokus.placement_value import mobility
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 torch.set_num_threads(1);sel=json.loads((R/'selection.json').read_text());fig,ax=plt.subplots(figsize=(7,4))
 for arm in 'ABC':
  curves=np.array([[r['dev']['value_mse'] for r in sel['models'][f'{arm}-{seed}']['curve']] for seed in SEEDS]);ax.plot(range(9),curves.mean(0),marker='o',label=arm)
 ax.set(xlabel='Epoch',ylabel='Development value MSE (mean over seeds)');ax.legend();fig.tight_layout();fig.savefig(R/'learning.png',dpi=160);plt.close(fig)
 fig,ax=plt.subplots(figsize=(8,4));summary={}
 for i,(a,b) in enumerate(PAIRS):
  d=json.loads((R/f'eval-{a+b}.json').read_text());summary[a+b]=d;p=d['point']*100;lo,hi=np.array(d['family_ci'])*100;ax.errorbar(i,p,yerr=[[p-lo],[hi-p]],fmt='o',capsize=4)
 ax.axhline(50,ls='--',color='gray');ax.set(xticks=range(5),xticklabels=[a+'/'+b for a,b in PAIRS],ylabel='Win points (%) / 99% prefix CI');fig.tight_layout();fig.savefig(R/'comparison.png',dpi=160);plt.close(fig)
 adopt=all(summary[p]['family_ci'][0]>.5 and summary[p]['difference']>0 for p in ['CG','CA','CB']);save(R/'conclusion.json',dict(C_adoption_criterion=adopt,rule='corrected superiority over G,A,B and positive point score differences',no_public_deployment=True))
 g=json.loads((R/'games/test/00000.json').read_text());_,ev=model(SEEDS[0],'C');examples=[]
 for ply in [g['prefix_length'],20,28]:
  if ply>=len(g['moves'])-2:continue
  s=state(g['moves'][:ply]);b,st=encode(s);y=terminal_labels(g['boards'],s.turn)
  with torch.inference_mode():v,l,p=ev.model(torch.tensor(b[None,:4]),torch.tensor(st[None]),torch.tensor(mobility(s)[None]));probs=p[0].numpy()
  fig,axs=plt.subplots(1,5,figsize=(15,3));axs[0].imshow(np.where(b[0],0,np.where(b[1],1,2)),vmin=0,vmax=2,cmap='viridis');axs[0].set_title(f'Current ply {ply}')
  for cls,label in enumerate(['Self','Opponent','Empty']):axs[cls+1].imshow(probs[cls],vmin=0,vmax=1,cmap='magma');axs[cls+1].set_title(label+' probability')
  axs[4].imshow(y,vmin=0,vmax=2,cmap='viridis');axs[4].set_title('Actual terminal')
  for ax in axs:ax.set_xticks([]);ax.set_yticks([])
  fig.suptitle(f'Fixed test game 0 / C seed1751401 / perspective player{s.turn}');fig.tight_layout();fig.savefig(R/f'occupancy-ply{ply}.png',dpi=150);plt.close(fig);examples.append(dict(game=0,ply=ply,turn=s.turn,predicted_value=float(v),probabilities=probs.tolist(),actual=y.tolist(),current_empty=(~b[:2].any(0)).tolist()))
 save(R/'heatmap-examples.json',dict(selection='fixed test game0 plies prefix/20/28; no best-case selection',examples=examples))
if __name__=='__main__':main()
