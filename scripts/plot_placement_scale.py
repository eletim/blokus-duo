"""Plot held-out learning curves; bands show seed range, not confidence intervals."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('artifacts/value-placement-scale'));args=ap.parse_args();r=args.root
seeds=(1751401,1751402,1751403);sizes=(384,1536,3072);colors={'A':'#1976d2','B':'#e67817','C':'#67707b'};names={'A':'A: legal-placement features','B':'B: inventory-conditioned CNN','C':'C: original structure, retrained'}
def load(n,v,k):return json.loads((r/f'n{n}'/f'learning-{v}-{k}.json').read_text())
base=np.mean([load(3072,'A',k)['curve'][0]['dev_mse'] for k in seeds])
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,1,figsize=(10,9));fig.suptitle('Does more training data improve the value network?',fontsize=17,fontweight='bold',y=.98)
for v in 'ABC':
 vals=np.array([[load(n,v,k)['curve'][-1]['dev_mse'] for k in seeds] for n in sizes]);axs[0].plot(sizes,vals.mean(1),'-o',color=colors[v],label=names[v],lw=2.5);axs[0].fill_between(sizes,vals.min(1),vals.max(1),color=colors[v],alpha=.12)
 vals=np.array([[x['dev_mse'] for x in load(3072,v,k)['curve']] for k in seeds]);epochs=np.arange(vals.shape[1]);axs[1].plot(epochs,vals.mean(0),color=colors[v],label=names[v],lw=2.5);axs[1].fill_between(epochs,vals.min(0),vals.max(0),color=colors[v],alpha=.12)
for ax in axs:
 ax.axhline(base,color='#222',ls='--',lw=1.4,label='Frozen gen3 (mean)');ax.set_ylabel('Held-out outcome MSE (lower is better)');ax.grid(alpha=.2)
axs[0].set_xscale('log',base=2);axs[0].set_xticks(sizes,labels=['384','1,536','3,072']);axs[0].set_xlabel('Unique training games (nested datasets, 16 epochs each)');axs[0].legend(loc='upper right',fontsize=10);axs[0].set_title('1. More data: same 192 held-out games for every model',loc='left')
axs[1].set_xticks([0,4,8,12,16]);axs[1].set_xlabel('Epoch (3,072-game models only)');axs[1].set_title('2. Training progress at the largest dataset size',loc='left')
fig.text(.1,.02,'Lines: mean of 3 seeds. Bands: min/max seed range, not confidence intervals.\nPrediction error is not playing strength; separate matches determine adoption.',fontsize=10,color='#444');fig.tight_layout(rect=[0,.07,1,.95]);fig.savefig(r/'learning-curves.png',dpi=180);print(r/'learning-curves.png')
