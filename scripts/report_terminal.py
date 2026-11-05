import sys,json,argparse
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from blokus.terminal_ppo import ROOT
from blokus.value_experiment import write

def main():
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(2,2,figsize=(10,7));runs=[]
 for seed in [1401,1402,1403]:
  r=json.loads((ROOT/f'seed-{seed}/result.json').read_text());logs=r['logs'];runs.append(r);groups=[logs[i:i+8] for i in range(0,len(logs),8)];x=[g[-1]['iteration'] for g in groups]
  for ax,key,title in [(axes[0,0],'value_mse','PPO value MSE vs terminal outcome'),(axes[0,1],'entropy','Policy entropy (nats)'),(axes[1,0],'approx_kl','Within-update approximate KL')]:ax.plot(x,[np.mean([l['update'][key] for l in g]) for g in groups],label=str(seed));ax.set_title(title)
  axes[1,1].plot(x,[np.mean([(l['opponents']['heuristic']+1)/2 for l in g]) for g in groups],label=str(seed));axes[1,1].set_title('Training point rate vs heuristic')
 for ax in axes.flat:ax.set_xlabel('Collection/update iterations');ax.grid(alpha=.3);ax.legend()
 fig.tight_layout();fig.savefig(ROOT/'curves.png',dpi=150);fig.savefig(ROOT/'curves.svg');plt.close(fig)
 svg=ROOT/'curves.svg';svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
 totals=dict(wall_seconds=sum(r['seconds'] for r in runs),games=sum(r['games'] for r in runs),ppo_updates=sum(l['update']['updates'] for r in runs for l in r['logs']),records=sum(l['update']['records'] for r in runs for l in r['logs']),generation_seconds=sum(l['generation']['seconds'] for r in runs for l in r['logs']),cpu_generation_pack_seconds=sum(l['generation']['cpu_generation_pack'] for r in runs for l in r['logs']),inference_collation_transfer_sync_seconds=sum(l['generation']['gpu_inference_sync'] for r in runs for l in r['logs']),update_seconds=sum(l['update']['seconds'] for r in runs for l in r['logs']))
 totals['unattributed_wall_seconds']=totals['wall_seconds']-totals['generation_seconds']-totals['update_seconds'];totals['note']='GPU inference phase includes collation/H2D/synchronization; not pure CUDA kernel time. Residual includes prefix init, Python dispatch, logging/checkpoint I/O. No external wait worker.';write(ROOT/'compute.json',totals)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',default=str(ROOT));ROOT=Path(p.parse_args().root);main()
