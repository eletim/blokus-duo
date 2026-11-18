"""Plot held-out learning curves; bands show seed range, not confidence intervals."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
font=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
if font.exists():font_manager.fontManager.addfont(str(font))
plt.rcParams['font.family']='Noto Sans CJK JP'
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('artifacts/value-placement-scale'));args=ap.parse_args();r=args.root
seeds=(1751401,1751402,1751403);sizes=(384,1536,3072);colors={'A':'#1976d2','B':'#e67817','C':'#67707b'};names={'A':'A：合法配置情報を追加','B':'B：持ち駒で盤面処理を調整','C':'C：従来構造を再学習'}
def load(n,v,k):return json.loads((r/f'n{n}'/f'learning-{v}-{k}.json').read_text())
base=np.mean([load(3072,'A',k)['curve'][0]['dev_mse'] for k in seeds])
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,1,figsize=(10,9));fig.suptitle('データ量と学習の進み方',fontsize=17,fontweight='bold',y=.98)
for v in 'ABC':
 vals=np.array([[load(n,v,k)['curve'][-1]['dev_mse'] for k in seeds] for n in sizes]);axs[0].plot(sizes,vals.mean(1),'-o',color=colors[v],label=names[v],lw=2.5);axs[0].fill_between(sizes,vals.min(1),vals.max(1),color=colors[v],alpha=.12)
 vals=np.array([[x['dev_mse'] for x in load(3072,v,k)['curve']] for k in seeds]);epochs=np.arange(vals.shape[1]);axs[1].plot(epochs,vals.mean(0),color=colors[v],label=names[v],lw=2.5);axs[1].fill_between(epochs,vals.min(0),vals.max(0),color=colors[v],alpha=.12)
for ax in axs:
 ax.axhline(base,color='#222',ls='--',lw=1.4,label='元のgen3（平均）');ax.set_ylabel('未使用局面の勝敗予測誤差（MSE）');ax.grid(alpha=.2)
axs[0].set_xscale('log',base=2);axs[0].set_xticks(sizes,labels=['384','1,536','3,072']);axs[0].set_xlabel('学習に使った対局数（各16周）');axs[0].legend(loc='upper right',fontsize=10);axs[0].set_title('① 同じ開発用192局で比較（誤差が小さいほど良い）',loc='left')

for v,epoch in {'A':6,'B':5,'C':7}.items():
 y=np.mean([load(3072,v,k)['curve'][epoch]['dev_mse'] for k in seeds]);axs[1].scatter([epoch],[y],color=colors[v],s=60,zorder=4)
axs[1].set_xticks([0,4,8,12,16]);axs[1].set_xlabel('学習の周回数');axs[1].set_title('② 3,072局を繰り返し学習したとき',loc='left')
fig.text(.1,.02,'線：3seedの平均／帯：seed間の最小〜最大（信頼区間ではない）\n点は開発誤差で選んだ停止点（A：6周／B：5周／C：7周）。\n予測誤差と対戦の強さは別。採用は独立した対戦で判定。',fontsize=10,color='#444');fig.tight_layout(rect=[0,.07,1,.95]);fig.savefig(r/'learning-curves.png',dpi=180);print(r/'learning-curves.png')
