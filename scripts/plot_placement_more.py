"""Plot common-bank validation only; intervals are descriptive, not a plateau proof."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
R=Path('artifacts/value-placement-6144');OLD=Path('artifacts/value-placement-scale')
font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
if Path(font).exists():font_manager.fontManager.addfont(font);plt.rcParams['font.family']=font_manager.FontProperties(fname=font).get_name()
plt.rcParams.update({'font.size':13,'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False})
out=json.loads((R/'validation-summary.json').read_text());fig,axes=plt.subplots(1,2,figsize=(15,6),layout='constrained');fig.suptitle('合法配置情報を加えたA：学習対局を6,144局へ拡大',fontsize=20)
x=np.arange(3);pairs=['PF','QF','AF'];y=np.array([out[p]['point']*100 for p in pairs]);lo=np.array([out[p]['point_adjusted_ci'][0]*100 for p in pairs]);hi=np.array([out[p]['point_adjusted_ci'][1]*100 for p in pairs]);axes[0].errorbar(x,y,yerr=[y-lo,hi-y],fmt='o-',color='#136f63',capsize=7,lw=2,markersize=9)
axes[0].axhline(50,color='#888',ls='--');axes[0].set_xticks(x,['1,536局','3,072局','6,144局']);axes[0].set_ylabel('現行gen3＋Hへの勝点率（％）');axes[0].set_xlabel('学習対局数（いずれも16周）');axes[0].set_title('同じ新規16開始局面・先後・3seedで比較');axes[0].set_ylim(min(35,float(lo.min()-5)),max(75,float(hi.max()+5)));axes[0].grid(axis='y',alpha=.2)
for a,b in zip(x,y):axes[0].annotate(f'{b:.1f}%',(a,b),xytext=(0,12),textcoords='offset points',ha='center')
for n,col in [(1536,'#8995a8'),(3072,'#3269b0'),(6144,'#136f63')]:
 root=R if n==6144 else OLD/f'n{n}';curves=np.array([[r['dev_mse'] for r in json.loads((root/f'learning-A-{seed}.json').read_text())['curve']] for seed in (1751401,1751402,1751403)]);t=np.arange(curves.shape[1]);axes[1].plot(t,curves.mean(0),color=col,label=f'{n:,}局');axes[1].fill_between(t,curves.min(0),curves.max(0),color=col,alpha=.12)
axes[1].set_xlabel('学習の周回数');axes[1].set_ylabel('共通の開発192局での予測誤差（低い方が良い）');axes[1].set_title('予測誤差は勝率とは別の指標');axes[1].legend();axes[1].grid(alpha=.2);axes[1].set_xticks([0,4,8,12,16])
fig.supxlabel('左：各96対局、開始局面単位の参考97.5％区間。右：3seed平均、帯は最小〜最大。\n確認試験とは別の図。有限標本の横ばいだけでは、強さの限界を断定できません。',fontsize=11)
fig.savefig(R/'learning-winrate.png',dpi=150);plt.close(fig)
