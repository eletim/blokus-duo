"""Report the predeclared common-bank curve and gated confirmation."""
from pathlib import Path
import json,gzip
import numpy as np
import placement_more as n
R=n.R;e=n.e

def table(out):
 lines=['|比較|勝–分–敗|勝点率|区間|平均得点差|','|---|---:|---:|---:|---:|']
 for p,v in out.items():lines.append(f"|{p}|{v['wins']}–{v['draws']}–{v['losses']}|{100*v['point']:.2f}%|{100*v['point_adjusted_ci'][0]:.2f}–{100*v['point_adjusted_ci'][1]:.2f}%|{v['difference']:+.2f}|")
 return '\n'.join(lines)
def main():
 validation=n.read(R/'validation-summary.json');selection=n.read(R/'selection.json');audit=n.read(R/'audit.json');assert audit['pass_'] and n.read(R/'checks.json')['pass_'];curves={};paired={}
 for pair in ('PF','QF','AF'):
  rows=[g for seed in e.SEEDS for g in json.load(gzip.open(R/f'validation-{pair}-{seed}.json.gz','rt'))['games']];curves[pair]=np.array([np.mean([g['point'] for g in rows if g['prefix']==i]) for i in range(16)])
 ix=np.random.default_rng(1772).integers(16,size=(20000,16))
 for left,right in [('QF','PF'),('AF','QF')]:
  d=curves[left]-curves[right];paired[left+'-'+right]=dict(difference=float(d.mean()),ci95=np.quantile(d[ix].mean(1),[.025,.975]).tolist())
 e.write(R/'paired-curve.json',paired)
 learns=[n.read(R/f'learning-A-{seed}.json') for seed in e.SEEDS];updates=learns[0]['updates'];states=learns[0]['train_labels'];devs={}
 for label,root in [('A1536',n.OLD/'n1536'),('A3072',n.OLD/'n3072'),('A6144',R),('C6144',R)]:
  variant=label[0];devs[label]=float(np.mean([n.read(root/f'learning-{variant}-{seed}.json')['curve'][-1]['dev_mse'] for seed in e.SEEDS]))
 confirmation=n.read(R/'confirmation-summary.json') if selection['passed'] else None
 if confirmation:
  passed=n.read(R/'decision.json')['passed'];ctext='事前条件達成。' if passed else '事前条件未達。';ctext+='以下は別32prefix・各192局の独立確認。3主要比較の補正98.3333%区間です。\n\n'+table(confirmation)
 else:ctext='選択条件未達のため、独立確認は実施していません。追加の調整・標本追加は行いません。';passed=False
 e.write(R/'report-summary.json',dict(candidate_passed=passed,validation_passed=selection['passed'],paired_curve=paired,dev_mse=devs,train_games=6144,train_states=states,updates_per_model=updates))
 pairs='\n'.join(f"- {p}：差{100*v['difference']:+.2f}ポイント、95%区間{100*v['ci95'][0]:+.2f}～{100*v['ci95'][1]:+.2f}ポイント。" for p,v in paired.items())
 timing=validation if confirmation is None else confirmation
 times='\n'.join(f"- {p}："+'、'.join(f"{k}平均{s['mean_ms']:.1f}ms・6ply完了{100*s['depth6']:.2f}%・550ms超過{100*s['over_550']:.2f}%" for k,s in v['stats'].items()) for p,v in timing.items())
 text=f'''# A評価器：学習対局3,072→6,144局の追加検証

6,144局版Aの独立確認条件：**{'達成' if passed else '未達'}**。main・Frozen gen3・Webは変更していません。

## 目的と固定条件

前回の少数prefixの段階比較だけでは勝率の頭打ちを判定できなかったため、[事前計画](PLAN.md)に従い、新規3,072局を追加しました。同じデータのepochだけを延長する試験ではありません。

学習は計6,144局・{states:,}局面、独立開発データ192局。A/C各3seed、各16epoch・{updates:,}更新。前回と同じAdam1e-4、batch64、終局勝敗MSE＋元評価からの逸脱抑制0.1です。各モデルは元gen3から開始し、policyの候補順位は別の元gen3に固定。教師は元gen3＋Hのままで、学生自身による反復自己対戦ではありません。

- P：前回の1,536局版A16。
- Q：前回に独立確認を通った3,072局版A16。
- A：今回の6,144局版A16。駒別の現在の合法配置情報をvalueへ追加。
- C：同じ6,144局で従来のvalue構造を再学習。
- F：変更していないgen3＋H。

全て幅4、最大6ply、500ms上限。native CPU、torch1thread、core0に固定して逐次対戦し、学習・重い監査は重ねていません。全モデル・評価bankは対戦開始前に固定しました。

## 共通局面での勝率曲線と選択

新16prefix×両席×3seed、各96局・計576局です。P/Q/Aは同じ相手F・開始局面・先後・seedを共有しています。勝点率は勝ち1・引分0.5。次表の区間は参考97.5%prefix-bootstrap区間で、6比較全体の有意差判定には使いません。

{table(validation)}

同じprefixで対応を取った勝点率差（記述的な95%区間、prefixごとに両席・3seedをまとめ20,000回bootstrap）：

{pairs}

選択条件はA/FとA/Qの勝点率50%超かつ平均得点差正、A/Cは勝点率50%以上、全3比較でAの550ms超過率1%以下。結果は**{'通過' if selection['passed'] else '未達'}**でした。誤差が広い場合、非有意を頭打ちの証明とはしません。

## 独立確認

{ctext}

確認の採用条件はA/F、A/Q、A/Cすべてで補正区間下限50%超、平均得点差正、550ms超過率1%以下です。選択対局や前回の試験と合算していません。3,072局版の以前の成功結果は今回の追加試験によって取り消されるものではありません。

## 誤差と時間

共通開発192局に対する16epoch終点のMSE・3seed平均：{json.dumps(devs,ensure_ascii=False)}。各epochの曲線はlearning JSONに保存しています。予測誤差と棋力は区別します。

{times}

同じ時間上限でも実消費時間は異なります。有限の2倍増量試験で学習の限界・無限の伸びを証明するものではありません。データ増量とともに更新回数も増えています。ランダム8〜12手後からの比較であり、人間対戦・空盤面からの優越は未検証です。

## 保存と監査

新規生成は{n.read(R/'generation-time.json')['seconds']/60:.2f}分、6モデル学習は{n.read(R/'training-time.json')['seconds']/60:.2f}分。対戦計測は{sum(n.read(p)['seconds'] for p in R.glob('*-time.json') if p.name in ('validation-time.json','confirmation-time.json'))/60:.2f}分です。

[監査](audit.json)：学習6,144局と評価対局を独立した座標集合実装で再生し、合法性・pass・終局得点・集計・訓練分離を確認。元訓練index、過去評価、今回train/devとの評価状態重複0。開始前のランダムprefixは訓練ラベルにしません。対称形や戦略的な類似を全て除いた意味ではありません。

[モデル検査](checks.json)：全6exportのCPU合法着手、policy不変、元モデル/data SHA、更新回数、CPU/CUDA別プロセスの実Adam/RNG保存再開を確認。棋譜・モデル・optimizer/RNG・事前計画を研究PRへ保存しています。[再現手順](README.md)。公開モデルの差し替えやサービス再起動は実施していません。
'''
 (R/'REPORT.md').write_text(text)
 print('REPORT',passed,paired,flush=True)
if __name__=='__main__':main()
