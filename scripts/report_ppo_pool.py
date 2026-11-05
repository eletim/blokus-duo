"""Describe a finite controlled pool experiment without selecting seeds."""
import sys,json,time
from pathlib import Path
from collections import Counter
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from blokus.value_experiment import write
R=Path('artifacts/ppo-generation-pool');selection=json.loads((R/'selection.json').read_text());budget=json.loads((R/'budget.json').read_text());audit=json.loads((R/'evaluation-audit.json').read_text());decision=json.loads((R/'decision.json').read_text());rows=[]
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(2,3,figsize=(13,7))
for kind in ('A','B'):
 for seed in (1401,1402,1403):
  root=R/kind/f'seed-{seed}';logs=[l for l in json.loads((root/'curve.json').read_text()) if l['iteration']>6144];games=Counter();states=Counter()
  for l in logs:games.update(l['game_counts']);states.update(l['state_counts'])
  row=dict(kind=kind,seed=seed,iterations=len(logs),games=sum(games.values()),states=sum(states.values()),game_counts=dict(games),state_counts=dict(states),game_ratios={m:v/sum(games.values()) for m,v in games.items()},state_ratios={m:v/sum(states.values()) for m,v in states.items()},updates=sum(l['update']['updates'] for l in logs),generation_seconds=sum(l['generation']['seconds'] for l in logs),cpu_generation_pack_seconds=sum(l['generation']['cpu_generation_pack'] for l in logs),gpu_inference_sync_seconds=sum(l['generation']['gpu_inference_sync'] for l in logs),update_seconds=sum(l['update']['seconds'] for l in logs),wall_seconds=next(x['wall_seconds'] for x in budget['rows'] if x['kind']==kind and x['seed']==seed),last64={k:float(np.mean([l['update'][k] for l in logs[-64:]])) for k in ('entropy','approx_kl','value_mse','clip_fraction')},opponent_point={m:float(sum((l['opponents'][m]+1)/2*l['game_counts'][m] for l in logs if m in l['opponents'])/games[m]) for m in games})
  rows.append(row)
  for ax,key in zip(axes.flat,('entropy','approx_kl','value_mse','clip_fraction','heuristic','past')):
   values=[l['update'][key] if key in row['last64'] else (l['opponents'][key]+1)/2 for l in logs];ax.plot([l['iteration'] for l in logs][63:],np.convolve(values,np.ones(64)/64,'valid'),label=f'{kind}/{seed}');ax.set_title(key);ax.legend(fontsize=6)
fig.tight_layout();fig.savefig(R/'learning-curves.png',dpi=140);plt.close(fig)
write(R/'compute.json',dict(rows=rows,budget=budget,note='Game and PPO-call counts matched; realized state/minibatch counts differ due trajectories and unchanged early-KL. GPU sync region is not pure kernel time.'))
lines=['# 複数世代pool A/B 対照実験','', 'main gen2維持。各seed最新6144の同一model/optimizer/past/RNGから分岐し、同じloss/報酬/モデル/LR/batch16/2epochs/early-KLで有限継続。A current50/past25/heuristic25。B current50/heuristic25/past・同seedgen2・同seed2048を各約8.33%。3iterationで過去枠比率均等、6iterationで先後均等、端数は以下の実数を参照。更新はcurrent自己対戦の双方とその他対局のlearner席のみ。', '', '最新6144は前回selected5120/5120/6144と区別する。固定gen2は3072/4096/4096、固定2048は各seed2048。全SHAはlock.jsonとpool-B.json。既存の成績をAの代用にせずAも新規実行した。', '', '## 計算量・割合','', '|条件/seed|追加局|PPO呼出|minibatch更新|wall秒|CPU生成秒|GPU推論同期秒|更新秒|','|---|---:|---:|---:|---:|---:|---:|---:|']
for row in rows:lines.append(f"|{row['kind']}/{row['seed']}|{row['games']}|{row['iterations']}|{row['updates']}|{row['wall_seconds']:.1f}|{row['cpu_generation_pack_seconds']:.1f}|{row['gpu_inference_sync_seconds']:.1f}|{row['update_seconds']:.1f}|")
lines+=['',f"6run合計wall {budget['total_seconds']:.1f}秒（生成＋更新＋保存、上限5400秒）。追加ゲーム数とPPO呼出数を揃え、実minibatch数や学習state数の差を隠さない。early-KLを変えて厳密な等勾配step数にした実験ではない。", '', '|条件/seed|対局割合|学習state割合|', '|---|---|---|']
for row in rows:lines.append(f"|{row['kind']}/{row['seed']}|{json.dumps(row['game_ratios'])}|{json.dumps(row['state_ratios'])}|")
lines+=['', '## 選択と進行gate','', '事前指定した共通最大保存iterationのみ採用。途中checkpointを成績で選ばない。全seedを保持。共通prefix乱数・除外集合・初期RNGを共有するが、行動変化・対局長・model別バッチ構成によりCUDA乱数消費と軌跡は変わり、完全な共通乱数対戦とは呼ばない。学習prefix一致はtraining-audit.jsonで検証。', '', json.dumps(selection,ensure_ascii=False), '', '新validation1670000/32prefixで前回selected/gen2/旧6ms/教師100msを評価。gateはB対A point>=.45かつ固定4相手の平均point B−A>=−.05、合法性/有限/分離監査PASS。これは大規模比較への進行条件であり強化成功判定ではない。', '', json.dumps(decision,ensure_ascii=False), '', '## 対戦結果','', '主評価は同追加経験量のB対A。新holdout1680000/96prefix、先後交換。CIはprefix単位bootstrap、2席と3seedをprefix内集約した固定モデル条件付きCIで、seed母集団の不確実性は含まない。教師100msは試合cold、時間計測は生成・学習・監査と直列。ネット単体と100ms教師は異なる実費用。', '', '|phase/相手|point [95% CI]|得点差 [95% CI]|B/相手ms|', '|---|---|---|---|']
for name,v in audit['aggregate'].items():lines.append(f"|{name}|{v['point']:.4f} {v['point_ci95']}|{v['difference']:+.4f} {v['difference_ci95']}|{v['timing'][0]['mean_ms']:.3f}/{v['timing'][1]['mean_ms']:.3f}|")
lines+=['','|各条件/seed/相手|勝敗分|point [CI]|得点差 [CI]|','|---|---|---|---|']
for name,v in audit['individual'].items():
 s=v['summary'];lines.append(f"|{name}|{s['wins']}/{s['losses']}/{s['draws']}|{s['point_rate']:.4f} {s['point_rate_ci95']}|{s['mean_difference']:+.4f} {s['difference_ci95']}|")
lines+=['', '## 検証・制約','', json.dumps(audit['phases'],ensure_ascii=False), '', '過去全履歴は既存全再生監査・checkpoint/history SHA参照で保全。新規履歴は全合法性・終局報酬・指定learner側target・pool割当を再生監査。A/B双方の全訓練訪問state和集合を評価開始状態から事前除外し実訪問も後監査。初期空盤面と評価decision局面は区別。固定policy SHA変更を拒否。CUDA別process6144→6150連続6 vs6144→6146保存＋追加4で、pool循環と6148過去policy交換をまたぎmodel/past/optimizer/RNG/pool/履歴bit一致。', '', '相手別訓練pointは学習中policyへの結果で、固定相手への評価成績と混同しない。複数世代枠の各ゲーム数は少なく単一seedの偶然を一般化しない。今回の有限範囲で停止しmain推奨を変更しない。', '', '## 再現CLI','', '```bash', '.venv/bin/python -m pytest -q', '.venv/bin/python scripts/check_ppo_pool_resume.py', 'for kind in A B; do', '  for seed in 1401 1402 1403; do', '    .venv/bin/python -m blokus.ppo_scale --seed "$seed" --resume "artifacts/ppo-6144/seed-$seed/iteration-6144.pt" --output "/tmp/pool-repeat/$kind/seed-$seed" --iterations 7168 --save-steps 6400 6656 6912 7168 --cap 850 --forbidden-prefixes artifacts/ppo-generation-pool/forbidden-prefixes.json --pool-config "artifacts/ppo-generation-pool/pool-$kind.json"', '  done', 'done', '.venv/bin/python -m blokus.ppo_pool_eval audit_eval', '.venv/bin/python scripts/report_ppo_pool.py', '```', '', '既存runを上書きしない。GPU時間探索のbit再現は保証しない。研究checkpointは履歴をSHA参照し、過去履歴を複製しない。']
(R/'REPORT.md').write_text('\n'.join(lines)+'\n')
with (R/'REPORT.md').open('a') as f:
 f.write('''
## 結論

複数世代poolの改善は確認できなかった。freshのB対Aはpoint48.26% [43.83,52.69]、得点差+0.101 [-0.766,+0.929]。seed別pointは1401=51.04%、1402=38.28%、1403=55.47%。1402ではBの退行があり、validationの平均57.55%はfreshで再現しない。seed選別・checkpoint再選択・gate変更はしていない。

B対gen2はpoint55.30% [51.65,58.85]、得点差+1.622 [0.882,2.387]。B対100ms教師はpoint57.12% [52.69,61.37]、得点差+1.759 [0.809,2.641]。この固定基準への平均成績を、pool変更による改善と混同しない。教師戦はseed1402がpoint47.14%/得点差−0.510であり、seed差もある。Bの推論は約1.40ms、教師約69.33ms/手（100ms超過0%）で同時間比較ではない。B対Aでは約1.328/1.321msで、両方とも同じネット単体アーキテクチャ。

main gen2と既存推奨は維持する。今回の1/12ずつの循環配分には採用根拠がなく、無制限な延長は行わない。固定世代が弱すぎた、直近pastへの露出を減らした等は仮説であり、この実験だけで原因を特定しない。次段階でpoolを再検討するなら、年齢だけでなく凍結相手への実力差と汎化を事前に測る対照が必要。現条件を優先して保持し、細かなpool割合探索を自動で繰り返さない。

## 最終確認・記録の読み方

6runは全て追加1024iteration/16384局、計98304局で完走。生成＋更新＋保存wall3895.82秒（64.93分）、minibatch更新計226775。生成からfresh評価・監査の完了まで5296.26秒（88.27分、最終文書・push時間を除く）。CUDA別process再開smokeは別途実施し、6=2+4、pool循環/past交換込みbit一致。mainの学習条件・重みは変更していない。

validation1728局、holdout1728局のengine.is_legal（合法手列挙とは別経路）による手再生・終局得点/勝敗・prefix組/席・checkpoint SHA・時間集計・分離監査PASS。採用開始状態と評価実訪問状態の訓練和集合重複は0。追加監査で対照モデルの指定とgate算術も再確認し、保存済み成績は変更なし。全112tests PASS。最終CLI動作確認の1局は評価成績には含めない。

準備関数の流用でlock.jsonに残ったlatest(4096)/criterion_weightsは未使用の旧項目であり、今回の再開元はresume(6144)、選択規則は全条件共通最大iteration、進行gateはPLAN.md/decision.json。元の凍結lockは改変せずmetadata-notes.jsonに訂正を記録し、将来の準備関数では未使用項目を除去した。モデル・相手・評価条件の変更ではない。

過去履歴は参照のみ、今回新規分の履歴を保存。6400/6656/6912は予算打切り時の共通保存点として残し、成績による選択に使っていない。推論例:

```bash
.venv/bin/python -m blokus.terminal_play --model artifacts/ppo-generation-pool/B/seed-1401/iteration-7168.pt --human-seat 0
.venv/bin/python -m blokus.terminal_play --model artifacts/ppo-generation-pool/B/seed-1401/iteration-7168.pt --opponent artifacts/ppo-generation-pool/A/seed-1401/iteration-7168.pt
```
''')
