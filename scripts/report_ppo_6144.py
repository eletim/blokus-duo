"""Finite continuation report, including the 1401 selected-vs-resume distinction."""
import json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from blokus.value_experiment import write
R=Path('artifacts/ppo-6144');selection=json.loads((R/'selection.json').read_text());audit=json.loads((R/'evaluation-audit.json').read_text());budget=json.loads((R/'budget.json').read_text());exclude=json.loads((R/'training-exclusion-audit.json').read_text());compute=[]
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(2,3,figsize=(13,7))
keys=('entropy','approx_kl','value_mse','clip_fraction','heuristic','past')
for seed in (1401,1402,1403):
 folder=R/f'seed-{seed}';logs=json.loads((folder/'curve.json').read_text());extra=[l for l in logs if l['iteration']>4096];row=json.loads((folder/'result.json').read_text());row.update(new_games=len(extra)*16,updates=sum(l['update']['updates'] for l in extra),generation_seconds=sum(l['generation']['seconds'] for l in extra),cpu_generation_pack_seconds=sum(l['generation']['cpu_generation_pack'] for l in extra),gpu_inference_sync_seconds=sum(l['generation']['gpu_inference_sync'] for l in extra),update_seconds=sum(l['update']['seconds'] for l in extra),last64={k:float(np.mean([l['update'][k] for l in logs[-64:]])) for k in keys[:4]});compute.append(row)
 for ax,key in zip(axes.flat,keys):
  vals=[l['update'][key] if key in keys[:4] else (l['opponents'][key]+1)/2 for l in logs];ax.plot([l['iteration'] for l in logs][63:],np.convolve(vals,np.ones(64)/64,'valid'),label=str(seed));ax.axvline(4096,color='gray',ls=':');ax.set_title(key);ax.legend()
fig.tight_layout();fig.savefig(R/'learning-curves.png',dpi=140);plt.close(fig)
write(R/'compute.json',dict(seeds=compute,new_games=sum(x['new_games'] for x in compute),updates=sum(x['updates'] for x in compute),budget=budget,note='Subprocess wall includes init/history loading, generation, CUDA updates, checkpoints and I/O. CUDA sync region is not pure device kernel timing. Previous histories and hash-index bases referenced, not duplicated.'))
lines=['# PPO4096→6144 有限継続','', 'mainはgen2のまま。3seed最新4096のmodel/optimizer/past/RNGから、同じloss/終局勝敗報酬/モデル/LR .0003/16ゲーム/current .5,past .25,heuristic .25/4iterationごとのpast交換で継続。seed1401のgen2は3072であり、再開元4096とは別モデル。これは同条件の有限追加学習であり、探索やpool変更は行っていない。', '', '## 計算と学習量','', '|seed|終了iteration|選択iteration|生成更新保存wall秒|追加games|','|---|---:|---:|---:|---:|']
for row in compute:lines.append(f"|{row['seed']}|{row['iteration']}|{selection[str(row['seed'])]['iteration']}|{row['additional_seconds']:.1f}|{row['new_games']}|")
lines+=['',f"CUDA別process再開smoke込みの保守的wall合計 {budget['total_including_smoke_seconds']:.1f}秒（上限5400秒）。内訳compute.json、学習曲線learning-curves.png。更新minibatch計{sum(x['updates'] for x in compute)}、追加{sum(x['new_games'] for x in compute)}局。", '', '## 分離と選択','', '既存の全評価prefix 2434件を凍結し、その開始状態を途中に含む新生成prefixを棄却。個別ゲームのprefix乱数だけを進め、他ゲームのprefix乱数は変えない。従来のprefix列と完全同一ではないがloss/pool等は固定。旧圧縮履歴と旧全訪問indexはSHA参照、追加データだけ保存。全訓練履歴を再生して報酬符号・合法性・得点を監査。', '', json.dumps(exclude,ensure_ascii=False), '', '新validation seed1650000/32prefixでgen2 .5、旧6ms .25、教師100ms .25のpointを選択criterionにし、同値時は同重み得点差、さらに早いiteration。gen2・latest4096も候補に残す。seed1401のlatest4096直接比較は別記しcriterionに加えない。選択固定後、fresh seed1660000/96prefixでgen2と教師へ先後交換。seed1401のlatest4096補足は結果を見る前から予定し、全モデルの再選択には使わない。', '', '|seed/候補|対gen2 point/得点差|旧6ms|教師100ms|対latest4096（1401のみ）|','|---|---|---|---|---|']
for seed in (1401,1402,1403):
 for c in json.loads((R/f'seed-{seed}/candidates.json').read_text()):
  vals=[]
  for op in ('gen2','oldvalue','teacher100','latest4096'):
   p=R/f"seed-{seed}/validation-{c['label']}-{op}.json"
   if not p.exists():vals.append('gen2と同一');continue
   v=json.loads(p.read_text())['summary'];vals.append(f"{v['point_rate']:.3f}/{v['mean_difference']:+.3f}")
  lines.append(f"|{seed}/{c['label']}|"+'|'.join(vals)+'|')
lines+=['', '## 選択固定後fresh holdout','', 'CIはprefix単位bootstrap、2座席・3seedをprefix内平均する。固定した選択済み3モデルに条件づけた区間で、更新seed母集団へのCIではない。ネット単体と100ms教師は思考時間が異なる。教師のcost/refine推定を試合ごとcoldにし、生成・学習・監査と時間対戦を重ねていない。', '', '|相手|point [95% CI]|得点差 [95% CI]|選択モデル/相手 実ms|','|---|---|---|---|']
for op,v in audit['aggregate'].items():lines.append(f"|{op}|{v['point']:.4f} {v['point_ci95']}|{v['difference']:+.4f} {v['difference_ci95']}|{v['timing'][0]['mean_ms']:.3f}/{v['timing'][1]['mean_ms']:.3f}|")
lines+=['', '|seed/相手|勝/敗/分|point [95% CI]|得点差 [95% CI]|','|---|---|---|---|']
for name,v in audit['individual'].items():
 s=v['summary'];b=v['clustered'];lines.append(f"|{name}|{s['wins']}/{s['losses']}/{s['draws']}|{b['point']:.4f} {b['point_ci95']}|{b['difference']:+.4f} {b['difference_ci95']}|")
lines+=['', '## 検証','', json.dumps(audit['phases'],ensure_ascii=False), '', 'CUDA別processで4096→4102連続6iterationと4096→4098保存＋追加4を比較。4100の過去policy交換を含めmodel/past/optimizer/Python/NumPy/Torch/CUDA RNG/履歴bit一致。全テスト、訓練報酬/終局/合法性、checkpoint SHA、criterion最大値/算術、評価開始状態/実訪問状態の訓練分離を監査。初期空盤面やprefix再生途中は評価decisionと区別している。', '', '## 再現','', '```bash', 'for seed in 1401 1402 1403; do', '  .venv/bin/python -m blokus.ppo_scale --seed "$seed" --resume "artifacts/ppo-4096/seed-$seed/iteration-4096.pt" --output "/tmp/ppo6144-repeat/seed-$seed" --iterations 6144 --save-steps 5120 6144 --cap 1760 --forbidden-prefixes artifacts/ppo-6144/forbidden-prefixes.json', 'done', '.venv/bin/python scripts/check_ppo_6144_resume.py', '.venv/bin/python -m blokus.ppo_6144_eval audit_eval', f".venv/bin/python -m blokus.ppo_6144_eval match --left {selection['1401']['path']} --right teacher100 --fixture artifacts/ppo-6144/holdout-prefixes.json --output /tmp/ppo6144-teacher-repeat.json", f".venv/bin/python -m blokus.terminal_play --model {selection['1401']['path']} --opponent artifacts/ppo-4096/seed-1401/iteration-3072.pt", '```', '', '保存済みprefixを成績で再選択しない。既存run出力の上書きは拒否する。教師の時間制御探索はOS/実時間に依存するので試合結果のbit再現は保証しない。今回の有限runで停止しmain推奨を変更しない。']
(R/'REPORT.md').write_text('\n'.join(lines)+'\n')
with (R/'REPORT.md').open('a') as f:
 f.write('''
## 判断と次案

対gen2はpoint 51.48% [48.17,54.86]、得点差+0.698 [-0.042,1.425]で、今回の追加学習による安定した改善は未確認。各seedの得点差もCIが0を含む。対100ms教師は固定3モデル平均でpoint59.55% [56.16,63.02]、得点差+2.083 [1.342,2.877]。seed1403の効果が大きく、1401/1402単独ではCIが同等を含む。教師との勝利をgen2からの改善の証拠へ置き換えず、一般的最強とも呼ばない。seedの再選択なし、main gen2を維持する。

1401の5120対再開元4096は100勝84敗8分、point54.17% [47.40,60.94]、得点差−0.010 [-1.370,1.333]。対gen2(3072)のpoint50.26%とも区別し、再開元からの改善も確定できない。

今回の範囲で同条件の追加量には頭打ちの兆候がある。次案は相手poolの過去policy枠25%だけを「直近past＋凍結gen2/2048」に分散する対照実験。現行の直近past単独poolと同生成更新wallで比較し、loss・報酬・モデルを固定する。古い世代への頑健性を狙う仮説であり、この敗勝結果だけから原因を断定しない。自動延長や新学習は実施していない。

## 再起動復旧と監査経緯

初回の追加分離再生でpassの独立全配置checkerを重複実行し遅延したため、その監査だけ停止。先行して完了した全履歴合法性・報酬監査とcheckpoint/history SHAを確認後、分離用の再生だけ重複検査を省いて再開した。旧pipeline.logを保存し、学習・対戦結果は再実行していない。

complete.jsonの完了時刻とstarted.jsonの差は6872.0秒（114.5分）。生成・更新・保存の3process合計3868.25秒、再開検証36.62秒を含む保守的計算wall3904.86秒（65.08分）。計測は再起動前のRTX5090環境。休止期間は計算時間に含めない。教師100ms超過率0%、教師最大95.31ms、ネットは固定deadlineのないgreedy推論で、同時間対照ではない。

2026-11-06明示再開後、保存済み6checkpointのmodel/pastロード、optimizer/RNG存在、全15履歴参照のSHA、選択hash、CPU/CUDA forwardの有限性と出力一致（rtol/atol 1e-4）を確認。全109テストPASS（9.04秒）。元evaluation-audit.jsonをpre-reboot-evaluation-audit.jsonに保全し、2176 validation局＋1344 holdout局の再生・算術・SHA・分離監査を再実行。実対局は再実行せず、監査JSONは再起動前と一致。全訓練訪問状態との採用開始/実訪問重複0。別process CUDA再開は既存の4096→4102連続6と4096→4098＋追加4のbit一致記録を保持し、学習を繰り返さない。

復旧検証: `.venv/bin/python scripts/check_ppo_6144_recovery.py`。全テスト: `.venv/bin/python -m pytest -q`。最終報告再生成: `.venv/bin/python scripts/report_ppo_6144.py`。履歴を参照する研究checkpointは研究branchの既存成果と併せて利用する。mainへの新モデル提供は今回行わない。
''')
