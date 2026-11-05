"""Report fixed confirmation, frozen-2048 deployment and finite continuation."""
import sys,json,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from blokus.value_experiment import write
R=Path('artifacts/ppo-4096');confirm=json.loads((R/'confirmation-audit.json').read_text());audit=json.loads((R/'evaluation-audit.json').read_text());selection=json.loads((R/'selection.json').read_text());compute=[]
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axs=plt.subplots(2,2,figsize=(10,7))
for seed in (1401,1402,1403):
 d=R/f'seed-{seed}';logs=json.loads((d/'curve.json').read_text());extra=[l for l in logs if l['iteration']>2048];row=json.loads((d/'result.json').read_text())
 row.update(generation_seconds=sum(l['generation']['seconds'] for l in extra),cpu_generation_pack_seconds=sum(l['generation']['cpu_generation_pack'] for l in extra),gpu_inference_sync_seconds=sum(l['generation']['gpu_inference_sync'] for l in extra),update_seconds=sum(l['update']['seconds'] for l in extra),updates=sum(l['update']['updates'] for l in extra),new_games=len(extra)*16,last64={k:float(np.mean([l['update'][k] for l in logs[-64:]])) for k in ('entropy','approx_kl','value_mse','clip_fraction')});compute.append(row)
 for ax,k in zip(axs.flat,('entropy','approx_kl','value_mse','clip_fraction')):
  ys=np.array([l['update'][k] for l in logs]);ax.plot([l['iteration'] for l in logs][31:],np.convolve(ys,np.ones(32)/32,'valid'),label=str(seed));ax.set_title(k);ax.axvline(2048,ls=':',color='gray');ax.legend()
fig.tight_layout();fig.savefig(R/'learning-curves.png',dpi=140);plt.close(fig)
write(R/'compute.json',dict(seeds=compute,additional_wall_seconds=sum(x['additional_seconds'] for x in compute),additional_games=sum(x['new_games'] for x in compute),additional_updates=sum(x['updates'] for x in compute),scope='generation+update+loading/IO; CUDA sync regions include host work and are not pure kernel timings'))
timing={}
for op in ('previous','oldvalue'):
 rows=[json.loads((R/f'seed-{s}/holdout-{op}.json').read_text()) for s in (1401,1402,1403)];ts=[]
 for side in (0,1):
  v=[v for row in rows for v in row['timing'][side]['all_ms']];ts.append(dict(mean_ms=float(np.mean(v)),p95_ms=float(np.quantile(v,.95)),max_ms=float(max(v)),calls=len(v)))
 timing[op]=dict(wins=sum(r['summary']['wins'] for r in rows),losses=sum(r['summary']['losses'] for r in rows),draws=sum(r['summary']['draws'] for r in rows),timing=ts)
write(R/'timing-summary.json',timing)
lines=['# PPO固定2048再確認・main提供・4096学習量曲線','', '## 固定2048の再確認','', '3seed/重み/設定を固定。新seed1570000の96prefix・先後交換。モデル再選択なし。100ms教師はpriority/予算100ms/追加reserve0、試合ごとにcost/refine見積りをcold reset。以前のwarm条件を改変した再計算ではなく、新しい独立標本。工房側の同実装検証であり秘書の独立実装監査とは呼ばない。','', '|相手|point [95% CI]|得点差 [95% CI]|PPO/相手の実平均ms|','|---|---|---|---|']
for op,v in confirm['aggregate'].items():lines.append(f"|{op}|{v['point']:.3f} {v['point_ci95']}|{v['difference']:+.3f} {v['difference_ci95']}|{v['mean_ms'][0]:.3f}/{v['mean_ms'][1]:.3f}|")
lines+=['','旧6msへの平均得点差の正の再確認は不確実。100ms教師には負けており、一般的最強とは扱わない。seed1401の2048対再開元512は147勝36敗9分、+7.536 [6.135,8.938]。これはselected128との比較ではなく512への直接比較。','', '## main提供','', 'main d8b9803（推論cherry-pick 6fbd3e9）。全合法手のPPOを追加選択肢として提供し、旧低予算AIと100ms教師/既存デフォルトを維持。既定例seed1401は評価前固定。exportは各約276KiB、weights_onlyロード、履歴/optimizer/RNG不要。3seed×CPU/CUDA×26局面で元モデルとの入力/logits/value/greedy bit一致。新cloneでCPU専用2.7.1+cpuの3seed・人間先後・旧6ms/教師対戦、CUDA対戦を確認。main57tests PASS。','', '## 継続条件と選択','', '各seed2048のoptimizer/past/RNGを継承。loss、terminal勝敗報酬、学習率、16ゲーム同期生成、current/past/heuristic比率、4iterationごとのpast交換を変更せず3072/4096を保存。再確認開始状態をprefix途中に含む生成だけは棄却し、件数を記録。損失/相手pool変更とは区別する。学習履歴は旧2048と追加分をSHA付きchain参照し重複保存しない。','', '新validation seed1590000/32prefixで2048を候補に残し previous .5 + heuristic .25 + oldvalue .25 のpoint、同値なら同重み得点差・早いiterationで選択。選択後fresh seed1600000/96prefix、先後交換で対2048/旧6msを評価。訓練全訪問状態と過去評価prefix/開始状態を事前除外し、事後にも実訪問を監査。','', '|seed|終了iteration|選択iteration|追加wall秒|追加games|','|---|---:|---:|---:|---:|']
for row in compute:lines.append(f"|{row['seed']}|{row['iteration']}|{selection[str(row['seed'])]['iteration']}|{row['additional_seconds']:.1f}|{row['new_games']}|")
lines+=['','### 学習量別validation','', '|seed/checkpoint|対2048 point / 得点差|対heuristic point / 得点差|対旧6ms point / 得点差|','|---|---|---|---|']
for seed in (1401,1402,1403):
 for c in json.loads((R/f'seed-{seed}/candidates.json').read_text()):
  vals=[]
  for op in ('previous','heuristic','oldvalue'):
   v=json.loads((R/f"seed-{seed}/validation-{c['label']}-{op}.json").read_text())['summary'];vals.append(f"{v['point_rate']:.3f} / {v['mean_difference']:+.3f}")
  lines.append(f"|{seed}/{c['label']}|"+'|'.join(vals)+'|')
lines+=['','## 選択固定後holdout','', '2座席と3seedをprefix内平均し96prefix bootstrap。これは固定した3モデルに条件づけた区間で、更新seed母集団へのCIではない。ネット単体と旧6ms探索の実時間は異なる。','', '|相手|勝/敗/分|point [95% CI]|得点差 [95% CI]|','|---|---|---|---|']
for op,v in audit['aggregate'].items():
 t=timing[op];lines.append(f"|{op}|{t['wins']}/{t['losses']}/{t['draws']}|{v['point']:.3f} {v['point_ci95']}|{v['difference']:+.3f} {v['difference_ci95']}|")
lines+=['','## 検証・計算量','', '評価実訪問の訓練重複: '+json.dumps(audit['phases'],ensure_ascii=False), '', 'CUDA別process再開は2048→2054連続6iterationと2048→2050保存＋追加4。2052のpast交換を含むmodel/past/optimizer/全RNG/履歴bit一致。報酬符号・合法性・終局得点・選択最大criterion・checkpoint SHA・試合算術を監査。','', f"追加wall計{sum(x['additional_seconds'] for x in compute):.1f}秒、追加{sum(x['new_games'] for x in compute)}games、{sum(x['updates'] for x in compute)}minibatch更新。CPU生成/GPU推論同期/更新の内訳はcompute.json。曲線はlearning-curves.png。同期区間は純kernel時間ではない。", '', '## 再現','', '```bash', 'for seed in 1401 1402 1403; do', '  .venv/bin/python -m blokus.ppo_scale --seed "$seed" --resume "artifacts/ppo-learning-curve/seed-$seed/iteration-2048.pt" --output "/tmp/ppo4096-repro/seed-$seed" --iterations 4096 --save-steps 3072 4096 --cap 1800 --forbidden-prefixes artifacts/ppo-4096/confirmation-prefixes.json', 'done', '.venv/bin/python scripts/check_ppo_4096_resume.py', '.venv/bin/python -m pytest -q', '.venv/bin/python -m blokus.ppo_extend audit_eval', '.venv/bin/python -m blokus.ppo_curve_eval match --left artifacts/ppo-4096/seed-1401/iteration-3072.pt --right oldvalue --fixture artifacts/ppo-4096/holdout-prefixes.json --output /tmp/ppo4096-repeat.json', '```','', 'mainの利用者はPPO.mdのCLIを使用。新継続モデルは研究branchのみ。今回の結果で2048のmain既定seedを選び直さず、追加学習もこの有限runで終了。']
lines += ['', '## 判断と制約', '', '追加学習後のvalidation選択済みモデルは、固定2048と旧6ms探索の両方に、新holdoutで得点差・pointの改善を示した。ただしseed1401は4096ではなく3072が選択されており、全seedで更新数に単調な改善があるとはいえない。100ms教師との比較は固定2048についてのみ実施し、新選択モデルが教師を超えるかは未測定。mainには2048のみを追加し、新モデルの採用判断は秘書へ返す。', '', '主holdoutの実平均時間は新モデル/固定2048が1.444/1.263ms、新モデル/旧6ms探索が1.270/4.602ms。合法手生成と同期を含み、同時間探索比較ではない。追加生成CPU処理が依然として主費用。終盤報酬のみのvalue MSEは末尾64iterationで約0.62〜0.69、entropy約0.80〜0.89で、非有限値や一律の方策崩壊は見られなかった。', '', '次段階は今回の選択済みモデルを固定した再確認または100ms教師との比較を先に行い、追加学習量・相手poolの変更は別実験として扱う。今回の結果だけで無制限の継続学習は行わない。', '', '開始から有限pipeline（再確認・main統合・継続・評価監査）完了まで94.0分。生成＋更新は66.3分。最終文書整理・push時間はこのpipeline値に含まない。再確認開始状態の混入防止で棄却したprefix件数はtraining-audit.jsonにseed別記録。全訓練ゲームの終局報酬再生と、新評価の実訪問状態非重複を確認した。']
(R/'REPORT.md').write_text('\n'.join(lines)+'\n')
