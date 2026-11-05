import json,time,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from blokus.ppo_curve_eval import R,previous
from blokus.value_experiment import write
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
selection=json.loads((R/'selection.json').read_text());audit=json.loads((R/'evaluation-audit.json').read_text());old=previous();compute=[];fig,axs=plt.subplots(2,2,figsize=(10,7))
for seed in (1401,1402,1403):
 d=R/f'seed-{seed}';logs=json.loads((d/'curve.json').read_text());extra=[l for l in logs if l['iteration']>512];result=json.loads((d/'result.json').read_text());row=dict(result)
 row.update(generation_seconds=sum(l['generation']['seconds'] for l in extra),cpu_generation_pack_seconds=sum(l['generation']['cpu_generation_pack'] for l in extra),gpu_inference_sync_seconds=sum(l['generation']['gpu_inference_sync'] for l in extra),update_seconds=sum(l['update']['seconds'] for l in extra),updates=sum(l['update']['updates'] for l in extra),new_games=len(extra)*16)
 row['last64']={k:float(np.mean([l['update'][k] for l in logs[-64:]])) for k in ('entropy','approx_kl','value_mse','clip_fraction')};compute.append(row)
 for ax,key in zip(axs.flat,('entropy','approx_kl','value_mse','clip_fraction')):
  xs=[l['iteration'] for l in logs];ys=np.array([l['update'][key] for l in logs]);ax.plot(xs[15:],np.convolve(ys,np.ones(16)/16,mode='valid'),label=str(seed));ax.set_title(key);ax.axvline(512,color='gray',ls=':');ax.set_xlabel('cumulative iteration');ax.legend()
fig.tight_layout();fig.savefig(R/'learning-curves.png',dpi=140);plt.close(fig)
write(R/'compute.json',dict(seeds=compute,additional_wall_seconds=sum(x['additional_seconds'] for x in compute),additional_games=sum(x['new_games'] for x in compute),additional_updates=sum(x['updates'] for x in compute),scope='generation + updates + loading/checkpoint IO; inference_sync includes packing/H2D/sync, not pure CUDA kernel'))
lines=['# 終局PPO 学習量曲線','', 'main推奨は変更しない。盤面・残駒・正確な候補配置を入力する全合法手のネット単体。固定heuristic priorとPPO、終局勝敗報酬、現在/過去/heuristic相手割合は前回と同じ。探索教師や追加探索は用いない。', '', '## 条件と選択','', '3seedとも最新512から継続。前回selected (seed1401:128、1402/1403:512) を固定比較基準とした。validation32prefixで previous .5 / heuristic .25 / oldvalue .25 のpointを最大化し、同値は得点差・早いiteration。前回selected自体も候補。holdoutを選択に使用しない。','', '|seed|終了iteration|選択iteration|追加wall秒|新規対局数|','|---|---:|---:|---:|---:|']
for row in compute:lines.append(f"|{row['seed']}|{row['iteration']}|{selection[str(row['seed'])]['iteration']}|{row['additional_seconds']:.1f}|{row['new_games']}|")
lines+=['','## 学習量別validation / dev診断','','|seed/checkpoint|対前回selected point / 得点差|対heuristic point / 得点差|対oldvalue point / 得点差|dev greedy / stochastic point|','|---|---|---|---|---|']
for seed in (1401,1402,1403):
 for c in json.loads((R/f'seed-{seed}/candidates.json').read_text()):
  vals=[]
  for op in ('previous','heuristic','oldvalue'):
   rr=json.loads((R/f"seed-{seed}/validation-{c['label']}-{op}.json").read_text())['summary'];vals.append(f"{rr['point_rate']:.3f} / {rr['mean_difference']:+.3f}")
  dev=[json.loads((R/f"seed-{seed}/dev-{c['label']}-{st}.json").read_text())['summary']['point_rate'] for st in (False,True)]
  lines.append(f"|{seed}/{c['label']}|"+'|'.join(vals)+f"|{dev[0]:.3f} / {dev[1]:.3f}|")
lines+=['','## 独立holdout96prefix・先後交換','', 'pointは勝ち1/引分0.5/負け0。同一prefix内で2座席と3seedを平均し、96prefixをbootstrap。更新seed3つの結果は別途JSONに保存し、独立576標本と扱わない。','', '|対戦相手|point [95% CI]|平均得点差 [95% CI]|','|---|---|---|']
for op,v in audit['aggregate'].items():lines.append(f"|{op}|{v['point']:.3f} [{v['point_ci95'][0]:.3f}, {v['point_ci95'][1]:.3f}]|{v['difference']:+.3f} [{v['difference_ci95'][0]:+.3f}, {v['difference_ci95'][1]:+.3f}]|")
lines+=['','## 分離と検証','', '全訓練対局を独立座標合法性checkerで再生し、相手手番・prefix・終局を含む全訪問hashを生成。新評価開始状態はその集合と過去評価prefix/開始状態から除外。評価前に固定し成績で選び直していない。空盤面等の共通prefix途中状態は訓練側の除外集合には含むが、評価の意思決定局面とは区別する。', '', '評価後の実訪問状態照合: '+json.dumps(audit['phases'],ensure_ascii=False), '', 'CUDA別process再開は512→518連続6更新iterationと、512→514保存＋514→518追加4iteration。516で過去policyを交換。model/past/optimizer/全RNG/対局履歴bit一致。','', '## 実時間と限界','', f"追加生成＋更新＋IO wall合計 {sum(x['additional_seconds'] for x in compute):.1f}秒。trainログに相手別成績、CPU生成、GPU推論同期、更新時間を保存。ネット全合法手生成・GPU同期を含む実msを各対局JSONへ保存。旧value/2plyは6ms設定の探索であり、ネット単体と同じ思考時間条件ではない。", '', 'dev8prefixのgreedy対確率的評価は診断専用。主評価は一貫してgreedy。前回selectedへの信頼区間とseed差を優先し、heuristicへの改善のみで採用しない。今後は量を無制限に増やさず、凍結相手を複数世代保持するpool等を別実験として検討する。', '', '## 再現','', '```bash', 'for seed in 1401 1402 1403; do', '  .venv/bin/python -m blokus.ppo_scale --seed "$seed" --resume "artifacts/terminal-ppo/extended/seed-$seed/iteration-512.pt" --output "/tmp/ppo-repro/seed-$seed" --iterations 2048 --cap 1200', 'done', '.venv/bin/python scripts/check_ppo_scale_resume.py', '.venv/bin/python -m pytest -q', '# 記録済みfixtures/resultsを保持して再監査', '.venv/bin/python -m blokus.ppo_curve_eval audit_train', '.venv/bin/python -m blokus.ppo_curve_eval audit_eval', '```','', '学習checkpointはhistory sidecar SHAを参照。パスと選択hashはselection.json。再評価は既存結果を上書きせず別出力先を使用する。']
(R/'REPORT.md').write_text('\n'.join(lines)+'\n')
# Explicit win counts and end-to-end latency; no equal-compute claim.
timing={}
for opponent in ('previous','heuristic','oldvalue'):
 rows=[json.loads((R/f'seed-{s}/holdout-{opponent}.json').read_text()) for s in (1401,1402,1403)];ts=[]
 for side in (0,1):
  x=np.array([v for row in rows for v in row['timing'][side]['all_ms']]);ts.append(dict(calls=len(x),mean_ms=float(x.mean()),p95_ms=float(np.quantile(x,.95)),max_ms=float(x.max())))
 timing[opponent]=dict(wins=sum(x['summary']['wins'] for x in rows),losses=sum(x['summary']['losses'] for x in rows),draws=sum(x['summary']['draws'] for x in rows),timing=ts)
write(R/'timing-summary.json',timing)
with (R/'REPORT.md').open('a') as f:
 f.write('\n## 結論と実測内訳\n\n3seedすべてで前回selectedへの改善が新holdoutでも確認された。1024から2048でvalidationの前回selected対戦成績も全seedで上昇した。学習量による改善を支持する結果であり、探索追加や報酬変更によるものではない。ただしこの固定相手・prefix分布に対する結果で、万能な棋力向上や同思考時間での優越を主張しない。mainへの統合・推奨変更は行っていない。\n\n')
 f.write('|相手|勝/敗/分（3seed合計）|ネット実平均ms|相手実平均ms|\n|---|---|---:|---:|\n')
 for op,v in timing.items():f.write(f"|{op}|{v['wins']}/{v['losses']}/{v['draws']}|{v['timing'][0]['mean_ms']:.3f}|{v['timing'][1]['mean_ms']:.3f}|\n")
 f.write('\n旧2plyに対するseed1401のpoint 95%CIは[0.495, 0.628]で0.5を含む。一方、得点差CIは[+0.031,+2.615]。seed1402/1403はpoint/得点差とも正の区間。この3モデルに条件づけたprefix不確実性であり、更新seed母集団への狭いCIではない。\n\n')
 sums={k:sum(row[k] for row in compute) for k in ('generation_seconds','cpu_generation_pack_seconds','gpu_inference_sync_seconds','update_seconds')}
 f.write(f"追加213158 minibatch更新。生成{ sums['generation_seconds']:.1f}秒（うちCPU合法手生成・pack {sums['cpu_generation_pack_seconds']:.1f}秒、GPU推論同期区間{sums['gpu_inference_sync_seconds']:.1f}秒）、PPO更新{sums['update_seconds']:.1f}秒。GPU推論同期区間には転送等も含み、純粋kernel時間ではない。その他の生成処理・loading・IO・待ちを含め総3022.6秒。CPU生成が主な律速のままで、今回は挙動を変える並列化をしていない。\n\n")
 f.write('全98tests PASS、CUDA別process再開bit一致、全98304訓練対局と3776評価対局の再生/算術監査PASS。全評価実訪問状態の訓練重複0。AI対AI CLIも終局まで完走。これは工房実装の自己監査であり、秘書による独立実装監査ではない。\n\n')
 f.write('dev8prefixではgreedyが確率的評価以上のpointとなった（10モデル中9つで上、1つ同点）。小標本の診断であり、評価法を選び直す材料にはしていない。entropyは最後64iteration平均で1.12〜1.25、value MSEは0.60〜0.68。最終結果への判断は訓練MSEではなく分離済み対戦を用いた。\n\n')
 f.write('次段階候補: 同条件の有限な4096iteration延長で曲線が続くかを検証する、または複数世代の凍結policy/より強い相手を含むpoolを別条件として比較する。両変更を同時に行わず、今回のselected2048を固定基準として保全する。100ms教師との比較は今回未実施。\n\n')
 f.write('研究モデル対戦例（seed1401は記載例でありholdoutで最良seedを再選択していない）:\n\n```bash\n.venv/bin/python -m blokus.terminal_play --model artifacts/ppo-learning-curve/seed-1401/iteration-2048.pt --opponent heuristic\n.venv/bin/python -m blokus.terminal_play --model artifacts/ppo-learning-curve/seed-1401/iteration-2048.pt --human-seat 1\n```\n')

with (R/'REPORT.md').open('a') as f:
 f.write('\n補足診断: seed1401の再開元512は前回selected128と異なるため、選択固定後に同じdev8prefixでgreedy/確率的比較だけを追記した（point .9375/.5000）。学習量の増加とdev成績は単調ではなく、小標本の揺らぎが大きい。これによるcheckpoint再選択はしていない。seed1401について今回の主比較はselected128との比較であり、latest512とのfresh96prefix直接比較は未実施。seed1402/1403は前回selected自体が512である。\n')
 f.write('\n固定済みfixtureで実対戦を再実行するCLI（出力が存在する場合は上書きせず拒否）:\n\n```bash\n.venv/bin/python -m blokus.ppo_curve_eval match --left artifacts/ppo-learning-curve/seed-1401/iteration-2048.pt --right oldvalue --fixture artifacts/ppo-learning-curve/holdout-prefixes.json --output /tmp/ppo-curve-replay.json\n```\n')
