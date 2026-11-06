import json,statistics
from pathlib import Path
from blokus.ppo_lr64 import R,SEEDS
load=lambda name:json.loads((R/name).read_text())
c=load('training-complete.json');selection=load('selection.json');rows=load('learning-curves.json')['rows'];audit=load('training-audit.json');final=selection['curves'][str(selection['fixed_iteration'])];hold=load('holdout-summary.json') if (R/'holdout-summary.json').exists() else None
lines=['# 公開gen2に対する4ch PPO / LR対照','']
if hold:
 d=hold['C-gen2'];claim='この固定モデル・prefix分布でpointの優位を確認' if d['point_ci95'][0]>.5 else '公開gen2へのpoint優位は未確認'
 lines.append(f"独立96prefixでC/gen2 point {100*d['point']:.2f}% [95% CI {100*d['point_ci95'][0]:.2f}, {100*d['point_ci95'][1]:.2f}]。{claim}。main変更なし。")
else:lines.append('新validationではC/gen2 point55.47% [95% CI51.43,59.38]、得点差+1.448。一方C/B point48.96% [43.75,54.04]で、事前のCI下限45%超という非劣性gateが未達。新holdoutは実施せず、validationだけで公開gen2への強化成功としない。旧gate/新gateとも変更せず、mainを維持。')
lines+=['','## 固定条件','', 'eedfa345基点。3seed1751401/2/3の最新4096（model/past/optimizer/Python/NumPy/Torch/CUDA RNG）からA16 LR3e-4、B64 LR3e-4、C64 LR6e-4へ分岐。旧selected2048ではない。新外生seed1771401/2/3、全条件で同じprefix列。初期16gamesは行動・targetまで一致、その後は方策/更新によって軌跡が分かれる。旧A/B対照を流用していない。', '収集16games/iter、epochs2、float32、Adam、loss/clip/報酬terminal WDL/heuristic prior/current50 past25 heuristic25/past4iter交換/native/常駐updateを固定。Cのみoptimizer.param_groups.lrを6e-4へ明示上書き、再開時にmetadataと実値一致を要求。等ゲーム数であり等optimizer stepsではない。新padding/compile/AMP実装なし。', '各条件各seed追加1024iter=16384games、終点5120固定、中間4608。基点累積65536games、終点81920games。対照gen2は公開6chネット、対応seed1401は3072iter=49152games、他2seed4096iter=65536games。全評価はgreedyネット単体、探索なし。公開gen2の学習量/入力条件は同一ではない。', '', '## 実費用と観測曲線','',f"9run計{sum(x['games'] for x in c['results']):,}games。生成+更新+保存合計{c['seconds']:.3f}秒（{c['seconds']/60:.2f}分）、runner全体{c['wall']:.3f}秒。3600秒上限内。実装/監査/評価時間は別。",'', '|条件|games|wall s|games/s|learner states|optimizer steps|生成s|update s|','|---|---:|---:|---:|---:|---:|---:|---:|']
for cond in 'ABC':
 x=[r for r in rows if r['condition']==cond];total=lambda k:sum(r[k] for r in x);lines.append(f"|{cond}|{total('games')}|{total('wall'):.2f}|{total('games')/total('wall'):.2f}|{total('states')}|{total('steps')}|{total('generation_seconds'):.2f}|{total('update_seconds'):.2f}|")
lines+=['','games/s、learner states/s、steps/s、CPU packとGPU推論/同期、相手別結果はlearning-curves.json。wall差はゲーム軌跡差・KL停止・step数差も含む。速度だけで同一algorithmの最適化としない。','', '|seed|条件|終盤entropy|KL|clip fraction|value MSE|区間earlyKL率|','|---|---|---:|---:|---:|---:|---:|']
for x in rows:
 if x['iteration']==5120:lines.append(f"|{x['seed']}|{x['condition']}|{x['entropy']:.4f}|{x['kl']:.5f}|{x['clip_fraction']:.4f}|{x['value_mse']:.4f}|{x['early_kl_fraction']:.3%}|")
lines+=['','対局割合は各条件current50/past25/heuristic25。learner state割合:','']
for cond in 'ABC':
 counts={k:0 for k in ('current','past','heuristic')}
 for x in audit['rows']:
  if x['main'] and '/'+cond+'-seed-' in x['path']:
   for k,v in x['learner_states'].items():counts[k]+=v
 lines.append('- '+cond+': '+', '.join(f'{k} {v/sum(counts.values()):.3%}' for k,v in counts.items()))
lines+=['','## Validationと事前gate','', '新64prefix(seed17741000)、各比較384games。prefix単位に3seed×両座席をまとめbootstrap4000。CIは固定3モデルに対するprefix不確実性で、全更新seed母集団のCIではない。C/gen2 point>.5かつ得点差>=0、C/B point CI下限>.45、健全性/分離監査合格でholdoutへ進む。この5pp許容は事前の探索的非劣性基準であり成功判定ではない。旧gateは変更していない。終点5120固定で4608へ選び直さない。','', '|iter|比較|point % [95% CI]|得点差 [95% CI]|','|---|---|---|---|']
for step,ds in selection['curves'].items():
 for label,d in ds.items():lines.append(f"|{step}|{label}|{100*d['point']:.2f} [{100*d['point_ci95'][0]:.2f}, {100*d['point_ci95'][1]:.2f}]|{d['difference']:+.3f} [{d['difference_ci95'][0]:+.3f}, {d['difference_ci95'][1]:+.3f}]|")
lines.append(f"\n事前gate: {selection['gate']}。全seed/条件の保存checkpointを固定、holdoutで選び直していない。")
if hold:
 lines+=['','## Fresh holdout','', 'seed17761000、96prefix、各比較576games。計測と学習/監査は分離。','', '|比較|point % [95% CI]|得点差 [95% CI]|','|---|---|---|']
 for label,d in hold.items():lines.append(f"|{label}|{100*d['point']:.2f} [{100*d['point_ci95'][0]:.2f}, {100*d['point_ci95'][1]:.2f}]|{d['difference']:+.3f} [{d['difference_ci95'][0]:+.3f}, {d['difference_ci95'][1]:+.3f}]|")
for label,d in (hold or final).items():
 lines.append('\n'+label+' seed別: '+', '.join(f"{seed}: point {100*v['point_rate']:.2f}% / 得点差 {v['mean_score_difference'] if 'mean_score_difference' in v else v.get('mean_difference',v.get('score_difference','see selection JSON'))}" for seed,v in d['by_seed'].items()))
lines+=['','## 実wallの観測点','', '同wallで未実施の強さは補間・推定しない。以下は各seedの追加累積wallと同時点gen2 pointの観測点。','', '|seed|条件|4608 s / point|5120 s / point|','|---|---|---|---|']
for seed in SEEDS:
 for cond in 'ABC':
  x=[r for r in rows if r['seed']==seed and r['condition']==cond];p=[selection['curves'][str(step)][cond+'-gen2']['by_seed'][str(seed)]['point_rate']*100 for step in (4608,5120)];lines.append(f"|{seed}|{cond}|{x[0]['wall']:.2f} / {p[0]:.2f}%|{sum(r['wall'] for r in x):.2f} / {p[1]:.2f}%|")
lines+=['','## 正しさ・分離','', '全条件のCUDA別process再開8 vs4+4でmodel/past/optimizer/Python/NumPy/Torch/CUDA RNG bit一致、past交換4100/4104を含む。condition/LR/minibatch/系列seedの暗黙変更を拒否。決定的演算/CPU1threadをrestore前に設定し、その後RNGはcheckpointから復元。workerなし、iteration境界保存。崩壊monitorはcheckpoint境界をまたぐ。', f"新学習全{sum(x['games'] for x in audit['rows'] if x['main']):,}gamesを合法性/得点/return視点/learner側/状態hash再生、action index256state間隔照合。全条件間外生prefix {audit['shared_external_prefixes']:,}組一致。過去評価開始3238件を新訓練prefixから除外。旧状態indexはSHA参照、統合{audit['union_states']:,}訪問状態から新評価開始を除外。新状態deltaのみ保存し過去巨大履歴は複製しない。", '全87テストPASS、9モデルのCPU/CUDA各デバイス内でtraining checkpointと推論exportのlogits/value bit一致。C seed1751401のCPU/pythonとCUDA/native CLIは両方完走（得点-16/-20、35手）。CPU-vs-CUDA一般のbit一致とは主張しない。validationネット単体の実時間は約.519–.530ms/手（合法手生成/転送/同期/推論込み）。', '評価後実訪問状態/対局再生/得点符号/座席対/clusterCI/hash/gate算術はevaluation-audit.json/arithmetic-audit.json。tests.log、export-audit.json、CPU/CUDA CLIを参照。初期空盤面を評価開始として使わない。実msは各matchのleft_ms/right_msに合法手生成/転送/同期/推論込みで保存、初期ロードwarmupは時間外。','', '## 判断・次案とGPU最適化','', 'LR倍増でBの終盤KL約.0012からC約.0032–.0040へ増えたが、C/Bの得点差は+0.005、point48.96%でLR増による改善は未確認。B/gen2もvalidation55.34%とCに近い。Cの明確な劣化を示すCIではなく、5pp非劣性を確認できる精度が足りないというgate未達。seed1751402のC/gen2は46.88%で、全seedが同じように改善したわけでもない。', '追加LR候補を増やさず、次段階ではB/Cの設定・重みを凍結した公開gen2への独立再確認を先に設計する案がある。C/Bの非劣性まで結論するなら、今回のprefix間ばらつきを使って必要精度と標本数を事前設定する。これは今のgateを救済したり旧データをholdoutへ転用する提案ではなく、新しい有限確認の計画。今回その追加評価は行わない。', 'NEXT-GPU.mdに実候補Ncへのflatten/segment処理と配置係数によるembedding backward集約案を具体化。どちらも未実装・未検証で、最適とは主張しない。LR対照へ混ぜていない。', 'READMEのblokus.ppo_lr64 CLIで最新4096から別outputへ再現できる。全9run設定scripts/run_ppo_lr64.py、再開scripts/check_ppo_lr64.py、集計scripts/summarize_ppo_lr64.py。既存出力を上書きしない。監査はpython -m blokus.ppo_lr64_audit、python -m blokus.ppo_lr64_eval audit、python -m scripts.audit_ppo_lr64_results。旧index cacheは前成果SHAと照合する（cache未作成環境では前成果のindex再構築が必要）。','']
(R/'REPORT.md').write_text('\n'.join(lines))
