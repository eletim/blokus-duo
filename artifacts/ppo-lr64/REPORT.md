# 公開gen2に対する4ch PPO / LR対照

新validationではC/gen2 point55.47% [95% CI51.43,59.38]、得点差+1.448。一方C/B point48.96% [43.75,54.04]で、事前のCI下限45%超という非劣性gateが未達。新holdoutは実施せず、validationだけで公開gen2への強化成功としない。旧gate/新gateとも変更せず、mainを維持。

## 固定条件

eedfa345基点。3seed1751401/2/3の最新4096（model/past/optimizer/Python/NumPy/Torch/CUDA RNG）からA16 LR3e-4、B64 LR3e-4、C64 LR6e-4へ分岐。旧selected2048ではない。新外生seed1771401/2/3、全条件で同じprefix列。初期16gamesは行動・targetまで一致、その後は方策/更新によって軌跡が分かれる。旧A/B対照を流用していない。
収集16games/iter、epochs2、float32、Adam、loss/clip/報酬terminal WDL/heuristic prior/current50 past25 heuristic25/past4iter交換/native/常駐updateを固定。Cのみoptimizer.param_groups.lrを6e-4へ明示上書き、再開時にmetadataと実値一致を要求。等ゲーム数であり等optimizer stepsではない。新padding/compile/AMP実装なし。
各条件各seed追加1024iter=16384games、終点5120固定、中間4608。基点累積65536games、終点81920games。対照gen2は公開6chネット、対応seed1401は3072iter=49152games、他2seed4096iter=65536games。全評価はgreedyネット単体、探索なし。公開gen2の学習量/入力条件は同一ではない。

## 実費用と観測曲線

9run計147,456games。生成+更新+保存合計1560.262秒（26.00分）、runner全体1561.171秒。3600秒上限内。実装/監査/評価時間は別。

|条件|games|wall s|games/s|learner states|optimizer steps|生成s|update s|
|---|---:|---:|---:|---:|---:|---:|---:|
|A|49152|652.96|75.28|1161419|122239|288.44|358.56|
|B|49152|453.98|108.27|1161684|38407|287.20|160.80|
|C|49152|453.33|108.43|1157789|38128|286.69|160.70|

games/s、learner states/s、steps/s、CPU packとGPU推論/同期、相手別結果はlearning-curves.json。wall差はゲーム軌跡差・KL停止・step数差も含む。速度だけで同一algorithmの最適化としない。

|seed|条件|終盤entropy|KL|clip fraction|value MSE|区間earlyKL率|
|---|---|---:|---:|---:|---:|---:|
|1751401|A|0.9340|0.00565|0.0590|0.6400|68.359%|
|1751401|B|0.8941|0.00123|0.0121|0.6653|0.391%|
|1751401|C|0.8855|0.00365|0.0413|0.6599|4.102%|
|1751402|A|0.9366|0.00516|0.0552|0.5985|74.023%|
|1751402|B|0.8947|0.00127|0.0129|0.6215|0.391%|
|1751402|C|1.0508|0.00323|0.0355|0.6482|2.344%|
|1751403|A|0.9199|0.00549|0.0566|0.6437|80.859%|
|1751403|B|1.0287|0.00109|0.0101|0.6369|0.391%|
|1751403|C|0.9259|0.00399|0.0407|0.6489|4.883%|

対局割合は各条件current50/past25/heuristic25。learner state割合:

- A: current 66.273%, past 16.540%, heuristic 17.187%
- B: current 66.277%, past 16.552%, heuristic 17.171%
- C: current 66.273%, past 16.565%, heuristic 17.162%

## Validationと事前gate

新64prefix(seed17741000)、各比較384games。prefix単位に3seed×両座席をまとめbootstrap4000。CIは固定3モデルに対するprefix不確実性で、全更新seed母集団のCIではない。C/gen2 point>.5かつ得点差>=0、C/B point CI下限>.45、健全性/分離監査合格でholdoutへ進む。この5pp許容は事前の探索的非劣性基準であり成功判定ではない。旧gateは変更していない。終点5120固定で4608へ選び直さない。

|iter|比較|point % [95% CI]|得点差 [95% CI]|
|---|---|---|---|
|4608|C-B|45.44 [40.49, 50.39]|-0.638 [-1.630, +0.372]|
|4608|C-A|52.73 [47.79, 57.94]|+0.250 [-0.773, +1.237]|
|4608|A-gen2|49.35 [44.79, 53.91]|-0.081 [-1.185, +1.034]|
|4608|B-gen2|57.16 [52.34, 62.11]|+2.016 [+1.042, +2.925]|
|4608|C-gen2|53.65 [49.09, 58.33]|+0.443 [-0.542, +1.445]|
|5120|C-B|48.96 [43.75, 54.04]|+0.005 [-0.982, +1.018]|
|5120|C-A|54.04 [49.61, 58.33]|+1.138 [+0.193, +2.109]|
|5120|A-gen2|52.47 [48.31, 56.64]|-0.419 [-1.297, +0.406]|
|5120|B-gen2|55.34 [50.39, 60.29]|+1.120 [+0.193, +2.005]|
|5120|C-gen2|55.47 [51.43, 59.38]|+1.448 [+0.476, +2.386]|

事前gate: False。全seed/条件の保存checkpointを固定、holdoutで選び直していない。

C-B seed別: 1751401: point 48.44% / 得点差 0.2734375, 1751402: point 47.66% / 得点差 -0.4453125, 1751403: point 50.78% / 得点差 0.1875

C-A seed別: 1751401: point 53.91% / 得点差 0.484375, 1751402: point 47.66% / 得点差 0.1015625, 1751403: point 60.55% / 得点差 2.828125

A-gen2 seed別: 1751401: point 58.20% / 得点差 0.8359375, 1751402: point 50.39% / 得点差 -0.84375, 1751403: point 48.83% / 得点差 -1.25

B-gen2 seed別: 1751401: point 61.72% / 得点差 2.4140625, 1751402: point 50.00% / 得点差 0.0390625, 1751403: point 54.30% / 得点差 0.90625

C-gen2 seed別: 1751401: point 57.81% / 得点差 2.15625, 1751402: point 46.88% / 得点差 -0.046875, 1751403: point 61.72% / 得点差 2.234375

## 実wallの観測点

同wallで未実施の強さは補間・推定しない。以下は各seedの追加累積wallと同時点gen2 pointの観測点。

|seed|条件|4608 s / point|5120 s / point|
|---|---|---|---|
|1751401|A|113.60 / 46.48%|223.66 / 58.20%|
|1751401|B|75.93 / 53.52%|151.97 / 61.72%|
|1751401|C|75.76 / 53.12%|152.05 / 57.81%|
|1751402|A|110.97 / 54.69%|219.58 / 50.39%|
|1751402|B|76.38 / 57.03%|151.51 / 50.00%|
|1751402|C|75.63 / 48.05%|151.09 / 46.88%|
|1751403|A|106.31 / 46.88%|209.72 / 48.83%|
|1751403|B|75.91 / 60.94%|150.49 / 54.30%|
|1751403|C|74.98 / 59.77%|150.19 / 61.72%|

## 正しさ・分離

全条件のCUDA別process再開8 vs4+4でmodel/past/optimizer/Python/NumPy/Torch/CUDA RNG bit一致、past交換4100/4104を含む。condition/LR/minibatch/系列seedの暗黙変更を拒否。決定的演算/CPU1threadをrestore前に設定し、その後RNGはcheckpointから復元。workerなし、iteration境界保存。崩壊monitorはcheckpoint境界をまたぐ。
新学習全147,456gamesを合法性/得点/return視点/learner側/状態hash再生、action index256state間隔照合。全条件間外生prefix 49,152組一致。過去評価開始3238件を新訓練prefixから除外。旧状態indexはSHA参照、統合27,533,566訪問状態から新評価開始を除外。新状態deltaのみ保存し過去巨大履歴は複製しない。
全87テストPASS、9モデルのCPU/CUDA各デバイス内でtraining checkpointと推論exportのlogits/value bit一致。C seed1751401のCPU/pythonとCUDA/native CLIは両方完走（得点-16/-20、35手）。CPU-vs-CUDA一般のbit一致とは主張しない。validationネット単体の実時間は約.519–.530ms/手（合法手生成/転送/同期/推論込み）。
評価後実訪問状態/対局再生/得点符号/座席対/clusterCI/hash/gate算術はevaluation-audit.json/arithmetic-audit.json。tests.log、export-audit.json、CPU/CUDA CLIを参照。初期空盤面を評価開始として使わない。実msは各matchのleft_ms/right_msに合法手生成/転送/同期/推論込みで保存、初期ロードwarmupは時間外。

## 判断・次案とGPU最適化

LR倍増でBの終盤KL約.0012からC約.0032–.0040へ増えたが、C/Bの得点差は+0.005、point48.96%でLR増による改善は未確認。B/gen2もvalidation55.34%とCに近い。Cの明確な劣化を示すCIではなく、5pp非劣性を確認できる精度が足りないというgate未達。seed1751402のC/gen2は46.88%で、全seedが同じように改善したわけでもない。
追加LR候補を増やさず、次段階ではB/Cの設定・重みを凍結した公開gen2への独立再確認を先に設計する案がある。C/Bの非劣性まで結論するなら、今回のprefix間ばらつきを使って必要精度と標本数を事前設定する。これは今のgateを救済したり旧データをholdoutへ転用する提案ではなく、新しい有限確認の計画。今回その追加評価は行わない。
NEXT-GPU.mdに実候補Ncへのflatten/segment処理と配置係数によるembedding backward集約案を具体化。どちらも未実装・未検証で、最適とは主張しない。LR対照へ混ぜていない。
READMEのblokus.ppo_lr64 CLIで最新4096から別outputへ再現できる。全9run設定scripts/run_ppo_lr64.py、再開scripts/check_ppo_lr64.py、集計scripts/summarize_ppo_lr64.py。既存出力を上書きしない。監査はpython -m blokus.ppo_lr64_audit、python -m blokus.ppo_lr64_eval audit、python -m scripts.audit_ppo_lr64_results。旧index cacheは前成果SHAと照合する（cache未作成環境では前成果のindex再構築が必要）。
