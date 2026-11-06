# 1回に学習する局面16→64: 有限対照結果

64局面更新は収集込みで約1.42倍高速だった。ただし公開gen2への改善効率の優位は未確認。固定5120のvalidationではB64対gen2 point53.65%、A16は54.04%。B対Aは52.08%で95% CIが50%をまたぐ。事前gateはBのgen2 pointがA以上という条件も要求し、0.39ポイント差で未達。明確な劣化とは解釈しないが、gateを後から緩めずfresh96prefix holdoutは実施しない。main/public gen2の推奨を維持。

## 条件と費用

12809f3からresearch/ppo-minibatch-64。3seed1751401/1751402/1751403の最新4096からmodel/past/optimizer/全RNGを同一で分岐。前回selected2048を混ぜていない。A16/B64以外の条件は4ch/float32/native/収集16games/Adam3e-4/2epochs/terminal WDL/既存loss/pool current50 past25 heuristic25/past4iter交換で固定。公開gen2は6ch、seed1401は3072iter、他2seedは4096iterのネット単体。両条件とgen2に探索なし。

追加6run各1024iter=16384games、計98304games。生成+更新+保存実測合計1113.456秒（18.56分）、runner全体1114.242秒。上限3600秒内。終了点5120、途中4608を成績を見る前に固定し、全seed/両条件を保持。再開元4096は各65536games、今回終点は各81920games累積。

| 条件 | 同量games | 生成+更新+保存s | games/s | learner states | optimizer steps | 生成s | update s |
|---|---:|---:|---:|---:|---:|---:|---:|
| 16 | 49152 | 653.99 | 75.16 | 1162007 | 122700 | 289.50 | 358.63 |
| 64 | 49152 | 459.47 | 106.98 | 1165918 | 38718 | 290.16 | 163.38 |

states/s・steps/s・CPU pack/GPU inference+sync時間はlearning-curves.json。収集時間はほぼ不変。行動が変わるため同ゲーム数でもlearner statesは少し異なる。対局割合はcurrent50/past25/heuristic25、学習state割合は以下。current対局は両側learner、それ以外はlearner側のみ更新。

- 16: current 66.329%, past 16.574%, heuristic 17.097%
- 64: current 66.304%, past 16.570%, heuristic 17.125%

外生prefixは条件間49152組すべて一致。モデル更新以降の行動・未来の訓練軌跡は異なる。等価実装/等optimizer更新量とは呼ばない。

## 固定rollout GPU測定

同一4096重み/optimizerと保存376states、各条件2warmup/6交互反復、初期化は時間外。compileなし/AMPなし。CUDA eventはhost dispatch gapも含むspan。純kernel時間とは呼ばない。

| minibatch | update中央値ms | CUDA span ms | peak MB | padding/実候補 | step数 |
|---|---:|---:|---:|---:|---:|
| 16 | 113.636 | 113.610 | 169.64 | 3.600 | 37 |
| 64 | 50.849 | 50.825 | 310.28 | 4.070 | 12 |

更新単体約2.23倍は37→12stepの違いを含む。16はearlyKLで実state visits584、64は2epochsを完了し752。rollout states/sとoptimizer state visits/sをbenchmark.jsonで区別。ランダムに混ざった序盤/終盤候補をバッチ最大数へ揃えるpaddingが64で増加した。profileでは小さいAdam/同期呼出しが減る一方embedding backwardは減らない。詳細PROFILE-NOTES.md。torch.compile/shape bucketは今回未採用・未検証。前runのCUDA graph capture失敗をcompile全般の不可能性とは扱わない。

## Validation: 64prefix ×3seed×両座席

新prefix seed17641000。各比較384対局、同prefix内6対局をまとめbootstrap4000。game単位の独立観測とは扱わない。以下はvalidationであり最終独立holdoutの強さ保証ではない。checkpointは5120固定、4608へ戻してgateを回避しない。

| iter | 比較 | point % [95% CI] | 得点差 [95% CI] |
|---|---|---|---|
| 4608 | B-A | 50.13 [45.96, 54.43] | -0.333 [-1.260, +0.617] |
| 4608 | A-gen2 | 50.26 [44.92, 55.86] | +0.755 [-0.242, +1.784] |
| 4608 | B-gen2 | 56.12 [51.30, 60.94] | +0.612 [-0.510, +1.724] |
| 4608 | A-base | 53.78 [49.74, 58.07] | +1.372 [+0.490, +2.286] |
| 4608 | B-base | 53.39 [48.96, 57.42] | +1.078 [+0.130, +2.068] |
| 5120 | B-A | 52.08 [48.05, 56.25] | +0.682 [-0.354, +1.716] |
| 5120 | A-gen2 | 54.04 [50.13, 57.94] | +1.227 [+0.482, +1.935] |
| 5120 | B-gen2 | 53.65 [48.83, 57.94] | +1.122 [+0.078, +2.050] |
| 5120 | A-base | 58.07 [53.65, 62.76] | +2.013 [+1.049, +3.000] |
| 5120 | B-base | 53.12 [48.83, 57.68] | +1.495 [+0.541, +2.536] |

終点のseed別point（selection.jsonに得点差/CIも保存）:

- B-A: 1751401: 47.27%, 1751402: 53.91%, 1751403: 55.08%
- A-gen2: 1751401: 53.91%, 1751402: 59.77%, 1751403: 48.44%
- B-gen2: 1751401: 53.52%, 1751402: 53.52%, 1751403: 53.91%
- A-base: 1751401: 53.91%, 1751402: 64.06%, 1751403: 56.25%
- B-base: 1751401: 55.08%, 1751402: 57.42%, 1751403: 46.88%

## 観測済みwallに対する曲線

等wallの未観測棋力は補間・外挿しない。各seedの累積追加wallとgen2 pointを対応させた実測点のみ。

| seed | batch | 4608時 累積追加s / point | 5120時 累積追加s / point |
|---|---:|---|---|
| 1751401 | 16 | 113.12 / 52.34% | 221.63 / 53.91% |
| 1751401 | 64 | 76.93 / 54.69% | 154.29 / 53.52% |
| 1751402 | 16 | 107.94 / 46.88% | 216.92 / 59.77% |
| 1751402 | 64 | 76.57 / 62.89% | 151.89 / 53.52% |
| 1751403 | 16 | 107.89 / 51.56% | 215.44 / 48.44% |
| 1751403 | 64 | 76.10 / 50.78% | 153.28 / 53.91% |

## 健全性・監査

- 全テスト82 PASS。CPU/CUDA minibatch64の有限勾配/方策更新、default16の既存基準bit一致を確認。両batchのCUDA別processで4096+8 vs4096+4+再開4がmodel/past/optimizer/Python/NumPy/Torch/CUDA RNGまでbit一致。past交換4100/4104を含む。保存minibatchの暗黙変更を拒否。workerは使わずiteration境界保存。
- 初回driverの決定的演算設定漏れでresume重み不一致を発見。本実験前に修正して再検証済み。失敗ログ/履歴を保全し状態除外にも含めた。RNG一致だけでは再開を合格にしなかった。
- 新学習98304gamesを全合法性/最終得点/return視点/learner側/状態hashまで再生、action indexは256state間隔で別列挙照合。旧indexはSHA参照、新deltaのみ保存。統合22,854,366訪問状態から評価開始を除外。過去評価開始3174件を新訓練prefixから除外。
- validation計3840gamesを再生・得点/勝敗/CI/hash/座席対/選択規則を監査。開始・実訪問状態の訓練重複0。prefix hash 0328eae4d8344eccf43045b14cdcba5074c4f881e5eaac42e4017ace519af366。空盤面は評価開始に使わない。
- 学習と計測対戦は重ねていない。各matchのleft_ms/right_msは合法手生成/転送/同期/推論を含む。初期ロードwarmupは除外。CPU/CUDAの推論用export一致・CLI結果はexport-audit.json/cli-*.json。

## 判断と次の仮説

速度面ではB64は有効だが、gen2へより効率よく改善するかは未確認。A16対gen2もvalidation上の正の結果であり、独立最終評価なしに公開推奨を変更しない。Bが明確に劣ると断定せず、gate未達の僅差を正直に残す。
Bの終盤平均KLは約.0011–.0012、Aは約.005。earlyKL停止はB約0.2–0.4%、A約62–70%。少ないoptimizer stepで更新が小さくなっている仮説が具体的な次候補。次はB64でLRのみを事前固定した小範囲（例3e-4対6e-4）で対照し、A16を残してKL/entropyとgen2成績を検証する余地がある。ただし本結果からLR増加の成功は未検証。今回は実行せず返す。等価な高速化を優先するならpaddingを処理しない候補表現/gather backwardの削減を別経路で検証し、学習条件変更と混ぜない。

## 再現

READMEのblokus.ppo_minibatch CLIは保存4096から別outputへ再現可能。全6runの原設定はscripts/run_ppo_minibatch.py、事前基準PLAN.md、profile scripts/bench_ppo_minibatch.py、再開 scripts/check_ppo_minibatch.py。既存結果がある出力へ履歴を上書きしない。
監査: `python -m blokus.ppo_minibatch_audit` → `python -m blokus.ppo_minibatch_eval audit` → `python -m scripts.audit_ppo_minibatch_results`。旧index cacheは前成果training-audit.jsonのSHAと一致する必要がある。新checkoutでは前成果の状態index再構築が必要（過去履歴を本branchへ再複製していない）。
推論例: `python -m blokus.ppo_play --checkpoint artifacts/ppo-minibatch/batch-64-seed-1751401/inference-5120.pt --device cpu --backend python --opponent heuristic`。これは研究モデルであり既定公開モデルではない。
