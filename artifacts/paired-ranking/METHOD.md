# 同一局面の候補差分学習：設計と再現

基点1b5cf87、既存エンジンは変更しない。正式子依頼work 0cd6f9db-532d-48a9-befc-5854a0cc0af2。

新規seed200000〜201199の1200対局から、残駒合計24以下で合法手4つ以上の局面を各2つ抽出。heuristic上位8手から4候補を抽出し、相手のheuristic最大応手を1手進めた2ply後の葉を作る。それぞれ同一初期乱数seedで2回、非学習探索/A2探索を継続方策として終局まで進め、rootプレイヤーから見た得点差を記録する。継続には5%ランダム行動を含む。候補間の共通乱数は開始seedの共有であり、状態依存の乱数消費が分岐しても以後の乱数事象が完全に対応する保証はない。

2400局面×4候補×2継続。学習ラベルは2回の終局得点差の平均/40。group内全6候補対の差分にsmoothL1、温度0.15のsoft順位BCE、弱い絶対得点smoothL1を使う。総損失=差分+.1×順位+.25×絶対値。勝った候補だけを複製する模倣ではなく、全候補の実ゲーム結果を利用する。最適対戦価値を証明するものではない。

既存pooled OutcomeNet seed201から全3seedを初期化。学習seed301/302/303はminibatchと対称変換の乱数を変える。AdamW lr0.0002、64group/batch、6000更新/seed、各最大180秒。元対局IDの末尾0を予測診断に回し、同じ対局のgroupをtrain/診断へ分割しない。group内では盤面対称変換を共通にかける。予測診断のpair MSE最小checkpointを保存し、新validation seed210000の32prefixでalpha .25/.5/1を比較して固定。最終holdout seed220000の96prefixは選択後に生成する。

非学習探索と現value(seed201,alpha.5,gate24)が対照。全方式同じincremental 2ply探索、相手応手幅3、最大root128、合法手生成と順位付けとGPU同期を含む6ms予算。遅れて終わった子は破棄し、全decision実時間・超過・node数を記録。強制停止できないnative演算があるので厳密なリアルタイム上限ではない。実消費時間をpaddingで一致させた比較ではない。壁時計で枝数が変わるため対戦ログの完全再現は保証しないが、学習は保存再開のbit一致を検証する。

各prefixで先後入替。相手を替えても同じprefix集合。prefix SHA256とcheckpoint SHA256を保存。prefix内2局を平均したbootstrap、さらに3seed×prefixを交差再標本化した区間を記録。共通prefix drawをseed間で共有し、同じ開始状態を別の独立観測として数えない。3seedと多数比較の限界は残る。

```bash
# 新しい出力先で一連の実験
.venv/bin/python -m blokus.ranking_experiment run --output artifacts/my-ranking
# 保存済みデータで単独学習・再開（stepsは追加更新数）
.venv/bin/python -m blokus.ranking_experiment train --data artifacts/paired-ranking/groups.npz --output artifacts/my-rank-model --seed 301 --steps 6000 --seconds 180
.venv/bin/python -m blokus.ranking_experiment train --data artifacts/paired-ranking/groups.npz --output artifacts/my-rank-model --seed 301 --steps 1000 --seconds 180 --resume artifacts/my-rank-model/latest.pt
# 固定spec/prefixで比較を再実行（時間制御のため対戦結果は変動しうる）
.venv/bin/python -m blokus.value_experiment tournament --prefix-file artifacts/paired-ranking/holdout-prefixes.json --specs artifacts/paired-ranking/play-specs.json --output artifacts/ranking-replay.json --budget-ms 6
.venv/bin/python scripts/audit_ranking.py artifacts/paired-ranking
.venv/bin/python scripts/check_ranking_resume.py
.venv/bin/python -m pytest -q
```

configとcheckpointにsource hash/torch/seed/データhash/初期checkpoint hashを保存。latestはoptimizer/RNG付き再開用、predictive-bestは予測診断選択、selectedはvalidation選択の固定モデル。巨大な途中checkpointは作らない。

## 時間制御の検出と修正

初回holdout220000のranking-303/priorで平均1.623ms/手を検出。既存incremental探索器の費用推定が一度の大きな遅延で予算を超えると、その後は子の試行を拒否し続け、推定値を更新する機会も失う問題でした。初回の全ログと実行時コードを保全し、計算費用の観測/推定を予算に対して上限化。100ms遅延の次手で探索が回復するテストを追加しました。

全対照へ同じ修正を適用し、checkpoint・alpha・gateは初回validationの選択のまま固定。旧holdoutで再選択せず、新seed230000/96prefixで全7組を再比較します。最終結果は `confirmed/` を参照。初回結果は削除せず、時間制御問題を含む診断用として残します。

```bash
.venv/bin/python scripts/confirm_ranking_timing.py
.venv/bin/python scripts/audit_ranking.py artifacts/paired-ranking --confirmed
# 確認版prefixで固定specsを再評価
.venv/bin/python -m blokus.value_experiment tournament --prefix-file artifacts/paired-ranking/confirmed/holdout-prefixes.json --specs artifacts/paired-ranking/play-specs.json --output artifacts/ranking-confirm-replay.json --budget-ms 6
```
