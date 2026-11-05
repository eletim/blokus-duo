# 終局勝敗による自己対戦PPO（2026-11-05）

新系統を実装・実学習し、3seedの初回pilotと経験量だけを増やした1回の再pilotを完了。再pilotの固定gateは**通過**。mainの既存低予算value/2ply・100ms教師は変更しない。以下はネット単体の結果で、探索教師の成績を流用しない。

設計・v1との差・報酬・制約は[METHOD.md](METHOD.md)、事前条件は[PLAN.md](PLAN.md)、再pilotの固定条件は[REPAIR-PLAN.md](REPAIR-PLAN.md)。全合法手、6ch盤面・49状態・正確な候補セルを使用。初期heuristic logit+ゼロ残差、valueゼロ。終局勝敗+1/0/-1、gamma1、shapingなし。得点差は評価指標のみ。PPO clip .2、2epochs、minibatch16、収集16games、固定pool現在50%/過去25%/heuristic25%。3seedは初期重みを共有、生成ゲーム・行動sample・更新順は異なる。

初回64iterations=1024games/seedでは、選択後の対heuristic点率.52344、CI[.46354,.58333]、平均得点差+.89583でgate未達。KL小・方策崩壊なし・経験量が少ないことから、許可された1回の修正として学習量だけを512iterations=8192games/seedへ増やした。初回で選ばれたcheckpointではなく各seedの最新64から継続し、評価prefixを1430000から1470000へ変更。他のモデル・損失・学習率・相手割合・gateは不変。これ以上の修正はしない。

## holdout分離の修正

当初seed1480000の96prefixを評価した後、厳密な分離監査で1prefixが訓練に一致と判明。seed分離のみでは短いprefixの重複を防げなかった。旧成績・監査失敗logをextended/holdout-1480000-overlapへ保全し、最終の独立holdout結果として採用しない。

checkpointを変更せず、seed1500000から訓練prefix hash/訓練state hashおよび旧validation/holdout/診断prefixとの一致を事前除外し、最初の96prefixを固定した。実際は先頭96件が全て条件を満たした。新結果を見てモデルやprefixを選び直していない。以下の最終holdoutはこの再評価。holdout-quality-failure/disjoint-planに経緯と固定hashを保存。

## 新validation（32prefix、各組64局）

|seed / iteration|対学習前 W-L-D / 得点差|対heuristic W-L-D / 得点差|対旧value/2ply6ms W-L-D / 得点差|
|---|---|---|---|
|1401 / 128|38-22-4 / +1.391|38-22-4 / +1.391|14-45-5 / -5.203|
|1402 / 512|38-19-7 / +2.484|38-19-7 / +2.484|17-44-3 / -4.406|
|1403 / 512|34-29-1 / +0.688|34-29-1 / +0.688|20-42-2 / -6.125|

選択にはheuristic/学習前を同じ比率で使用し、旧valueは除外。学習前はheuristic事前値を持つため両対照はほぼ同じ方策であり、独立した2種類の改善証拠と数えない。旧value/2plyは6msの追加探索を使うため、同一思考時間の比較ではない。validationはcheckpoint選択に使った結果であり、独立最終成績と扱わない。

固定gateの集計:

```json
{
  "heuristic": {
    "point_rate": 0.6041666666666666,
    "difference": 1.5208333333333335,
    "point_ci95": [
      0.5389973958333333,
      0.671875
    ],
    "difference_ci95": [
      0.31770833333333337,
      2.7812499999999996
    ],
    "each_seed_nonnegative": true,
    "advance": true
  },
  "untrained": {
    "point_rate": 0.6041666666666666,
    "difference": 1.5208333333333335,
    "point_ci95": [
      0.5389973958333333,
      0.671875
    ],
    "difference_ci95": [
      0.31770833333333337,
      2.7812499999999996
    ],
    "each_seed_nonnegative": true,
    "advance": true
  }
}
```

選択固定後fresh96prefix先後交換を実施。集計（3seedはprefix内平均、CIはprefix単位）:

{
  "heuristic": {
    "point_rate": 0.6050347222222222,
    "difference": 1.6006944444444446,
    "point_ci95": [
      0.5694444444444444,
      0.640625
    ],
    "difference_ci95": [
      1.0069444444444444,
      2.196180555555555
    ]
  },
  "untrained": {
    "point_rate": 0.6067708333333334,
    "difference": 1.6336805555555554,
    "point_ci95": [
      0.5720486111111112,
      0.6415147569444447
    ],
    "difference_ci95": [
      1.0364149305555554,
      2.227473958333333
    ]
  },
  "oldvalue": {
    "point_rate": 0.33159722222222227,
    "difference": -3.684027777777778,
    "point_ci95": [
      0.2977430555555555,
      0.36547309027777797
    ],
    "difference_ci95": [
      -4.465277777777778,
      -2.9634548611111104
    ]
  }
}

対heuristicのseed別（各192局）:

|seed|勝-敗-分|得点差|得点差CI95%|
|---|---|---|---|
|1401|104-77-11|+1.19792|[0.2864583333333333, 2.1041666666666665]|
|1402|114-70-8|+1.72396|[0.6302083333333334, 2.7345052083333337]|
|1403|117-67-8|+1.88021|[0.801953125, 2.9063802083333337]|

各seedの平均差/CIは上表の通り。CIはこの3seedと共有prefixに条件付けたもので、全ての初期化や相手への安定した優越を保証しない。1401はvalidationにより128回（2048games）を選択、他2seedは512回（8192games）を選択し、holdoutで再選択していない。旧value/2ply6msには全seedで負けている。

## 計算量・検証

初回を含む3seed累積の生成+更新wall **919.32秒**、**24576対局**、**70975 minibatch更新**、current-policy記録**556410件**。収集691.17秒（CPU合法手生成/pack 589.09秒、batch推論collation/転送/同期 83.65秒）、更新214.92秒。初回112秒はこの累積に含み、二重加算しない。上限1800秒内。GPU更新だけで予算を数えていない。初期化・最終export・評価は別計測。主な負荷はCPU手生成/入力で、GPUだけを増強しても解決しない。

実対局の全推論費用・p95・全手msは各validation JSONのtiming、CUDA eventとbatchメモリはextended/policy-diagnostic.json。思考時間に合法手生成・特徴・転送・同期を含む。GPU phase時間と純device時間は区別。phase間に生成/監査を重ねていない。

合法手/mask、配置の区別と勾配、実方策更新、両者視点、pass、15/20点bonus、終局returnを検証。CUDA別process再開は過去policy交換をまたぐ6iterations連続 vs4保存+追加2、model/past/optimizer/対局履歴/Python/NumPy/Torch/CUDA RNGをbit比較。全テスト結果はtests.log。全訓練/評価対局再生、初期log_prob再計算、終局target、相手割合、選択最大値、prefix対の算術をaudit.jsonに保存。これは実装側監査であり秘書の独立実装監査とは呼ばない。

## 原因分析と次段階の判断材料

曲線はextended/curves.png/svgとextended/learning-summary.json。終局returnとpolicyは実際に変化したが、PPOの学習損失低下や自己対戦勝率を固定基準への強化成功と扱わない。heuristic事前分布の強さ、確率的訓練対greedy実戦の差、終局勝敗の高分散、短い過去policy pool、少数seed・prefixに限られる。詳細な方策変化/初期とのKLはextended/policy-diagnostic.json。

この依頼ではroot補正へ戻らず、自己対戦系統の再現可能な土台を残した。改善が確認できない場合は、追加の無根拠な長時間更新ではなく、固定poolに対する方策改善の再現性と確率的/greedy評価の差を先に診断する。次の学習条件変更や推奨変更は秘書の判断へ返す。

## 再現コマンド

```bash
cd /home/eletim/blokus-duo-ai
# 初回（他seedは1402,1403）
.venv/bin/python -m blokus.terminal_ppo --seed 1401 --iterations 64 --output /tmp/ppo-first
# 経験量だけ増加。--iterationsは累積値、最新64から開始
.venv/bin/python -m blokus.terminal_ppo --seed 1401 --iterations 512 --resume /tmp/ppo-first/iteration-64.pt --save-steps 128 256 512 --output /tmp/ppo-extended
# 評価は学習/生成と同時実行しない
.venv/bin/python -m blokus.terminal_eval validation --root artifacts/terminal-ppo/extended --prefix-seed 1470000 --steps 128 256 512
.venv/bin/python -m blokus.terminal_eval holdout --root artifacts/terminal-ppo/extended --fixture artifacts/terminal-ppo/extended/holdout-disjoint-prefixes.json
.venv/bin/python scripts/audit_terminal_holdout.py
.venv/bin/python scripts/check_terminal_resume.py
.venv/bin/python scripts/audit_terminal.py --root artifacts/terminal-ppo/extended
.venv/bin/python -m pytest -q
# ネット単体CLI（選択checkpointはextended/selection.json）
.venv/bin/python -m blokus.terminal_play --model artifacts/terminal-ppo/extended/seed-1401/iteration-128.pt --opponent heuristic
# 同コマンドに --human-seat 1 で人間対AI
```

結果JSON・source hash・設定・seed・checkpoint metadataを保存。過去bank/holdout/探索教師ラベルを本学習には使わない。必要な初期/選択/継続用checkpointを保持し、重複latestと未選択中間checkpointは検証後にpruned-checkpoints.jsonへhashを記録して削減する。学習対局ログはgzip圧縮で保存し、auditは両形式に対応する。

最終検証: **94 tests PASS**。最終seed1500000 holdoutの訓練prefix/訪問状態重複はいずれも0、1728対局再生・算術・選択hash監査PASS。旧1480000の1件重複は隠さず記録として保持。

最終holdoutでのネット単体1手平均（合法手生成・転送・同期込み）: seed1401 1.322ms (p95 2.609ms), seed1402 1.329ms (p95 2.669ms), seed1403 1.198ms (p95 2.506ms)。
