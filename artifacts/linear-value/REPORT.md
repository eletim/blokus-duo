# 非ニューラル終局強化学習・価値探索の有限比較

学習前には改善したが、公開gen3超えは未達。特徴追加は同経験量の旧30係数方策に劣った。一方、終局勝敗から学んだ非NN価値による候補限定2ply探索は、この新方策単体および同じ候補幅の静的葉評価より良かった。main・Web・公開モデルは変更しない。

## 何を学んだか

ニューラルネットや探索教師は使わない。実ゲーム終局の勝ち+1、引分0、負け-1から、候補softmaxの係数をREINFORCE/NumPy Adamで更新した。得点差は学習報酬でなく評価指標。初期値はheuristic priorで、完全ランダム初期化ではない。旧版は10特徴×3段階=30係数、新版は配置数の少なさ、空き連結領域、相手接続点封鎖の3特徴を追加した39係数。同時に、 learner側局面の終局勝敗を予測する8特徴×3段階=24係数の線形価値を別損失で学習した。価値の学習は方策更新へ入れていない。gen3は対戦対照のみ。

空き連結領域は現在の盤面から計算する代理特徴で、着手後の全残駒の厳密配置可能性ではない。行動に依存しない旧特徴2個は除外済み。旧特徴のfrontier増分と総数には局面内線形従属もあり、係数個々に独立した意味を与えない。

## 固定した実験

PLAN.mdを先に保存。速度pilot後、両方式3seed(3101/3102/3103)を新規初期値から各1024iteration×16games=16,384gamesへ固定、計98,304games。途中256/512は保存のみ、終点の後付け選択なし。旧pilotの56.4%を今回の最終成績に再利用していない。

相手はbatch開始時current50%、past25%、初期greedy25%。各相手で先後同数。learner側だけを更新。実learner局面比はcurrent49.71–49.77%、past24.84–24.90%、初期25.34–25.40%（learning-summary.json）。旧新で外生prefix系列を共有し、更新後の軌跡は異なる。過去評価開始状態を生成から除外。全訪問状態3,078,061件（速度pilotも含む）を監査用に索引化した。

新validation64prefixで探索の6/25msを比較。価値MSEがゼロ予測より良く、単体にpoint>=50%、fallback<50%なら探索へ進行。6/25msは同pointで、事前tie規則の小予算6msを固定。root上位6、相手上位3、深さ2。全rootを同深さまで完了した場合だけ採用し、途中打切りは元の方策greedyへ戻す。完全探索とは呼ばない。

## 未使用128prefixの確認結果

各組は3seed×先後交換×128prefix=768局、6組計4,608局。gen3も対応する固定3seed1751401/2/3を使用。pointは勝ち1・引分0.5の割合。CIは同prefix内の先後・3seedをまとめたbootstrapであり、6局を独立標本扱いしない。得点差は左側から見たゲーム得点差。固定3seedでの結果で、seed母集団への一般化は未検証。

|比較（左/右）|point・通常95%CI|平均得点差・95%CI|左/右 実ms/手|
|---|---|---|---|
|新方策/旧30係数学習済|45.25% [42.06,48.44]|-0.36 [-0.90,+0.18]|0.081/0.033|
|新方策/学習前|64.71% [61.33,68.10]|+2.63 [+2.12,+3.16]|0.077/0.037|
|新方策/gen3|19.01% [16.47,21.48]|-7.08 [-7.65,-6.50]|0.104/0.272|
|新価値探索/新方策|77.02% [73.96,80.01]|+4.55 [+4.06,+5.05]|2.332/0.082|
|新価値探索/静的葉探索|78.84% [75.65,81.90]|+6.57 [+5.94,+7.20]|2.310/0.682|
|新価値探索/gen3|35.87% [32.81,39.00]|-3.45 [-4.03,-2.86]|2.402/0.315|

最初の3組を主比較familyとし、Bonferroni3補正point CIは順に[41.28,49.15]、[60.48,68.88]、[15.95,22.07]%。追加探索3組は診断の通常CIで、主比較と混ぜない（機械JSONには全組の補正計算欄もある）。旧30係数へのpoint差は負、得点差CIは0を含む。seed別新/旧pointは56.25/42.38/37.11%と差がある。seedを選別しない。

静的葉は得点差/109、終局は正確なWDL。探索幅・深さ・6ms上限を揃えたが実消費時間は異なるため、同思考時間の優越とは主張しない。これは最強手作り評価との比較でもない。gen3は先読みなしの公開PPOで学習経験量も異なり、非NNとの等経験量比較ではない。

価値calibrationは新規192games、3seedのMSE .676/.689/.709、ゼロ予測 .972/.906/.923。近い相手分布での予測改善であり、minimax価値の正確性や未知相手への校正を保証しない。calibrationのpast相当slotは最終currentで代用している。

## 費用・制約

本学習全体372.38秒（生成/更新/保存・索引処理込み）、GPU学習0。旧各34.72–35.29秒、新各87.43–88.33秒。同ゲーム数で特徴/value費用により遅くなった。最終64iterationのentropyは旧2.22–2.26、新2.16–2.19、value MSE約.703–.712。有限性破綻なし。学習時間上限90分以内で終了し追加延長しない。

独立8開発ゲームの4/16/28ply、warmup後5回交互測定の入力生成中央値msは旧 .0548/.0357/.0175、新 .1315/.0898/.0455、価値特徴 .1855/.1244/.0681。旧新は同保存局面。新係数計504bytes、配置テーブル2,855,632bytes、profile process peak RSS約477MiB（共有基盤のTorch import込み。モデルがNNという意味ではない）。cold import/初期配置テーブル費用は定常値に含まれない。候補数・特徴変動幅はprofile.json。候補ごとの全列挙重複はせず、局面内共有計算を使用。

対gen3の探索は平均約19.87nodes/手、同深さ完了99.975%、fallback0.025%、6ms超過観測0%。対新方策では超過1/12,205手。soft deadlineでありOS/初回初期化のhard保証なし。CPU Python fallbackは動作するがnative測定速度を保証しない。時間制御探索の着手は負荷に依存し、固定ノード/無制限小終盤テストとは区別する。

## 検証と次案

98tests PASS。native/Python候補順・特徴、独立全探索小終盤の値/最善手、pass/連続手/終局+15/+20/手番符号、有限差分勾配、打切りfallbackと状態不変性を検証。別process66更新 vs 63+保存再開3更新（past交換64を跨ぐ）で旧新とも全state JSON一致（weights/value/optimizer/past/RNG）。CPU-only環境でPython方策対戦とnative探索対戦を完走、人間先後入力/EOF経路を確認。

98,304訓練対局と全validation/confirmationを合法手・終局得点・報酬・算術・model hashで再生監査PASS。評価開始状態だけでなくpostprefix実訪問状態も訓練との重複0。初期空盤面等、採用prefix以前は分離対象外。calibration値も再生。旧pilot/過去prefixから独立。詳細はtraining-audit/evaluation-audit/arithmetic-audit/resume-audit.json。

次は新特徴を増やすより、旧30係数方策と今回の学習価値を組み合わせる事前固定のablationが具体的候補。旧方策の方が強く速いため。ただし今回holdoutから作った仮説で、未実験・未採用。より強い非NN探索相手で終局価値を再学習する案は別段階。main/gen3置換の根拠はまだない。

## 再現

リポジトリroot、既存requirementsのvenv、OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1を使用。nativeのビルド手順は既存README参照。モデルJSONは履歴不要、CPU推論のみで利用可。

```bash
python -m blokus.linear_play2 --checkpoint artifacts/linear-value/new-3101/policy.json --backend python --human-seat 0
python -m blokus.linear_play2 --checkpoint artifacts/linear-value/new-3101/policy.json --backend native --search-ms 6 --opponent-checkpoint artifacts/linear-value/old-3101/policy.json
python -m scripts.check_linear_value_resume
python -m pytest -q
python -m scripts.audit_linear_value
python -m scripts.linear_value_eval audit
python -m scripts.audit_linear_value_results
```

学習再実行は保存物を上書きせず新outputへ `python -m scripts.linear_value_run pilot --output artifacts/linear-value-rerun`、次に `train --output artifacts/linear-value-rerun`。既存評価開始状態registryの絶対参照はexclusions.jsonへ記録済み。厳密再現時は保存済みregistryを使用すること。評価scriptsの既定artifact rootは今回のlinear-value。巨大派生training-states索引はgit対象外、audit_linear_valueが棋譜から/tmp/linear-value-training.npyへ再構築する。過去巨大訓練履歴は複製していない。

## main向け最小候補（未統合）

`integration/linear-value-candidate` commit `abc41cf` は最新main86d9e51から、実装5ファイル・テスト・6個の推論JSON・短い利用文書だけを追加した候補。研究履歴/評価棋譜を含まない。研究版とのコードとモデルbyte一致をintegration-audit.jsonで確認。全93tests PASS、CPU-only/PythonでAI対戦完走。コンパイラのないCPU-only venvではnativeビルドに失敗したため、開発venvのziglangでビルドした（Python fallbackには不要）。研究branch全98tests PASSとの差は既存pilot testsの有無。

研究版の最終source-manifestは入力検証とCLIオプション追加後のソース。評価後にモデル重み・学習式・探索の選択条件は変更していない。今回保存された結果は単なるpilotの速度評価でなく固定終点の比較だが、追加特徴の失敗と公開gen3未達を含む。候補をmainへ自動統合しない。
