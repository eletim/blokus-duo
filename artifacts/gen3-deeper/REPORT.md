**結論: Eは公開反映済み。固定幅6plyのHは新しい独立確認でEへの改善条件を達成。Hはまだ本番未適用の候補です。**

# E公開統合と固定幅6ply研究

## 現在使える変更
Eはmain **8fb3e9d**へ通常FF+分離記録commitで統合し、http://100.95.30.96:8765 の既存Webへ反映。公開済み表記の文書補足は**2adc36f**（再起動不要）。rollbackは4f13460、設定/起動コマンドはartifacts/value-leaf-deployment/DEPLOY.md。NN探索の葉でpolicy候補処理を省略する計算効率改善であり、棋力改善とは表示しない。単体gen3/非NNを含む人間先後smoke PASS、gen3全重みとFrozenタグを保持。

## 今回の研究条件
Hは幅4を全深度で固定し、深さ2→4→6の各全root完了時のみ採用。時間切れは最後の完成段階（通常4ply）へ復帰。alpha-betaとTTは固定木を変えない。TT keyは盤面/手番/残駒/last/pass/深さ/幅、EXACT/LOWER/UPPERを区別、未完了ノードは保存しない。boundは既存窓で確実なcutoff時のみ利用し、窓を狭めない保守的方式。act間再利用なし。モデル再学習0、候補拡幅Dは不採用のまま。

固定木の事前許容差はatol1e-7/rtol0。枝刈りなし幅4/depth6、幅2/depth4/6、従来E4との値/選択、小終盤独立oracle、pass/bonus、打切り4ply復帰をテストした。初回1件はテスト側が旧クラスに上限外20000msを指定した失敗で、5000msへ修正（元ログ保持）。TT窓/flag分類の潜在的不整合をvalidation前レビューで保守化し、pilotを再実施。修正前結果は別名で保持、validation中の変更なし。

## 費用pilotと固定計画
新9保存局面×2反復、E/Hを交互順に測定。500ms E69.21ms/H271.10ms、H6ply8/18。2000ms E67.82ms/H341.91ms、H6ply14/18。成績を見る前に500ms/maxdepth6を固定、2000ms対戦の追加探索はしない。合法手生成/NN推論/同期を含む実時間、同上限であり実消費同一ではない。

新validation16prefix×先後×固定3seed（1751401/2/3）、各比較96局。H/Eが主、H/単体policyが補助。pointは勝1/分0.5。両席/3seedを同prefixクラスタとしてbootstrap20,000回。以下は選択用通常95%CIで、確認標本と合算しない。

|比較|point|95% CI|得点差|得点差95% CI|平均ms 左/右|
|---|---:|---|---:|---|---|
|H-E|69.79%|63.02–76.56%|+3.250|+1.917–+4.698|254.20/65.58|
|H-policy|70.83%|61.98–80.21%|+4.000|+2.583–+5.625|254.82/0.84|

事前gate: **True**、主比較でHの非pass6ply完成率78.15%。gate/標本数は変更していない。

## 独立確認
新32prefix×先後×3seed、各192局。

|比較|point|95% CI|得点差|得点差95% CI|平均ms 左/右|
|---|---:|---|---:|---|---|
|H-E|62.76%|57.81–67.45%|+2.417|+1.562–+3.302|262.69/68.92|
|H-policy|68.75%|62.24–74.74%|+4.688|+3.323–+6.109|255.00/0.87|

H-E 2比較補正97.5% point CI: 57.29–68.23%。

H-policy 2比較補正97.5% point CI: 61.20–75.78%。

主H/Eの事前改善条件達成: True。

## 候補推定値と実継続WDLの診断
新6元ゲーム、各1局面（12/20/28ply各2）、seed1751401のみ。共通top8（greedy保持）候補の固定4ply幅4のNN backed-up値と、その候補を置いて終局まで実プレイしたWDLを比較。継続分布は固定greedyと固定E2幅4の完成探索。E2はもう一つの探索継続だがE4より強いとは仮定しない。共通乱数2繰返し、同ゲームの候補・繰返しは相関し独立標本ではない。gen3重みは変更しない。

6局面中、root4→8で推定値が上昇して選択が変わったのは1局面（推定+0.017程度）のみ。その局面で両継続WDLの増加は0。他5局面は推定選択不変。これだけで「value誤差が主因」とは断定できない。離散WDLの同値が多い、小標本、継続方策依存、ゲーム段階分布が限られる。推定値はminimax真値ではなく、継続結果も最善応手を保証しない。人間棋譜は未取得。

各候補の値/段階/終局得点/全棋譜、元ゲームID、予測MSE/符号精度/予測レンジはdiagnostic.json。棋譜追加保存の再実行は同じ固定データでsummary一致を確認し、条件変更には使っていない。

## 再現
```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m pytest -q tests/test_ppo_deeper.py
python scripts/pilot_deeper.py
python scripts/diagnose_deeper_value.py
# 過去bankと訓練index参照が必要。既存評価bankを成績で再生成しない。
python scripts/run_deeper.py prepare
python scripts/run_deeper.py run
python scripts/run_deeper.py audit
python scripts/audit_deeper.py
python -m blokus.ppo_deeper_play --method H --budget-ms 500 --human-seat 0
```

研究Hは本番未適用。mainには検証済みEのみ。独立棋力確認なしに深さ増を強化成功と呼ばない。研究の選択/モデルhash/全棋譜/seed別・席別成績/冷起動warmup/超過と完成深さは各JSONに保存。過去巨大訓練履歴はSHA参照のみ。

## 最終検証・費用・採用範囲

研究CPU-only全111PASS/4skip。候補関連4PASS、新cloneはnativeビルド後CPU-only全100PASS/4skip、合成合法番号入力で人間先後CLI完走（実際の人間棋譜ではない）。候補 `integration/gen3-deeper-candidate` **0740f31c19e03cefdc8a7747c573eca9dcf14216** は探索器/CLI/テスト/説明の4ファイルだけ。研究moduleから使っていたkey関数を同一式でinline化し、その他の探索source一致を検証した。重み/訓練履歴の複製なし。

評価576局（validation192、確認384）と診断180継続の独立座標合法性/pass/得点/報酬・集計・CI再計算・SHA監査PASS。訓練訪問状態との評価重複0、確認開始対validation訪問重複0、全訪問のphase間重複0。診断開始局面も訓練重複0。診断45候補×2継続分布×2繰返しは6元ゲーム内で相関し、180独立標本とは呼ばない。

対戦計測wall **2416.86秒（40.28分）**。費用pilot2回・診断2回・軽い検証を含め90分枠内、学習0。全対戦を単独実行し、監査/テストは対戦終了後。cold_load_secondsはモデル構築と内部warmupを含み、warmup_secondsは追加1回の観測。実着手時間には全合法手/特徴/NN/CPU同期を含む。研究メモリ観測は約518MiB、厳密peak測定ではない。

独立主H/EはH平均262.69ms、E68.92ms。Hは非pass2487手中1850手（74.39%）で6ply、残り637手は完成4plyへ戻った。未完成部分値は採用しない。非passで4ply未満やpolicy fallbackは0。500ms超過は637/2742手（pass含む23.23%）、p99501.19ms、最大550.09ms。補助H/policyの最大は551.54ms。超過が全て小さいとは保証せず、hard上限とは呼ばない。

主比較seed別（各64局、勝/分/敗、point、得点差）:
- 1751401: 40/2/22、64.06%、+2.48
- 1751402: 38/4/22、62.50%、+3.05
- 1751403: 39/1/24、61.72%、+1.72

この固定3seed・合法ランダム4〜10ply開始分布・500ms上限ではH/E改善が確認された。全seed母集団、人間、通常空盤面からの全局面、同じ実消費時間への一般化はしない。HはEより約3.81倍実時間を使う。強化を新たな重み学習や価値の真値化とは呼ばない。診断の小標本だけでD失敗やH改善の因果を断定できない。追加幅/深さ/LR探索をせず、この候補の統合判断を秘書へ返す。
