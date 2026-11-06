# Minibatch 16 vs 64 事前計画
基点12809f3、3seed1751401/2/3の最新4096(model/past/optimizer/RNG)からA16/B64へ分岐。選択2048は使わない。収集16games、epochs2、LR3e-4、loss/terminal WDL/pool current50 past25 heuristic25/past交換4iter、float32、native、常駐updateを共通固定。唯一minibatchを変更。optimizer step数/earlyKLは変わるため等価実装・等更新量とは呼ばない。
同一保存rolloutを2warmup+6交互反復、GPU event/実wall/候補padding/peak memory/stepsを比較。小pilot8iterationsと別process再開4+4で有限性とmodel/past/optimizer/RNGを検証。batch16既定経路の既存テストを維持。
主実験は6run各+1024iterations=16384games、共通保存4608/5120、計98304games。合計生成更新保存wall3600秒以内。共通checkpointで終了、Bだけ追加しない。結果を見る前に終了点5120を固定。checkpoint選択探索なし。明確な非有限・崩壊は停止。外生prefix RNGはseed/iterationから独立生成し両条件で一致、行動と将来rolloutは更新により異なる。
新validation64prefix(seed17641000)、全3seed保持、4608/5120の両条件対gen2/固定4096とB対A。主要指標gen2 point、直接B/Aを必須。5120が共通終了点で固定、途中成績で変更しない。進行gate: 終点B対A cluster point>=.5、B対gen2 point>=A対gen2 point、有限性/再生/分離監査合格。gate未達ならholdoutなし（学習条件の効果未確認とする）。通過時fresh96prefix(seed17661000)でB/A・B/gen2・B/4096、同prefix3seed両座席cluster bootstrap4000、後からseed/checkpoint選択なし。gen2への優越はCI下限>.5でのみ主張。
全訓練訪問状態/過去評価開始状態を評価開始から除外。前成果の全状態indexをSHA検証して参照、新履歴のみ追加、全新ゲームの合法手/得点/学習side/return/状態hashと評価再生/算術を監査。過去評価開始状態を新訓練prefixから除外。学習と計測対戦を重ねない。観測したiteration/game/wall曲線のみ報告、未観測の等wall棋力を推定しない。
compile/shape: 前runのcapture失敗は保持し今回batch条件に別最適化を混ぜない。ランダムminibatchの候補最大長paddingが増える原因と64の実測比を報告。main不変更、公開gen2不変更。

実行前修正記録: 新driverでseed_allによる決定的演算/CPU thread設定が抜けていたため、初回resumeはmodel不一致（RNG一致）。本実験前にrestoreの前へ設定を追加。修正前history/logをdiagnostic-before-determinism-fixへ保全。修正後8 vs 4+4 CUDA別processで全bit一致。両条件に同じ修正適用。checkpoint/gate/更新条件の変更なし。
小pilotとbenchmarkは本実験開始前に完了。固定rollout 376states、2warmup/6交互測定。16の中央値.113636秒、64 .050849秒。ただし37 vs12stepで等更新量ではない。padding3.600→4.070、peak169.6→310.3MB。compile/AMP/別packingは採用しない。
