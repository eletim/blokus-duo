# 非NN局所表価値：事前計画（2026-11-06）
基点5776630。ec64667元混成の旧30係数方策/元24係数価値seed3101/2/3を固定。拒否済みP不使用。main/Web/gen3を変更しない。

F=元線形価値凍結。L=Fの24係数から同じ新データで再学習。T=Fを凍結してゼロ初期局所表残差だけを学習。初期raw/clipped予測はF/L/Tで同じ。方策は全条件不変。推論は全条件同じCompletedSearch、base6x3/2ply→target24x8/2ply、100ms。Tが速くても幅変更なし。

表: 2x3と転置3x2の盤内全312窓、空0/手番側1/相手2の3進6桁729キー。盤端に接する窓/内部の2群、残駒平均で元価値と同じ3段階。180度反転キーはmin(code,reverse(code))で共有し、転置窓も同じ表。3×2×729=4374 float64係数（34992byte）。翻訳共有による意図した集約であり、hash衝突なし。盤外を空マスに誤認せず盤内窓だけ列挙し端群を識別。残駒/得点/開始点frontierなど大域情報は元線形価値を保持。
残差は312窓の(table(自相)-table(相自))/624。色/手番交換で反対符号、盤転置と180度回転+色交換は開始点を保存するルール対称性として検証。他の盤対称性をデータ拡張に使わない。終局は表を無視し実得点のWDL。最終非終局値は[-1,1]へclip。

L/Tとも原raw出力へのhalf-MSE、終局WDLを全非終局訪問状態の当時の手番視点へ割当。Adam float64 lr=.01 beta=.9/.999 eps1e-8、勾配norm上限5。8epoch、minibatch256、同じ順序/RNG seed、各1sample=1局面。同じ対局/局面/ラベル、同じ更新回数。LとTでparameter数は違う。NN/autogradなし。学習した値を生成探索へ戻さない。

生成は凍結Bによる100%自己対戦（両席同じ対応seed）。時間切れなしの固定24x8/2ply（同じ完成段階方式ms=0）で負荷による軌跡変化を防ぐ。4ply合法ランダムprefixから開始、Python Randomのゲームseedを保存。全3seed、独立process最大3。pilot4games/seedで時間/表訪問を測る。1seed1024gamesの推定生成wallが20分以内なら1024、超えるなら512に一度固定。生成+学習90分上限、非有限時中止、validation成績で延長しない。学習量固定後に新train開始。校正は別32games/seedの同凍結生成、学習に使わない。

新validation32prefix×3seed×先後（192局/組）でT/L、T/F、L/F、F/gen3、L/gen3、T/gen3。対応gen3全3seed。Tだけを候補とし、T/L>=.50、T/F>=.50、T/gen3>=F/gen3-.02、追加段階完成>=.80、greedy fallback<.10で確認へ進む（探索的gate、成功主張ではない）。T未達なら追加条件/再学習なし。
通過時固定Tをfresh128prefix×3seed×先後でT/L、T/F、T/gen3に確認。3つのpoint主比較にBonferroni family3 CI、通常95%CIと得点差95%CIを併記。prefix単位6局cluster bootstrap10000回。モデル/seed選別なし。

既存非NN/PPO訓練indexと過去評価開始prefixを除外、新訓練全状態も保存し評価前後に照合。未使用ゲームの段階別MSE/符号/予測レンジ、表キー訪問率・未訪問率・意図した共有率/実メモリを測る。棋譜独立座標再生、報酬、解析更新式、別process再開、モデルSHA、CPU-only CLIを検証。計測対戦は生成/学習/監査と重ねない。
