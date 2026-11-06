# 固定旧方策の価値再学習：新規事前計画

基点ec64667。前回gate・未確定結果は保持し、今回は別実験。old3101/2/3の1024iteration方策を固定、旧混成exportを対照としてSHA固定。root6/reply3/depth2/6ms、value_features 3x8、終局WDLと視点を不変更。

診断：新しい自己対戦12games/seedとgen3対戦12games/seed（対応gen3固定1751401/2/3、先後同数）。旧方策greedyが訪問する局面について元valueのMSE/符号精度/段階校正/レンジを記録。各ゲームの12/24plyからrootを採り、旧方策root上位6、reply上位3の実際の探索葉を作る。同じ固定旧方策greedy両者で2回完走した実WDLを葉ラベルとする。NNの値/着手を学習教師にしない。gen3対戦の実結果は診断のみ。追加でroot順位7/13を同深度・同継続で調べ、候補落ちを限定的に診断する（全合法手最善ではない）。葉の継続は同root内で共通乱数、経路差あり。学習には診断履歴を使わない。

必須arm P：旧方策固定の新終局履歴から価値のみ学習。元24係数から継続、optimizerは新規NumPy Adam(lr=.01,beta=.9/.999,eps1e-8,normclip5)、半MSE、batch16games、各seed4096games/256updates終点固定。同経験量3seed。各ゲームでlearner席交替、相手75%同seed固定旧方策greedy/25%初期heuristic greedy（先後均衡）、learner側は固定旧方策greedy。learner手番特徴に最終learner WDLを付け、game内符号を不変更。方策weightsは全更新で不変を検査。終点以外は選択しない。

任意arm S：自己対戦由来の探索葉MSEが実訪問局面MSEの1.10倍以上なら、分布ずれ仮説に基づき1条件だけ追加。learnerと75%相手を固定旧混成探索に変更、残25%相手は初期heuristic。再学習値を生成探索に戻さない。全root完了の同6x3/depth2を固定node処理（生成のみdeadlineなし）で実行して、再開を壁時計から独立させる。P/Sは同prefix系列・同4096games/256updates。P/Sで行動とRNG消費後の経路は異なる。データ生成方策の差を等価実装と呼ばない。

小pilot：学習用とは別seedで各候補16gamesの生成・更新を測る。Sが追加基準に達しても合計見込みが1500秒を超える場合は事前に両armを2048gamesへ揃える。結果を見ない速度判断のみ。診断＋生成＋更新の計算cap1800秒。非有限/係数絶対値>1000で停止、無断延長なし。学習seed3101/2/3固定。新prefix base2290000000+seed*100000、旧評価開始状態/診断開始状態は生成から除外。

新validation64prefix base2300000000、全3seed先後。元混成/gen3、各arm/元混成、各arm/gen3を評価。gateは対元point>=.48かつ対gen3が元のpoint-.03以上、fallback<.10。通過armのうち対gen3 point最大、tieは対元point、なお同値はPを選ぶ。全arm不通過なら大確認なし、停止原因を報告。全3seedの同じarmを選びseed/checkpoint選別しない。予測MSEで選ばない。

候補固定後、新128prefix base2310000000、先後3seed、選択/診断/訓練全訪問状態/過去評価開始状態から分離。選択arm/元混成と選択arm/gen3を主比較family2、prefix単位bootstrap通常95%CIとBonferroni2の97.5%個別point CIを分けて示す。候補が2つでも確認はvalidation選択1つだけ。対元の補正CI下限>.5で方式内強化、対gen3下限>.5で公開AI超え。未達を許容、標本増量・再選択なし。

生成と時間対戦/監査は分離。全棋譜の独立合法性/得点/報酬、手番/pass、更新式、別process保存再開、SHA、分離・算術を監査。NNは対照のみ、追加方策学習/特徴/LR探索なし。main/Webは保持。
