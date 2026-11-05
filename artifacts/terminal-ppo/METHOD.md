# 終局自己対戦の実装と制約

初期v1は12候補要約+46状態MLP、終局得点差/109を教師とする単純actor-critic（初期には模倣phase）。本系統はv1重み/教師を引き継がず、エンジン・特徴計算・盤面encode・prefix評価だけを再利用する。

6ch盤面CNNの全盤面特徴、49状態（残駒、score、pass）、39候補要約、候補を占める各セルの学習埋込とその位置のconv特徴を結合する。位置埋込の和は占有セルmaskへの線形射影と等価。候補IDだけを入力する方式ではない。同一要約・異なる配置を出力/勾配で区別する回帰テストを設けた。padding座標196はゼロ、実盤面0..195。残駒/合法性による全合法手集合に確率を正規化し、候補数の上位打切りはない。passの確率は合法手なし時だけ1。

初期logitはheuristic+ゼロ残差、初期valueゼロ。学習前も同一ネットで推論する。学習時はCategorical sampling、実戦評価はgreedyと乱数による同値処理。両者の分布差はある。相手poolの現在policyと過去policyも訓練時sampling、heuristicはgreedy。過去policyは4更新単位ごとに交換する1スロットで、長期leagueではない。

全ゲームを終了させ、scoreの大小から+1/0/-1を**実プレイヤー番号**で各current-policy記録へ割り当てる。gamma1のMonte Carlo return、advantage=return-old_valueを収集batch内で標準化。passを飛ばしたtraceの偶奇では符号を決めない。途中局面のscore差は報酬に使わない。valueは勝敗returnを回帰し、得点差は評価指標のみ。entropy正則化は原報酬へのshapingではない。

PPOは凍結した収集時log_prob/valueを用い、clip .2、2epoch、value MSE .5、entropy .01、global grad norm .5、minibatch16。近似KL .03超で残り更新を打切る。各iterationのゲームは同じcurrent重みで収集し、相手側past/heuristic行動はpolicy lossに混ぜない。現在同士の場合は両者ともcurrentサンプル。同期16ゲームで推論をまとめ、CPU側の手生成は直列。参照: [PPO原論文](https://arxiv.org/abs/1707.06347)。

時間は収集・更新wallを計上し、CUDA同期待ちと入力転送も含む。generation内のinference値はcollation/転送/同期を含むphase時間で純kernel時間ではない。独立profileでCUDA event kernel時間も記録する。初期モデル構築・最終artifact exportは学習ループ計測の外、全体所要時間は別途記録。CPU worker待ちはなく、並列CPU生成は今回未導入。

ネット単体を評価し、旧value/2ply6msは追加計算を使う強い基準として扱う。時間同等・探索付き同士の比較とは主張しない。学習済み同士の勝率は監視にとどめ、固定validationの学習前/heuristicで選択・gate判定する。選択後のvalidation結果には最大値選択の楽観性があり、通過時のみfresh holdoutで確かめる。CIは同一prefixの2座席を一組とし、3seed比較も共有prefix単位に平均する。モデルseed自体は3つなのでseed一般化の精度は限定的。

制約: heuristic事前分布が探索を狭める可能性、終局勝敗returnの大きな分散、相手poolの短期性、有限ゲーム数、greedy評価との分布差。PPOを実装しただけで強化成功とはしない。

割合50/25/25は**対局数**の比率。現在policy同士だけは両側から記録するので、学習state比率は概ね2/3・1/6・1/6となり、厳密な件数はaudit.jsonに保存する。固定heuristicへの直接経験は全学習stateの25%ではない。

学習前logitはheuristic事前値なので、学習前ネットとheuristicはほぼ同じ方策（浮動小数点と同値処理で差が出ることはある）。両対照の成績が一致しても、独立した2種類の強化成功証拠とは数えない。CIは相手別に計算し、同一結果を足して標本数を増やさない。
