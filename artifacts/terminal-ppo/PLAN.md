# Terminal PPO 事前計画
基点f6db8c9、新branch research/terminal-ppo。main変更なし。root補正は終了。
全合法手（passは合法手なし時のみ）の可変長分布。盤面6ch CNN、残駒/score/pass49状態、候補39要約と正確な占有セルの位置埋込・盤面conv局所特徴を使用。maskでpaddingを除外。候補上位制限なし。ネット単体、探索なし。
初期policyは従来heuristic logitへのゼロ残差、valueゼロ。初期seed1400共有、学習seed1401/1402/1403（対局乱数・sample・minibatchは別）。初期checkpointを保持。同一アーキテクチャ学習前を対照。
終局勝敗+1/0/-1のみをMC return（gamma1）とする。各記録の実プレイヤー視点へ終局scoreから直接割当、advantage=return-old_value、batch内標準化。得点差は評価指標のみ、shapingなし。PPO clip .2、value MSE係数.5、entropy係数.01、Adam lr .0003、grad clip .5、2epochs、minibatch16states、近似KL .03超でepoch中断。
16並列対局を1収集単位とし、current同士50%/凍結過去policy25%/heuristic25%を毎単位固定。後2種では座席を均等化しcurrent側のみ更新、current同士は両側記録。凍結過去は初期から開始し4単位毎に更新する1スロットpool（過去全履歴ではない）。毎対局新seedの合法random2–6ply prefix。全局終局まで完了後更新、保存は単位境界。iteration=収集+更新。
まずsmokeとCUDA別process再開。pilot各seed64iterations=1024games、保存16/32/64。3seed合計生成+更新wall上限1800秒（各seed600秒、途中の単位は終了して記録）。上限時は未達と明記。timingはCPUlegal/pack、GPU推論同期、更新、残余を記録。途中ckptは3個以下。
validation固定新prefix seed1430000、32prefix先後交換。相手poolはheuristic/各seed学習前を等重み。各seedのcheckpointは勝敗点率平均最大、同値は得点差→早いiteration。oldvalue/2ply6msは選択後の強い比較基準、選択に使わない。
対戦進行gate: 選択後各seedについて両基準への点率>=.5、3seed平均点率>=.53かつ平均得点差>0、さらに各基準へのprefix単位seed平均bootstrap95%下限点率>.5。旧教師MSE gateは使わない。未達なら最終holdoutは作らない。診断で根拠があれば1回だけ修正・新validationで再pilot可、その条件は開始前に追記する。
gate通過時のみ設定固定、fresh seed1440000 96prefix先後交換で3seed/学習前/heuristic/旧valueと比較。時間同等を主張しない。単独計測、試合cold。全対局再生・終局報酬・算術監査。PPO参考 https://arxiv.org/abs/1707.06347 。初期v1は46状態+12要約MLP・単純終局得点差actor-critic、今回は空間/正確配置・勝敗PPO・batch/pool・clip/固定検証を導入。
