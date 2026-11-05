# 複数世代pool A/B 事前計画

2026-11-06。基点cb74785、main変更なし。3seed各最新6144（model/optimizer/past/RNG）からA/Bを分岐。追加1024iterations=16384games/条件seed、全6run計98304新規局。生成/更新/保存合計5400秒上限、全体3時間目安。6400/6656/6912/7168を保存し、打切り時は全6run共通の最大完了checkpointのみ主比較に用いる。崩壊/非有限停止時は診断し拡張しない。

A=current 1/2,past 1/4,heuristic 1/4。B=current 1/2,heuristic 1/4,past/gen2/gen2048各1/12。固定世代は対応する同seedの公表選択gen2(3072/4096/4096)と2048（全seed2048）。全SHAを学習前lock。past4枠を(iteration+枠番号)%3で循環し、3iterationで各4局、6iterationで先後も均等。端数は正確な実数を報告する。相手の固定モデルやpastに勾配を流さず、current自己対戦では双方をlearnerとして収集、それ以外は指定learner席だけを更新。

loss/報酬/モデル/LR/batch16/2epoch/PPO early-KL/4iteration過去policy交換は変更しない。等追加ゲーム・等PPO呼出回数を主比較とする。対戦軌跡長と既存early-KLにより学習state数・実minibatch更新数は異なるため、厳密な等勾配step数とは主張せず全数報告。これを合わせるためloss/early-stopを変更しない。共通乱数は同じ初期RNGとゲーム別prefix seed/除外集合のみ。行動と対局長が変わるとCUDA抽選の消費や軌跡は異なる。完全なpaired trajectoryではない。

全過去評価prefixを凍結し新生成prefixの途中に評価開始状態があれば棄却。A/B全訪問状態の和集合（過去履歴込み）を新評価bankから除外、評価後実訪問重複も監査。固定世代SHA・相手循環・指定learner更新・pool循環とpast交換をまたぐ別processCUDA再開を検証。

新validation seed1670000、32prefix、各A/B×3seedの共通最終checkpointのみ（途中checkpointを成績で選ばない）。前回selected/main gen2/旧6ms/教師100msへ先後交換。選択は全6runの共通最大iterationという計算量規則で固定。前回selectedとgen2は1403では異なり得るので別評価。B対A直接対戦も32prefix実施。進行gate: 完走・合法/有限/分離監査PASS、かつB対A平均point>=.45、4固定相手平均pointのB−A>=−.05。未達ならpilot止め、gateを変更しない。差が小さくても不確実性を測る目的で通過を認め、成功判定とは呼ばない。

gate通過時のみfresh seed1680000/96prefix・先後交換でB対A、B対gen2、B対教師を全3seed実施。主要pool効果はB対同seed同経験量A。prefix単位CI（席/seedはprefix内集約）とseed別結果、実ms、教師100ms超過率。全seed保持、holdout再選択なし。時間対戦と学習/生成/監査は直列。ネット単体と教師100msは異なる実費用。無改善でも有限runで停止する。
