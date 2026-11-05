# 固定gen2 policy/value短時間探索

基点952d4cb、追加学習0、main変更なし。公開gen2 seed1401/1402/1403（3072/4096/4096）のexport重みを固定しSHA記録。PPO valueは手番側の終局勝敗期待値[-1,1]。得点差として使わない。終局は実得点差のsignで上書き。両者全置きも残る強制passで得点が変わらないため既知終局値とする。

初版はpolicy上位候補を用いるbatched 2ply selective negamax。root policy greedyを確実なfallbackとして先に計算し、root候補すべての相手policy上位応手を評価してroot視点minを比較。候補幅制限があるため厳密minimaxではない。途中計算のみで未評価候補を0扱いせず、未完了なら単体policyへfallback。初案6ms=4x2、25ms=8x3。CPU合法手/特徴生成・GPU同期/推論をact開始から計測、段階ごとの予測費用と0.4ms以上の余裕を取り打切り。OS由来のhard deadline保証はしない。試合ごとcost推定をcold reset、TTなし。

新dev1690000/36prefixで独立36ゲーム（3seed×self/heuristic/旧6ms×4）を生成し、各ゲーム最大3root（ply8/20/32）でvalue校正とCPU/GPU費用を診断。ゲーム内rootを独立標本とは扱わない。valueはこの継続相手での勝敗と比較する条件付き小標本診断。小終盤では独立座標checkerによる全探索と照合し、2plyで全意思決定が終わる場合だけ一致を要求する。一般中盤で厳密解とは呼ばない。

新validation1700000/32prefix、各3seedで6/25ms探索対同checkpointネット単体。補助として選択方式対旧6msを評価。方式/予算は全seed共通、seed/checkpoint選別なし。優先criterion=単体対戦の平均point、同値は平均得点差、さらに短予算。進行gate=平均point>=.52、平均得点差>0、各seedのpoint CI上限>=.5、非強制手でdeadline fallback率<.5、合法/分離監査PASS。

初版がgate未達の場合、一度だけ保守的修正を許可: policy greedyを置き換えるのは全root同じ深度完了かつvalue改善margin>=.15（勝敗期待値単位）の場合。幅/予算は同じ6/25ms。この修正にも同gateを適用し、未達なら大規模holdoutへ進まず終了。修正の発動理由を保存。予測費用の初期値はdevプロファイルで事前固定しvalidation中に設定変更しない。試合内の実費用推定更新のみ可。

gate通過時だけ新fresh1710000/96prefix先後交換、3seed全てで選択探索対同ネット単体/100ms教師。ネット単体約1.4msと6/25ms、教師100msは同時間比較と呼ばない。prefix単位CI、実ms/超過/展開数/fallback/policy保持率を記録。全訓練状態はppo-generation-poolのA/B和集合indexをSHA参照し、過去評価開始状態とともに新bankから除外。評価後の実訪問も監査。計測対戦と生成/監査を重ねない。全体2時間上限目安、学習や追加runの自動延長なし。

dev実測後・validation前の費用予測固定: 初期policy batch0.8ms/leaf batch0.85ms（dev25msの平均は各0.56/0.55ms、完了呼出だけでも約0.70/0.68ms）。child準備予測は既存測定と当該root観測生成×1.25の小さい方、最低0.1ms。初期6msは過大な固定費予測で全89非強制手がpreflight fallbackとなったため、終盤の小候補でも実測準備費を反映できるようにした。深さ・幅・着手条件は変更しない。初期dev結果を保持し、新予測の再プロファイルを別保存する。
