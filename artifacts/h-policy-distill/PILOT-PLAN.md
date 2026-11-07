# H方策蒸留・適否診断（対戦成績を見る前）
本番/Frozenは不変。人間棋譜0件との報告を受領、読込みも教師化もしない。
TerminalNetのconv/global_netはpolicy/value共有。positionはpolicy専用だが今回は凍結。policy MLPのみ更新可能、prior/valueも凍結する。教師値はWDL実targetに使わない。
まず各gen3 seed1751401/2/3で新4ゲーム、各12/20/28ply最大3root。初期6plyランダム、その後固定policy greedy80%/上位8ランダム20%。完走した合法履歴を保存。固定幅4深さ6全root完成木、60秒は安全上限のみ、未完成ラベルは不採用。ノード上限は有限4分岐×最大6ply木。共通外生seed、軌跡はモデルに依存。
旧評価開始状態/元PPO訓練indexからrootを除外。先頭の空盤面等は生成に必要だがラベルにしない。診断教師とgreedy差が全体10%未満又は6plyラベルが24件未満なら学習せず終了。十分なら費用を見て有限PLANを固定。最大1学習方式、救済スイープなし。
