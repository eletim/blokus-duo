# 候補配置入力の対照実験（事前固定）
基点312e9dc。main変更なし。既存3600ゲームpoolのみ再利用し、各fold訓練は2400 nominal games。過去bank/holdoutを学習・選択に使わない。新bank360ゲームID200000–200359、seed1330000–1330359。pool/seat/fold層化、45/65%進行root、ゲーム/局面hash分離。教師depth4-depth2、固定value、width3、top4+random2ラベル、Huber、alpha .5は保持。教師値は真のゲーム価値ではない。
2表現: summary=従来6ch盤面CNN+39候補要約、cells=候補の占有セルを7ch目に追加して候補ごとCNN。共通重みは同一初期値、追加channelのみ新規。両者1000更新、AdamW lr.001 weight_decay0、batch32 root、既知候補のみ損失。3fold×更新seed1101/1102/1103。ゲーム分割/データ/初期共通部分/ミニバッチ乱数は共有。追加正則化・サイズ・checkpoint探索なし。
進行gate（前回と同じ、不変）: 各foldの3seed平均でraw MSEとalpha .5選択損失の双方がゼロ補正より5%以上改善、各seedのfold平均選択損失がゼロより改善、ゲーム単位でseed平均後bootstrap4000の両改善95%CI下限>0。1000更新のみ判定。失敗なら対戦なし。raw MSEにはalphaを掛けない。条件を途中変更しない。
訓練GPU主計算合計1200秒以下、今回90分目安。通過時のみ新validation32prefix、その後96prefix先後交換25msへ進む（全体2h）。入力衝突回帰、勾配、CUDA別process40 vs20+20、教師再生、分割/算術監査、候補batchメモリ/遅延を記録する。
