# 教師再確認とroot policy蒸留

基点12a9963、research/root-policy-distillation。既存value checkpoint/alpha=.5/gate24を固定。教師のroot128・内部幅3・priority100ms・時間余裕・cold resetも固定。fresh920000/96prefixで同100ms旧2plyへ再確認した。同一実装の独立標本であり、秘書による独立実装監査ではない。再確認の成績による教師調整なし。

## 収集・ラベル

教師再確認の時間計測を終了してから300独立ゲーム(seed930000+)を生成。固定value/2ply6ms対heuristic、各手15%randomで変化を入れ、ゲーム中の30/45/60/75%地点から最大4rootを採取。前240ゲームtrain、次30dev、最後30supervised validation。対局によるvalidation921000/16prefixは別途用意した。

教師実装のtop-level search完了を記録するラッパーで、選択・初期値・最終値・差・完了深度・候補hash・実時間を保存。教師設定/重み/探索値計算は変えていないが、完了記録の小さなオーバーヘッドは100msに含まれる。depthは完了した要求深度であり、途中終局なら実際の木は短い。時間制御の完了範囲は同seedでも変わり得るため、再生は保存された候補・完了深度に固定して値/選択を照合する。

1192root中追加完了587、選択と同じ深度の比較候補が2個以上ある571rootを採用（train463/dev59/validation49）。**教師選択を模倣する分類ラベルであり、真の最善手とは呼ばない。** loss maskは選択候補と同じ完了深度だけ。depth2と4の値回帰や相互の負例化、未評価候補を負例にすることはしない。平均比較候補18.29。教師に追加完了がないrootは学習に混ぜない。

候補は元のheuristic上位128で固定。全候補に対するtop1/top5/平均rankと、同深度maskのNLLを区別する。maskの狭さで一致率を見かけ上高めない。ゲーム分割に加え採用rootのhash重複0を確認。全収集rootには未採用7種類のsplit間重複があり、隠さず監査JSONへ記録した。

## 学生と対照

root特徴39（既存12+駒one-hot21+座標平均/範囲6）とstate49。盤面全体は入力せず、固定valueは従来のまま。小MLPのresidualを従来heuristic priorへ加算し、同一128候補内の順序だけを変更する。相手応手上位3と2plyの最終値による選択は変更しない。

未学習residual最終層はzeroで、heuristicと同じ順位になる。未学習モデルにも入力構築・GPU転送・推論・cpu()による同期費用を課す。heuristic順対照はその推論を省く。全方式の手番時計は合法手生成より前に開始。CPU/GPU初期化・モデルwarmupは手番時計外。

3更新seed901/902/903は初期重み(seed900)、データ、候補、分割を共有。ミニバッチ抽出とdropout乱数のみ独立。各3000更新、batch32、Adam lr=.001、CE、gradient clip5。1000/3000checkpointを比較した。平均再利用は1000更新でtrain root約69回、3000で約207回であり、小pilotの過適合を調べる設定。3000更新でdevが悪化したためデータ/更新数の追加はしない。GPU主学習は3seed計約8.05秒。

## 選択・事前目安からの変更

当初のpilot目安「dev top1が未学習より5pt改善」は、3seed平均では不達（1000更新31.64%対28.81%、+2.82pt）。推論中央値約0.28msは25ms予算の10%未満。目安を満たしたとは記載しない。

小規模対局validationでは25ms/1000が全3seedで正、平均+1.156点（CIは0を含む）だった。この実戦信号を根拠に、元依頼の「教師一致率だけで採用しない」に沿って、追加学習・データ増量には進まず、予定済みfresh棋力評価だけを実施する判断へ変更した。**holdoutを見る前**にPROGRESSとselection.jsonへ記録した事前簡易gateからの変更である。

予算6/25ms×checkpoint1000/3000を全3seed平均得点差で選択。25ms/1000を全seed共通に固定し、seedを選ばない。fresh922000/96prefixの先後交換で旧2ply・未学習順位モデルと比較。prefixごとに先後2局×3更新seedを平均し、その96clusterをbootstrapする。同データ3seedを独立288prefixと数えない。学習重み・設定のholdout再選択は禁止。

## 再現

プロジェクトの`.venv`を使用。既存artifactsを別場所に保全してから生成・対局コマンドを再実行すること。保存data.npzからの学習再現を優先し、時間制御収集の完全同一を保証しない。

```bash
cd /home/eletim/blokus-duo-ai
.venv/bin/python -m blokus.root_experiment teacher
.venv/bin/python -m blokus.root_experiment collect
.venv/bin/python -m blokus.root_train --output artifacts/root-policy-distillation/seed-901 --seed 901 --steps 3000
# 902,903も同じコマンドでseed/outputだけ変更
.venv/bin/python -m blokus.root_experiment validation
.venv/bin/python -m blokus.root_experiment holdout
.venv/bin/python scripts/check_root_resume.py
.venv/bin/python scripts/audit_root_distillation.py
.venv/bin/python -m pytest -q
.venv/bin/python -m blokus.root_experiment play --seed 901
```

再開例: `--output <再開先> --seed 901 --steps 3000 --resume <latest.pt>`。stepsは追加数でなく累積目標。CUDA別process40更新対20+20でmodel/optimizer/Python/NumPy/Torch/CUDA RNG一致を検査する。時間計測中に収集・学習・テスト・監査を重ねない。各runは有限、学習上限30分、今回全体目安2時間。
