# ゲーム数による補正学習曲線（事前計画）
基点c46a180。main変更なし。固定value SHA8a5795a2...、depth4-depth2教師（相手幅3）、最終root値補正alpha0.5を維持。教師は真のゲーム価値/厳密解ではない。

## データ・分割
新seed1130000+indexの入れ子pool 360/1440/3600ゲーム。各ゲーム2root（進行度45%/65%）。固定評価bank360ゲームはseed1230000+index。pool/bankとも6ゲームblockに相手3種(random/heuristic/既存value6ms)×先後2を均等配置、block番号mod3でfoldを固定。fold fの訓練はpoolのfold!=f、評価はbankのfold==f。これにより実訓練ゲーム数240/960/2400、各fold固定120評価ゲーム。3分割の訓練集合は重なるので独立実験と数えない。全体のユニーク元ゲーム数と、実際にrootを提供したゲーム数を区別する。root数だけ増やす比較ではない。
候補は従来通りheuristic上位128の初期値上位4＋残りから一様2を同深度評価。重複rootはbankを優先して一度だけ保存。過去holdoutは一切読まない。sourceゲームが時間制御のため生成はbit再現ではなく保存手順の再生で検証する。

## 学習曲線
条件は同じCorrectionNetでAdamW weight_decay=0 / 0.1の2つだけ。対称変換は今回は使わない。共通初期重みseed1100、更新seed1101/1102/1103。全条件・seed・サイズで同じ初期重み、各分割の同じデータを共有し、minibatch順だけ変更。損失masked Huber固定、lr0.001/batch32、各3000更新。1000/3000のtrain/test MSEと候補選択損失を記録するが、gateと設定選択には事前固定1000更新のみを使用（3000は過学習診断）。固定更新数の曲線であり各サイズの実効epoch数も記録する。
240→960で、少なくとも一方の条件が固定bank平均MSEとregretの両方で悪化せず一方が5%以上改善した場合のみ2400を生成する。そうでなければ打切り。GPU主学習は全run合計1800秒以内、全体2時間目安。

## 対戦gate（覆さない）
最大到達サイズの1000更新で、同じ条件が全3foldでzeroに対して平均MSEとregretを各5%以上改善、各更新seedをfold平均したregretも改善、全bankのゲームcluster bootstrapでMSE改善量・regret改善量の95%CI下端>0をすべて満たした時のみ対戦へ進む。foldと更新seedの変動を分けて報告する。未達なら対戦validation/holdoutは作らない。
通過時のみ新validation seed1121000/32prefix、共通25msで合格条件から全seed共通設定を選ぶ（foldごと3モデルを選ぶのではなく、全poolへ同条件1000更新で3seed再学習して対戦）。対戦validationは平均対旧>0かつ対zero>0が必要。設定を固定しfresh1122000/96prefixで先後交換・対旧/zeroを行う。主比較の時間計測に学習/監査を重ねない。

## 選択偏り診断
固定bankから層化した36局面を使用。事前固定seed1101/fold0の各条件1000更新モデルの全128候補からの選択、zero選択、既存のdepth4教師選択、追加ランダム候補を共通depth4で評価。未観測の学生選択をzero正解にしない。これは選択後の診断で、gate再調整には使わない。

## 前報告の再開表記訂正
旧scripts/check_correction_resume.pyとcuda-resume.logはfull40、part20、resumeの目標総step40を実行している。正確には「40更新連続 vs 20更新+保存再開後追加20更新（合計40）」であり20+追加40=60ではない。前報告は表記が曖昧だった。今回は前半/追加/合計を別フィールドで保存する。

## 有効ゲーム数・不確実性の読み方
予定数は源ゲームIDの数。指定進行度に合法候補が2つ未満ならrootを保存しないため、240条件の実際の訓練寄与ゲームは237/239/238、固定評価は119/117/120（合計356）となった。予定数を実データ数と偽らず、各サイズの有効ゲーム・root数をdata-summaryへ記録する。source IDと完全手順が異なるゲーム数もauditで区別する。
ゲームCIは同じゲーム内のrootと3更新seedを平均した後のbootstrapで、学習済みモデルに条件付けた不確実性。訓練集合の選び方や初期重みまで含む万能なCIではない。fold別の範囲と更新seed別の範囲を別に示す。3foldは訓練集合が重なり、更新seedも同じ初期重みなので独立な9標本とは扱わない。

## 再現
保存済み成果を再監査・再集計する手順。生成は保存済みゲームを飛ばす追記方式、学習gridは完了resultのあるrunを飛ばす。既存成果を上書きせず固定データで再学習するには、下記の別出力先を指定する。新規に全データを生成する場合は専用cloneで成果ディレクトリを別名へ退避してから生成する。時間制御の源対局はbit再現を保証しない。
```bash
uv pip install --python .venv/bin/python matplotlib==3.11.2
.venv/bin/python -m blokus.scaling_data --size 240
.venv/bin/python -m blokus.scaling_experiment --size 240
.venv/bin/python -m blokus.scaling_data --size 960
.venv/bin/python -m blokus.scaling_experiment --size 960
# extension-gate.jsonのextend=trueの場合だけ:
.venv/bin/python -m blokus.scaling_data --size 2400
.venv/bin/python -m blokus.scaling_experiment --size 2400
# decision.jsonでadvance=falseなら対戦へ進まない
.venv/bin/python scripts/check_scaling_resume.py
.venv/bin/python scripts/diagnose_scaling_bias.py 2400
.venv/bin/python scripts/audit_scaling.py
.venv/bin/python scripts/report_scaling.py
.venv/bin/python -m pytest -q
```
主学習上限は全grid合計1800秒、1run40秒。途中停止は完了と見なさずassertで停止する。保存再開の `--steps` は追加更新数ではなく目標総更新数。全runのモデルhash・config・ゲーム別損失を残し、代表fold0/seed1101のみ1000/3000checkpointを保持する。各seedの全checkpointを大量複製しない。

## 生成gateの例外（960完了後・2400生成前に記録）
`extension-gate.json`はfalseのまま保持。960ではMSEが240より15–17%改善し、選択損失のみ微増した。これは依頼の「悪化のみなら早期打切り可」の条件には当たらず、計算予算にも余裕があるため、依頼された2400点を測定する。独自追加gateの「両指標非悪化」が依頼より厳しすぎた点を訂正し、`extension-exception.json`で生成のみ許可した。これは手続き上の例外であり隠さない。対戦gate（1000更新固定・各fold5%以上・ゲームCI下端正・各更新seedの改善）は一切変更しない。

指標の区別: MSEは生の補正出力とdepth4-depth2教師補正の差（等価にv2+生出力とv4の差）。regretは実戦と同じalpha0.5を適用した最終候補選択で計算する。MSEをalpha0.5適用後の校正誤差と混同しない。いずれも限定幅教師との一致であり、真のゲーム結果に対する誤差ではない。

固定データから代表モデルを再学習するCLI:
```bash
.venv/bin/python -m blokus.scaling_train --size 2400 --fold 0 --seed 1101 --condition plain --steps 1000 --output /tmp/blokus-scaling-replay-1101
```
result.jsonの1000更新model_sha256を保存済みn2400-f0-s1101-plain/result.jsonの1000更新と比較できる。代表保存checkpointは研究診断用で、推奨対戦モデルではない。
