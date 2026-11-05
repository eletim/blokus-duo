# Root値補正（事前計画）
基点 d3c4698。低予算の推奨変更なし。教師/value/相手上位3応手は固定。
新ゲーム seed1030000以降、各ゲーム中盤2局面を収集。先頭80% train、10% dev、10% validation。更新seed1001/1002/1003は初期重みseed1000と全データを共有し、minibatch乱数のみ変更。
教師depth2を全heuristic上位128候補で計算、初期値上位4＋残候補から一様2候補をdepth4で完了評価。これは固定node型の同深度診断であり100ms教師そのものの成績・低予算実現の証拠ではない。選択偏りとrandom群を別記。教師値はroot視点の点数。未完了はNaN/mask、同値は許容集合。
第1段階: 新24ゲーム48局面でoracle補正による同候補集合内depth4 regretが改善するか。改善局面が10%以上なら学習pilotへ進む。代数的な再現だけで棋力向上としない。
第2段階: 合計240ゲーム480局面を上限、全盤面CNN＋候補特徴を用い、masked Huber lossでdepth4-depth2を回帰。3seed各1000/3000更新まで。devでzeroよりMSE/意思決定regretが改善しない場合も新validation実戦の小規模対照で確認するが、追加データ/更新はしない。
新validation seed1021000/16prefix、6/25ms、補正強度0.25/0.5/1、共通checkpointを選択（3seed平均得点差、同値なら低予算/弱補正/少更新）。計算節約のためdevでcheckpointと強度を上位2設定まで絞り、この絞り方はregret最小。主要gate: 3seed平均対旧value得点差>0、各seedが-2点より良い、zero対照平均差>0、推論平均<1ms、dev regretがzeroより改善。満たした時のみfresh seed1022000/96prefixで旧valueとzeroへ先後交換。未達ならpilot停止、例外は事前記録。
時間は合法手生成/入力作成/GPU同期含む。zero対照も同モデルを推論し補正強度0。terminalは補正禁止。試合ごとcold。全主学習GPU30分/全体2時間目安。主比較の計測と生成/監査は重ねない。

## 実装と再現
`CorrectionNet`は6x14x14全盤面→conv8ch→flatten32と状態49、候補39特徴を結合。ゼロ初期化の出力ヘッドでroot視点の点数補正を回帰。深度の混在や単一正解classificationは使わない。`CorrectionAgent`は旧heuristic上位128を同順序で2ply評価、相手上位3・固定valueはそのまま。完了した候補だけ `v2 + alpha*delta` で最終選択する。確定終局値は補正しない。ゼロ対照は同じcheckpoint/forwardを実行しalpha=0。全候補最低数を要求せず、完了0なら従来heuristic先頭へfallback。
教師参照値は限定幅3のdepth4であり厳密解/真の勝率ではない。上限診断の候補集合は初期値上位4+ランダム2の共通maskであり、全合法手のoracleではない。ランダム2もheuristic上位128以内。未観測候補にも実戦ではモデルを適用するため、選択偏りによる外挿の限界がある。

```bash
# 以下は成果ファイルを同じ場所へ再生成するため、別cloneで実行する。
.venv/bin/python -m blokus.correction_data --games 240
for seed in 1001 1002 1003; do
 .venv/bin/python -m blokus.correction_train --seed "$seed" --output "artifacts/root-value-correction/seed-$seed"
done
.venv/bin/python -m blokus.correction_experiment validation
.venv/bin/python -m blokus.correction_experiment holdout  # gate未達時は拒否
.venv/bin/python -m blokus.correction_experiment play --seed 1001
.venv/bin/python scripts/check_correction_resume.py
.venv/bin/python scripts/audit_correction.py
.venv/bin/python -m pytest -q
```

ゲーム生成の旧6ms探索は壁時計制御なので、同seedでも別実行の源ゲームが完全一致する保証はない。保存したsource-gamesと教師ラベルは固定再生で検証できる。更新seed間で初期重み・データ・分割を共有、minibatch乱数だけを変える。再開には `--resume .../latest.pt --steps 総更新数`。保存はatomic replace、モデル/optimizer/全RNG/データSHA/source SHAを含む。3seedを独立データ標本と見なさず、CIは先後とseedをprefix内で平均してからbootstrapする。

pilot24ゲームは先頭24ゲームの予備診断であり、240ゲームと独立な追加標本とは数えない。主データには1件の重複rootがあり、分割をまたいだ漏洩を避け全体で先出のみ残した。検証はこの実装担当による再生・算術チェックで、秘書の独立実装監査とは区別する。
