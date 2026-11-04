# 実探索に合わせたgroupと損失ablation

## 統合と実験範囲

origin/main=69065fcを確認、前2段階1b5cf87/3dc3eb6との差分をレビューし全38テストPASS後にfast-forward。README冒頭を既存value推奨・paired-ranking改善未確認へ整理したeb96a90を通常pushしました。エンジンは変更せず、force pushもしていません。新規実験はresearch/search-alignedです。

## 候補・応手・継続

新規base対局から残駒合計24以下のrootを2つ抽出。rootのheuristic上位4候補それぞれに、実探索と同じheuristic上位3応手を付けます。応手が不足するときはmask、合法手なしなら強制passを1応手として扱います。root候補は実探索の先頭4手に限られ、6msで実探索が4手以上を読む場合の全root分布を網羅してはいません。

継続poolは非学習評価と既存value（pooled seed201, alpha0.5, gate24）の固定4×3探索です。双方非学習、双方value、value対非学習、非学習対valueの4条件を周期的に実行し、5%のランダム合法手を含めて終局得点を得ます。生成時はCPU固定幅で、壁時計打切りによる非決定性を避けます。最終対戦評価は同一6msの可変root幅なので、この違いは残ります。

候補・応手間で継続seed/方策条件を共有し、各相対ply・手番で固定整数mixからPython RNGを再生成します。途中で乱数消費数が違っても次plyへずれを伝播しません。合法手集合が違うため同じ乱数が同じ行動に対応する保証はなく、終了時刻も異なります。元局面生成のseedと継続seedは分けて保存しています。

24groupのpilotで12継続を生成し、最初2/4/8継続から作る候補値を独立な末尾4継続の候補値と比較しました。各候補値は応手ごとの平均終局得点差/40の最小値です。4→8で候補MSEと意思決定損失がともに減る事前ルールに従い8継続を採用。基準の4継続もノイズを含み真値ではありません。2→4はpool条件のカバレッジも変わり、純粋な標本数効果ではありません。4→8は同じ4条件を1→2周します。

## 共通データの3損失

600base対局×2root、各4候補×最大3応手×8継続。全方式・全seedで同じデータと既存value初期checkpointを共有。元対局ID末尾0を予測診断へ回し、同一対局の2rootがtrain/診断に分かれないようにします。

- absolute: 有効な葉の絶対値smoothL1。
- difference: absolute + 0.5×候補値の全6対差分smoothL1。
- ranking: absolute + 0.1×候補差のsoft順位BCE（温度0.15）。

候補値はネット予測の上位3応手内最小値で、訓練targetも同じ構成。3seed501/502/503は更新乱数のseedであり、事前学習やデータ生成を独立に3回行ったものではありません。各seedの3損失間でminibatch・盤面対称変換乱数は同じです。各6000更新、32group/batch、AdamW lr0.0002、最大180秒/モデル、9モデルで最大1620秒。損失だけを比較するため更新数・初期重み・データを固定します。

400更新ごとに未使用対局groupの意思決定損失（真の候補ラベルの最大値−予測が選んだ候補のラベル）を測り、最小のcheckpointを選びます。同点では候補差MSE。これはrawネットの診断です。実際のalpha0.5混合・終局厳密評価を入れた意思決定損失/順位精度も別途actual-inference-diagnostics.jsonへ保存し、両者を混同しません。

## 方式選択と最終評価

新validation310000/32prefixにて3seed×非学習/既存valueの平均得点差で方式を1つ固定。alpha0.5/gate24は全方式共通。選択後にfresh holdout320000/96prefixを生成し、選択方式の3seedだけを先後交換して2相手と比較します（対照同士を加え7組1344局）。他方式のholdout評価による再選択はしません。

生成・特徴・GPU同期込み共通6ms、遅延子の評価は破棄。実消費時間をpaddingで揃えるものではなく、native演算の厳密な強制停止もできません。全decision時間、超過、探索nodes、prefixとcheckpointのSHA256を保存。prefix内2局を平均した区間と、共通prefix drawをseed間で共有したseed×prefix bootstrapを記録します。

## 再現

```bash
# 新しい出力先でpilot→品質監査→共通データと全実験
.venv/bin/python scripts/pilot_aligned.py --output artifacts/my-aligned/pilot.npz
.venv/bin/python scripts/check_aligned_data.py artifacts/my-aligned/pilot.npz
.venv/bin/python -m blokus.aligned_experiment run --output artifacts/my-aligned
# 個別学習・再開（stepsは追加更新数）
.venv/bin/python -m blokus.aligned_experiment train --data artifacts/search-aligned/groups.npz --mode difference --seed 501 --output artifacts/my-aligned-model --steps 6000 --seconds 180
.venv/bin/python -m blokus.aligned_experiment train --data artifacts/search-aligned/groups.npz --mode difference --seed 501 --output artifacts/my-aligned-model --steps 1000 --seconds 180 --resume artifacts/my-aligned-model/latest.pt
# 今回保存結果の監査と再開検証
.venv/bin/python scripts/check_aligned_data.py artifacts/search-aligned/groups.npz
.venv/bin/python scripts/audit_aligned.py
.venv/bin/python scripts/diagnose_aligned.py
.venv/bin/python scripts/check_aligned_resume.py
.venv/bin/python -m pytest -q
# 固定specs・prefixで再評価（壁時計枝数は揺れる）
.venv/bin/python -m blokus.value_experiment tournament --prefix-file artifacts/search-aligned/holdout-prefixes.json --specs artifacts/search-aligned/play-specs.json --output artifacts/aligned-replay.json --budget-ms 6
```

全モデルに再開用latestと選択用selectedだけを保存。configにはtorch、seed、source/data/初期checkpointのhash、更新数を保存します。新実験をmainへ自動統合はしません。秘書が結果を受領後に判断します。
