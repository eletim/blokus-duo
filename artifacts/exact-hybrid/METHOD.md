# 低mobility終盤の厳密探索

初回開発で残駒数だけの厳密切替が少なかったため、初回holdoutを参照する前にPROGRESS.mdへ記録した追加候補。初回成果はcommit dd8e7b5 / artifacts/deep-searchに保存。

既存value重み・alpha=.5/gate24・engine・旧2plyを固定。残駒合計24以下で両者の現在合法手数を数え、合計が6または10以下のときだけ全合法手の厳密探索を試す。それ以外は、判定に使った時間を引いた残り予算を旧2plyへ渡す。低mobilityは探索の発動条件であり、探索中の枝を削る条件ではない。将来合法手が増えても全合法手を扱う。

対象rootではまず全合法root×相手heuristic上位3応手の2plyを完了させる。旧2plyと同じ同率候補の乱数選択を使う（許容差は得点単位へ変換）。その後、残り時間でTT付きalpha-betaを終局まで実行し、root全体を完了した場合だけ厳密解として採用する。中断時は完了済み2plyへfallback。最初の2plyも未完了の場合のみheuristic fallback。候補幅を絞った2plyを厳密解とは呼ばない。

新開発511000/32prefixでmobility6/10を25/100msの同予算旧2plyと比較し、平均得点差で閾値を選ぶ。6ms補助比較は511999/8prefix。新validation521000/32で主要予算を固定した後、fresh531000/96prefix・先後交換で旧2plyとの主比較と、初回validationで選ばれた探索方式との副比較を行う。副比較も同予算、再選択には使わない。初回holdout530000は選択に使わない。

追加学習0更新。全厳密解は各結果JSONのexact_certificatesへ保存。旧探索のnode数（生成child）と新探索のnode数（TT参照込み訪問）は定義が異なる。共通予算であって、実消費時間をpaddingした比較ではない。判定費用・超過率・平均/p95/max・厳密解率・完了深さを保存する。

```bash
# 新規checkout/出力未作成状態で有限run
.venv/bin/python scripts/run_exact_hybrid.py
# 保存した固定方式の対旧2ply再実行。再選択用ではない
.venv/bin/python -m blokus.deep_experiment tournament --spec artifacts/exact-hybrid/selection.json --prefix-file artifacts/exact-hybrid/holdout-prefixes.json --output artifacts/hybrid-replay.json
# 人間対AI
.venv/bin/python -m blokus.deep_experiment play --spec artifacts/exact-hybrid/selection.json
# 計測完了後に検証
.venv/bin/python scripts/audit_deep.py --root artifacts/exact-hybrid
.venv/bin/python -m pytest -q
```

時間依存の完了深さ・候補数により同seedの着手bit一致は保証しない。証明済み小終盤の値は独立全探索で一致を確認する。学習済み重みの改善とは主張しない。
