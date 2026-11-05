# 固定重みの探索強化

基点a88905e。追加学習は0更新。既存推奨 `artifacts/value-search/compact/seed-201/predictive-best.pt` の重みを固定し、alpha=.5、gate24の同じ葉評価を使う。engineと旧BudgetAgentも変更しない。

## 初回比較

- deep: 残駒合計24以下でroot幅8・内部幅3、深さ2/4/6/8/10の反復深化。
- late: 残駒合計14以下でroot幅12・内部幅3、同じ反復深化。
- 対象外の局面は既存2plyへ残り時間を渡す。
- 残駒合計8以下かつroot合法手32以下では、最初の2ply反復完了後に全合法手を終局まで読む厳密探索を試す。期限内に完了したrootだけを厳密解と呼ぶ。

選択探索の各ノードはheuristicで固定した候補集合を使う。TT手はその集合内でだけ順序を変える。TTは盤面・残駒・実得点（最後mono bonusを含む）・手番・passesと深さ/完全探索modeを区別し、EXACT/LOWER/UPPER boundを保存する。期限超過例外が出た未完了ノードは保存しない。反復が未完了なら前の完了反復を採用、最初も未完了なら合法heuristic手へfallbackする。

passは合法手なし時だけ、1plyとしてturnを反転。相手がpassして同色が再び指す場合もnegamaxの符号反転を2回行う。終局はボーナス込み得点差で、学習推定値を使わない。小終盤の別実装全探索で値と最善手集合を照合する。

## 分割と時間

新開発seed510000の32prefixを両設定・25/100msで共通使用。設定は2予算の平均得点差で選び、新validation520000/32prefixで主要予算を固定する。6msは別開発510999/8prefixの補助比較。最後はfresh530000/96prefix・先後交換で選択方式と同予算旧2plyを比較。旧holdoutは選択に使わない。

時間は合法手生成・評価・探索を含む1手のwall time。共通上限でありpaddingした実時間一致ではない。GPU warm-upは対局開始前。原子的な処理が期限を超えることはあり、全超過・平均/p95/最大msを保存する。旧探索node数は生成child数、新探索はTT参照を含む訪問node数で、完全に同じ計数定義ではない。旧openingを利用する呼出と選択探索の深さを分ける。厳密解割合は全呼出・試行回数の両方を示す。

完了した厳密解はstate/視点付き得点/最善手を `exact_certificates` に保存する。将来の教師候補であり、今回は学習しない。監査は独立全探索で標本を再証明する。全対局着手再生、得点・先後・CI・時間・選択・hashも確認する。CIは2局を組にしたprefix bootstrapであり、個別対局を独立標本とはしない。

## 再現

```bash
# 新規出力先（同じseedの再実行は再現用であり再選択用ではない）
.venv/bin/python -m blokus.deep_experiment run --output artifacts/deep-replay
# 固定済み最終対戦の再実行
.venv/bin/python -m blokus.deep_experiment tournament --spec artifacts/deep-search/selection.json --prefix-file artifacts/deep-search/holdout-prefixes.json --output artifacts/deep-replay-result.json
# 人間対AI
.venv/bin/python -m blokus.deep_experiment play --spec artifacts/deep-search/selection.json
# 監査・テスト（計測中には重ねない）
.venv/bin/python scripts/audit_deep.py --root artifacts/deep-search
.venv/bin/python -m pytest -q
```

時間依存の完了深さは環境負荷で変動し、同seedでも着手と勝敗のbit再現は保証しない。全ログと確定prefix、重みhashを保存する。TTは各Agent内で局をまたいで保持するが、意味状態keyが一致したものにしか適用しない。各方式は別TT、上限100,000entryで消去する。相手・seatの乱数seedを共通化する。
