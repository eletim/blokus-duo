# 広い2plyの後のroot追加探索

固定checkpoint: artifacts/value-search/compact/seed-201/predictive-best.pt、alpha=0.5、gate=24。追加学習・重み変更なし。branch research/minimax-refinement、基点eb7dbd4。main推奨は変更しない。

rootは旧方式と同じheuristic上位128、初期相手応手は上位3。各rootの初期評価は葉のroot視点値のminimum。初期評価を全候補終えるまでは追加処理しない。priorityは初期評価値の高いroot順、uniformは元のheuristic順に各root一回だけ深さ4へ更新。追加は内部各節点上位3、alpha-beta negamax。controlは同一初期処理で追加なし。このuniformは各root一回の順次配分であり、期限まで全候補同量を保証しない。

negamaxの各返値はその局面の手番側視点、親は-child、rootは最大化し相手は最小化。passでも実際の手番交替に従い符号反転。同じ色が連続で置く場面も相手pass節点を省略しない。終局はボーナス込み実得点差で上書き。初期評価と深さ4評価は厳密解でない。探索幅外の相手応手は対象外。混合深さの値比較にはホライズンによる偏りが残る。

deadlineまでに追加root全体が終わらなければそのrootは初期値のまま。部分探索値をpublishせず、初期未評価rootは候補にしない。初期root皆無ならheuristic先頭、合法手なしはpass。TTなし、試合・phase間の評価順依存キャッシュなし。費用予測は試合ごとに初期化、試合内のみ更新。予測1.5倍とmax(1ms,予算5%)余裕。OS・GPU・合法手列挙のatomic処理にhard deadline保証はない。

固定node診断は2048上限、同一seedで2回一致をassert。nodeは局面訪問（深さ1一括評価の子も加算）。予算に収まらない一括評価を開始しないため少量未消費あり。12独立randomゲームseed820000+から中盤1・終盤1、同ゲーム2局面は独立24ゲームと数えない。小終盤は別座標checker・独立遷移・独立score全探索oracleと比較。標本は易しい終盤に偏り、完全探索が中盤でも可能という証拠ではない。

validation: seed821000,16prefix・先後交換。priorityと旧2ply、uniform、controlを比較。pilotの追加完了率50%以上、対旧平均得点差>=-1、95%CI上限>=0、超過率10%未満が進行基準。失敗なら一度だけ事前記録した修正、新validation821100。進める場合fresh822000,96prefix固定後評価。旧holdoutは使用しない。CIはprefixごとの2試合を1clusterとしてbootstrap。小validationは暫定であり同等性証明ではない。

再現:
```
.venv/bin/python -m blokus.refinement_experiment pilot
.venv/bin/python -m blokus.refinement_experiment validation --budget 100
.venv/bin/python -m blokus.refinement_experiment holdout
.venv/bin/python scripts/audit_refinement.py
.venv/bin/python -m pytest -q
.venv/bin/python -m blokus.refinement_experiment play --budget 100
```
既存成果を保全してから再実行すること。時間測定中は他の生成・監査・テストを走らせない。コードは未知の評価結果を自動で選択せず、selection.jsonは条件確認後に固定する。
