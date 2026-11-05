# 固定valueでの探索強化：結果

既存2plyを安定して上回る証拠は得られなかった。推奨は既存value/2plyのままとし、main統合・推奨変更は秘書レビューへ委ねる。今回の変更は探索器だけで、学習更新は0。重みSHA256は `8a5795a2e4fe73e4055ce94d11f761fddbc7a66db7ec0863b39df8ba195bf2ad`。

## 独立最終評価

反復深化（残駒14以下、root12/応手3、2/4/6/8/10ply、少残駒は全合法厳密探索）は新validationで25msを固定。fresh530000・96prefix/192局では旧同予算2plyに94勝91敗7分、平均得点差+0.104、prefix単位95%bootstrap区間[-0.151,+0.339]。選択探索の平均完了深さ6.13、厳密解12/3009呼出。候補を削った探索は完全探索ではない。

初回holdoutの観測前に事前記録した追加案は、残駒24以下かつ両者合法手合計6以下だけ全合法厳密探索するhybrid。その他は旧2plyへ時間を渡し、対象局面でも旧同等2plyを先に完了して中断fallbackとする。新開発で閾値6、新validationで25msを固定した後、fresh531000・96prefix/192局で旧同予算2plyに90勝89敗13分、+0.214 [-0.068,+0.510]。初回反復深化方式との副比較も同じfreshprefix/25msで89勝92敗11分、−0.203 [-0.656,+0.161]。どちらのholdoutも再選択に使っていない。

hybrid主比較は厳密解96/2958呼出(3.25%)、98試行中96完了(97.96%)。残り2回は完了済み2plyへ戻り、初回2plyの未完了fallbackは0。発動の少なさが全局への寄与を制限している。厳密探索の値・着手・局面はexact_certificatesとして保存し、将来の教師に使えるが今回は学習していない。

## 実時間・制約

共通25msは上限目標であり、消費時間をpaddingした厳密な時間一致ではない。主比較の実平均はhybrid17.624ms/旧17.416ms、超過2/2958(0.068%)/旧1/2959(0.034%)、最大25.154/25.100ms。初回反復深化は18.772/17.329ms、超過9/3009(0.299%)/旧0/3008、最大41.297ms。副比較のhybridにも最大43.196msの外れ値がある。単一操作・GPU呼出・OS schedulingは途中で強制停止できないためhard real-time保証ではない。速度優位は確認していない。

TTは手番・passes・実得点を含む状態key、mode/深さ別、EXACT/LOWER/UPPER。全合法モードは終局までのalpha-beta/TTのみで枝幅制限なし。root全体が期限内に終わったときだけ厳密解とする。TTは試合をまたいで有効な値を再利用するため、計測はこの実装のwarm-cache条件。node定義は旧=生成child、新=TT参照を含む訪問で、単純なnode/sec比を速度優位の根拠にしない。時間依存の完了範囲により同seed着手bit一致は保証しない。

各prefixの先後2局を1単位として4000回bootstrap。prefix集合は開発/validation/holdoutで別seed・別hash、再生した中終盤状態もphase間完全一致0を確認（類似状態の独立性まで保証しない）。主比較・副比較とも区間は0を含み、単一機種・各96prefixの暫定結果。6msは8prefixの補助診断のみで推奨判断に用いない。実時間計測中に生成・テスト・監査を重ねていない。

## 全比較

勝/敗/分、得点差は左の新方式視点。ms/手は左/右。

|run/phase|条件|prefix|勝/敗/分|得点差 [95%CI]|実ms/手|厳密解/呼出|
|---|---|---:|---|---|---|---|
|deep-search/dev|deep-25|32|30/29/5|-0.547 [-1.641,+0.500]|20.538/17.448|5/997|
|deep-search/dev|deep-100|32|31/30/3|-0.641 [-2.016,+0.719]|72.698/52.231|6/993|
|deep-search/dev|late-25|32|29/28/7|+0.172 [-0.094,+0.453]|18.885/17.365|3/988|
|deep-search/dev|late-100|32|35/27/2|+0.406 [+0.031,+0.844]|56.223/51.911|8/1003|
|deep-search/six|late-6|8|5/10/1|-2.250 [-4.877,-0.062]|4.803/4.796|1/251|
|deep-search/validation|late-25|32|32/30/2|+0.641 [+0.125,+1.469]|18.642/17.268|3/1004|
|deep-search/validation|late-100|32|33/30/1|+0.578 [+0.219,+0.969]|56.115/52.578|1/994|
|deep-search/holdout|selected-vs-old|96|94/91/7|+0.104 [-0.151,+0.339]|18.772/17.329|12/3009|
|exact-hybrid/dev|mobility6-25|32|32/30/2|+0.516 [+0.000,+1.250]|17.968/17.819|27/987|
|exact-hybrid/dev|mobility6-100|32|30/30/4|+0.141 [+0.000,+0.312]|52.364/52.075|24/1017|
|exact-hybrid/dev|mobility10-25|32|32/30/2|+0.406 [+0.047,+0.828]|18.024/17.784|54/995|
|exact-hybrid/dev|mobility10-100|32|30/30/4|+0.125 [-0.094,+0.344]|52.498/51.986|54/1019|
|exact-hybrid/six|mobility6-6|8|8/8/0|-0.375 [-1.812,+1.312]|4.590/4.603|5/246|
|exact-hybrid/validation|mobility6-25|32|30/29/5|+0.156 [+0.000,+0.359]|17.890/17.688|34/984|
|exact-hybrid/validation|mobility6-100|32|32/32/0|+0.094 [+0.000,+0.234]|52.969/52.667|37/1003|
|exact-hybrid/holdout|selected-vs-old|96|90/89/13|+0.214 [-0.068,+0.510]|17.624/17.416|96/2958|
|exact-hybrid/holdout|selected-vs-earlier|96|89/92/11|-0.203 [-0.656,+0.161]|17.510/18.768|91/2972|

有限run計1376局、run実時間合計1203.17秒（初回562.76秒＋追加640.42秒）。実装・検証時間は含まない。GPU学習0秒/0更新、固定valueのCUDA推論を使用。追加checkpoint保存なし。全53テスト7.04秒PASS、CLI人間対AIを自動入力して終局確認。全対局再生・得点/勝敗/CI/時間/固定重みhash・開発/validation選択監査PASS。独立座標checker＋別遷移/得点実装の全探索で13小終盤の値/全最善手集合一致、保存証明は各run12局面標本を再証明（全証明の独立再探索ではない）。初回37、追加277のunique厳密解局面について同一hashの値の矛盾なし。

## 原因と次案

深さを増やしても上位候補に最善手が含まれる保証はなく、静的葉valueの誤差も残る。広いrootを狭める損失と未完了反復のコストが深さの利益を打ち消した可能性がある。これは原因仮説であり、今回の対戦だけで因果は確定できない。低mobility厳密探索は発動率を改善したが、手の約3%に限られ全局得点差は小さい。

次の候補は、(1)固定node予算で候補幅と深さを切り分ける診断、(2)証明局面で旧評価が誤るroot候補と局面属性を分析、(3)その分析に基づく厳密探索の予算配分。ただしholdoutで条件を調整せず、新開発/validationを準備する。今回の有限runは完了し、追加探索・学習プロセスは残さない。

## 再現

[初回方式](../deep-search/METHOD.md)／[追加方式](METHOD.md)に実行手順とseedを記載。人間対研究AI:

```bash
.venv/bin/python -m blokus.deep_experiment play --spec artifacts/exact-hybrid/selection.json
.venv/bin/python -m blokus.deep_experiment tournament --spec artifacts/exact-hybrid/selection.json --prefix-file artifacts/exact-hybrid/holdout-prefixes.json --output artifacts/hybrid-replay.json
.venv/bin/python scripts/audit_deep.py --root artifacts/deep-search
.venv/bin/python scripts/audit_deep.py --root artifacts/exact-hybrid
.venv/bin/python -m pytest -q
```

研究モデルの試用であり、既存推奨の置換ではない。再対戦の時間依存結果を使って保存済み方式を再選択しない。
