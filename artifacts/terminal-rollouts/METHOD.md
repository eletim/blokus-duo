# 終局rolloutと逐次配分

基点376dce2。学習なし、旧value/2ply・engine・既存成果は変更しない。新モジュールrollout_search、rollout_experimentとpilot/監査スクリプトを追加した。

## 候補とcontinuation

最大8root候補。開発比較はheuristic上位8と、上位4＋駒ID/anchor象限の組が異なる候補を補う多様化集合。新しい12ゲームの中盤で、両配分×3sample seedの選択手を候補集合の和集合上の参照値で評価し、ゲーム単位平均regretの小さい**上位8**を選択した。固定された候補集合の外の良手は選べない。

root着手後、両色ともheuristic上位3から一様ランダムに終局まで着手する。合法手なしのみpass、最終得点差には全置き/最後monoのボーナスを含む。ラベル視点はrootの手番。盤面をcloneし、入力状態や対局RNGを変えない。continuationにCUDA推論は使わずCPUのみ。

共通乱数はsample seedを全rootで共有し、各continuation plyでpassを含め一度random()を消費する。同じquantileを各状態の上位3へ写像する。同seedが同一の盤面意味/着手を保証するものではなく、分岐後に対応は弱くなる。分散削減効果そのものを独立乱数対照で証明したとは主張しない。

平均rollout値は固定の確率的continuation方策に対する期待値であり、相手が候補ごとに最も強い応手へ適応するminimax値ではない。終局まで完走していても「厳密なゲーム値」にはならない。

## 配分・deadline・fallback

均等配分は8候補×4完走。逐次配分は8候補×2=16、上位4候補×2追加=8、上位2候補×4追加=8、計32。候補が8未満の場合は同じ半減規則で総上限32を守り、round単位の端数を消費しない場合がある。同率は元の候補順で決める。

比較の更新単位は全active候補の1round。時間内に各rolloutが完走していても、そのround全体が完了しなければ追加標本を選択へ反映しない。途中中断/期限後の完走はラベルとして返さず、部分roundの完了標本も破棄する。completedは期限内に完了した個別rollout、committedは比較に採用した完了標本、両者の差は破棄した部分round。

全rootに最低1標本が揃わなければheuristic先頭へfallback。旧2plyへ戻るfallbackではない。最初のroundが揃った後は最後に完了したroundの比較を保持する。clock/deadlineは合法手生成・特徴計算の前後と終局後に検査する。単一操作は強制中断できず、超過が発生する。超過率と最大msを保存し、hard real-timeとは主張しない。

候補生成を含めて予算を開始する。試合ごとにrollout seed/手数counterを初期化し、旧2plyの適応cost履歴もreset。TTなし。統計だけ累積。旧方式のモデル初期化/warmupは両者の思考時間外。色に紐付くseedを先後交換で共有する。

## pilotと参照

新規独立12ゲームseed720000+game。ゲーム方策はheuristic上位8からランダム。各ゲーム中盤1、小終盤1。sample seed731000/732000/733000＋game×100。参照は別seed750000＋game×10000＋sampleの64完走/候補平均。有限標本であり誤差を含む。中盤で真の最善手損失を測ったとは言わない。

小終盤は別実装の全合法minimaxで厳密値を調べる。20000node上限を超えた局面はcappedと明示し、厳密値/損失の集計から除外。終局直前の少合法手状態への採取偏りがある。

pilotの継続基準は事前にPROGRESSへ記録した。100msで中盤80%以上の全候補最低標本coverage、完走中央値16以上を要求。25msは1/12、100msは12/12・中央値20であり、100msを選択した。予算を増やした効果を探索改善と混同せず、主対照も100ms旧2plyとする。単一CPUプロセスのままで基準を満たしたため、未測定の並列化/GPUbatchを導入しない。

修正前pilotで配分数テストが予定16+8+8との不一致(8+12+12)を検出した。開始直後の実戦runを停止し修正。pre-fix-pilotへ結果を隔離、選択や最終比較には使用しない。修正後pilot全体を再実行し、独立oracleの上限到達は未証明として処理する。新規runとpilotの途中再開では既存完了結果を保全する。

## 選択・評価

新devで候補集合と予算を固定、新validation721000/32prefixでuniform/halvingを同予算旧2plyと比較し平均得点差で方式を選ぶ。selection保存後、fresh722000/96prefixで先後交換。主比較は選択方式対旧2ply、副比較はhalving対uniform（同じ候補集合・100ms）であり、配分効果の対照。副比較による方式再選択は行わない。旧holdoutは不使用。

区間は先後2局を1prefix単位にしたbootstrap4000回。固定rollout診断の3seedは同一ゲーム内で平均し、ゲームをbootstrap単位にする。32rootを32独立ゲームと見なさない。時間計測中に生成・監査・テストを重ねない。

## 再現

```bash
# 新規出力状態で有限pilot。途中失敗は完了済み状態/時間測定/診断から再開
.venv/bin/python scripts/pilot_rollouts.py
# pilot合格後、新validation選択→fresh holdoutと配分直接対照
.venv/bin/python -m blokus.rollout_experiment run
# 固定済み選択の再対戦は別output。結果を再選択に使わない
.venv/bin/python -m blokus.rollout_experiment tournament --prefix-file artifacts/terminal-rollouts/holdout-prefixes.json --output artifacts/rollout-replay.json
.venv/bin/python -m blokus.rollout_experiment play
# 時間計測完了後
.venv/bin/python -m pytest -q
.venv/bin/python scripts/audit_rollouts.py
```

既存の最終成果がある場合runは拒否する。再現用checkoutでは成果を別名へ保全して出力先を空ける。設定・source hash・seed・checkpoint hash・生ラベル・着手列・思考時間を保存。巨大checkpointは新規作成しない。
