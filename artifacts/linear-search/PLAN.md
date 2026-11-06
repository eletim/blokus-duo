# 固定非NN探索：幅と深さ（事前計画）
基点ec64667。old-3101/2/3の1024iteration方策と元24係数価値を固定。cb25af9の価値は不使用。学習0、main/Web変更なし。前回gate/結果は変更しない。

A: root6/reply3/depth2。B: 完成Aの後root24/reply8/depth2。C: 完成Aの後root6/reply3/depth4。追加段階は全root完成時のみ採用。未完了は完成A、それも未完了ならgreedy。rootにgreedyを必ず含める。内部枝は方策上位reply、枝を省くので完全探索ではない。枝刈り/TTは使わない。各着手内のみ費用予測、試合を跨ぐ状態なし。

費用pilot: 新独立24ゲーム、各8/18/28plyの保存局面（終局前のみ）。同じ局面で6x3/24x3/6x8/24x8の2plyを比較しroot/replyの影響を分離。値の優劣は元価値に対してのみ。B/Cの追加段階が各90%以上完成する最小の25/100msを採用。100msでも未達なら100msを固定し、該当条件はvalidation gateで完了率判定。幅は24/8から変更しない。

validationは新32prefix、3seed×先後の192局/組。A/B/C対対応gen3、B/A、C/A、旧6ms対gen3。評価前後に全訓練訪問状態・過去開始状態と照合。探索的進行gate: B/Cそれぞれ対A point>=.50、対gen3 point>=A/gen3-.02、追加段階完成率>=.80、greedy fallback<.10。通過候補をgen3 point、対A point、低実ms、B優先の辞書順で最大1本固定。僅差順位は強化成功ではない。

通過時だけ新128prefix×3seed×先後、選択候補/A と 選択候補/gen3 の2主比較。pointの通常95%とBonferroni family2の97.5%区間をprefix（6局）cluster bootstrap10000回で報告。得点差95%CIも報告。旧6msはvalidationの実用対照。gen3は1751401/2/3対応、探索なし。思考上限が同じでも実wall同等とは呼ばない。

計算は診断・評価合計2700秒以内を目安に固定標本で実施、途中成績で標本延長なし。モデル/seed再選択なし。計測中に監査/学習を重ねない。冷初回・warm分離、legal/特徴/value/探索費用、深さ/幅/完成率/fallback/時間超過を記録。native scanは合法列挙+特徴の融合処理なので個別microbenchは加算内訳とは呼ばない。

実装中修正（validation前）: 最初のpilot保存局面が元stateのlist参照を保持し進行で変化した不具合を検出。invalidファイルとして保全し速度/幅判断から除外。deepcopyと保存hash一致assertへ修正。同一の事前devゲームで再生成し、計画の予算決定規則を適用する。無効pilotの25ms判断は採用しない。
