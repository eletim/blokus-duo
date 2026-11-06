# 葉value専用経路と完成4ply後の拡幅
2026-11-07。基点99cb70c。main/Web/Frozen gen3を変更しない。追加学習0。

試験前の数値条件: Python/native観測のboard/stateはbit一致、専用valueと通常forward valueはCPU atol1e-7/rtol0、同じ完成木の値atol1e-7、最善手一致（同値の場合は同じtie順序）。超過時は原因と差を保存し、黙って緩めない。初期4/6ch、3seed、序中終盤/強制pass/terminal/bonusを検証する。valueはconv/global_netとstate49だけを使い、候補入力依存はない。state49内の残駒・面積・frontier・score・passesは既存encodeをそのまま使用する。frontierは合法接続可能点で、これを省略しない。

A=現旧SearchAgent width4/depth2→4。E=同条件で葉の全候補packとpolicy forwardのみ省略。D=Eのwidth4/depth4完成後だけ同じdepth4のroot幅を段階拡大する。internal幅4固定。phaseごと全root完成時のみ更新、timeoutは最後の完成stageへ戻る。Dだけact内の完全値memoを再利用、keyはboards/remaining/last/turn/passes/depth/internal幅。候補列を決めるpolicyは固定モデル・幅・順序、root幅が変わってもchild部分木の候補条件は同じ。部分値/alpha-beta boundは保存しない。actをまたぐcacheなし。greedyを保持しroot集合は入れ子。

開発pilotでroot段階8→12の費用を測り、一度固定してからvalidation開始。主500ms、補助2000ms。nativeは内部観測の別速度要因として同じ保存局面でPythonと分離計測する。主対戦は本番と同じCPU/Python backend。純Python fallbackを残す。

validation/確認の局面数・gate・標本上限は費用pilot後、対戦成績を見る前に本PLANへ追記固定する。3seed全保持、勝ち1/引分.5のpoint、得点差、prefix単位CI。確認は成功した固定Dのみ新32prefix以上、旧/単体の2比較補正CI。前回8prefixは使用しない。全計測90分上限、未完なら標本を勝手に縮小しない。人間棋譜未取得。

## 費用pilot後の対戦計画固定（validation前）
root段階は4→8→12、全てinternal4/depth4、まず4幅depth2→4を必須。Python同完成木2000ms診断でA320.3ms/E97.9ms、nativeは別要因としてA92.1/E39.6ms。実行可能量は次に固定。
新validation500: 16prefix×先後×3seed、A-policy/E-A/D-A/D-E/D-policyの5組（480局）。新validation2000: 8prefix×先後×3seed、D-A（48局）、補助診断で主予算選択には使わない。
Dの進行gate: 500msでD-A、D-policyともpoint>=.50かつ平均得点差>=0、D-Eの通常95%point上限>=.50（明確な退行を除外する探索的条件で非劣性証明ではない）、p99<=750ms、非passで幅8以上の完成>=20%。Eは固定木の数値/着手一致と交互pilotの明確な短縮があれば棋力と別の最小統合候補を作る。
D gate通過時のみ新confirmation500=32prefix×先後×3seedでD-A/D-policyの2組（384局）。通常95%CIと2比較Bonferroni97.5%CIをprefix単位（両席/固定3seedを同クラスタ）で報告。両主比較の補正point下限>.50、得点差>0をDの改善確認条件とする。未達は未確定/不採用、seed/幅/時間の再調整なし。標本途中変更なし。モデルは全3本同じgen3固定、追加学習0。
計測は単独実行、runはtimeout5000秒（83.3分）、pilot等を合わせ90分以内。aux2000で良くても主500msの失敗を救済しない。成績確認後の新規条件追加はしない。
