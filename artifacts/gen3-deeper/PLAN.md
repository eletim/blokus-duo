# Fixed width deeper search, 2026-11-07
E deployed separately (main8fb3e9d), Frozen models unchanged. Previous D/gate preserved, no additional root widening or learning. New research only.
Numeric tolerance before tests: absolute1e-7 relative0 for fixed-tree values, identical root tie ordering/selected moves. Fixed model seeds1751401/2/3, width4 at every ply. Iterations2→4→6; only complete all roots commits selection. Timeout retains completed4 (or2 if4 cannot complete). Alpha-beta/TT only reduces the fixed tree; TT key includes board/turn/remaining/last/pass/depth/width; bound EXACT/LOWER/UPPER, unfinished nodes never cached as exact. Cache cleared each act. No terminal special threshold or rule change.
Diagnostic: new independent random games, common top8 root candidates, compare fixed4ply width4 NN estimates with actual terminal WDL under fixed greedy policy continuation and fixed E continuation (fixed completed depth2 width4 to keep finite cost). Two distributions are policy-conditional outcomes, not minimax truth. All roots from one source game form one correlated cluster. No NN weights/labels learned. No human records available. Pre-fix diagnostic 6 source games, 1 midgame snapshot each, seed1751401, two repetitions under common continuation RNG. This is limited calibration evidence, not population accuracy.
Pilot: fresh saved opening/mid/end positions, both500/2000ms and E/deeper6, same candidates/weights. Freeze primary budget/sample counts/gate once cost is known, before any validation outcomes. Maximum timing90min, no parameter rescue. Confirmation only on gate pass, at least32 fresh prefixes paired×3seed. Main comparison deeper/E, policy auxiliary. Prefix-cluster CI; two confirmation comparisons use Bonferroni97.5%CI. No learning.

## Pilot後、validation前の固定
Pilot18測定/条件: E平均68.7ms、H6ply平均270.0ms、500msで6ply8/18。2000msはH平均336.9ms、6ply14/18。主500ms/maxdepth6を固定（短い実用上限内でも一部で完成、2000msは費用診断のみで追加対戦条件なし）。
新validation16prefix×先後×3seed、H/E・H/policyの2比較192局。H/Eがpoint>=.5かつ得点差>=0、H/policy通常95%point上限>=.5、H p99<=750ms、非pass6ply完成>=20%を確認進行gateとする。これは探索的選択で優越/非劣性証明ではない。
通過時のみ事前確保した新32prefix×先後×3seed、H/E・H/policyの384局。主H/Eの2比較補正97.5%point下限>.5かつ得点差>0で改善確認とする。policyは補助、単体への優越を述べる際も同じ補正を使用。全seed固定、途中標本変更なし。固定源モデルはmodels.json。
新bankは訓練index/過去開始/診断局面を除外。主run timeout4800秒、診断/pilotと合わせ90分以内。gate未達は追加条件なし。D不採用は変更しない。

## validation前のboundレビュー
TT窓を狭めた後に元窓でflag分類する危険をレビューで除去。boundは既存窓で確実にcutoffできる場合のみ利用し、窓を狭めない保守的方式へ修正。未完成ノードは保存しない。修正前pilotは別名で保持。修正後pilot500ms H271.1ms/6ply8of18、2000ms341.9ms/14of18で計画条件は不変。validation成績はまだないためgate/標本救済変更ではない。
診断は6元ゲーム、共通候補に対してgreedyと固定E2継続。固定E2は必ず同じ2ply木を完成、E4より強いと仮定しない。1局面で広幅推定値増、両継続WDLの増加0、残り5局面は推定選択不変。少数・相関・WDL離散性のため主因特定とはしない。
