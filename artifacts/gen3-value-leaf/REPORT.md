# 葉value専用経路Eと完成4ply後拡幅D

**結論: Eを同一完成木の処理高速化として最小統合候補化。Dは旧探索への改善gate未達で不採用、独立確認は実施しない。**

追加学習0。Frozen gen3全3seed（1751401/2/3）とmain/Webは変更していない。人間棋譜は未取得で、人間より強いとは主張しない。

## 固定設計と等価性

A=既存幅4/深さ2→4。E=同じ木で葉のpolicy候補生成/forwardを省略。D=Eの幅4/depth4完成後、内部幅4のままroot8→12へ拡幅。各段階は全root同深度完成後だけ採用する。途中打切りは最後の完成段階へ戻り、greedyを候補から落とさない。Dの完成値cacheは1着手内のみ、状態/手番/残駒/last/pass/深さ/内部候補幅を区別し未完成値を保存しない。

value入力は既存board4/6chとstate49（残駒・面積・frontier・両者得点・pass）。候補policy特徴への依存はない。事前atol1e-7/rtol0、1188値比較の最大差0、102完成木の着手一致。CPU/Python/native、4/6ch、序中終盤/pass/bonusを含む。同一完成木の結果と、deadlineにより完成深さが変わる実対戦着手は区別する。

## 処理費用pilot

8保存局面・2反復をA/E/D順と逆順で交互測定、初期化除外。2000msで同じ4ply完成木のPython A320.3→E97.9ms（約3.27倍）、native A92.1→E39.6ms（約2.33倍）。native自体の効果は別要因。Dは追加rootを評価するためEより実消費が増える。小pilot分布での測定であり一般速度保証/棋力改善ではない。全計測192件はpilot.json。

## 新validation（選択用）

新16prefix×先後×固定3seed、各比較96局、5比較480局。pointは勝1/分0.5。両席・3seedを同じprefixクラスタとして20,000回bootstrap。以下は探索的通常95%CIで、多数validationからの優越証明ではない。

|比較|point %|95% CI %|平均得点差|得点差95% CI|実ms/手（左/右）|
|---|---:|---|---:|---|---|
|A-policy|68.23|61.98–73.96|+3.740|+2.156–+5.323|179.67/0.77|
|E-A|52.08|50.00–55.21|+0.302|-0.104–+0.854|65.15/180.36|
|D-A|47.92|39.06–55.73|-0.104|-2.115–+1.750|187.97/187.66|
|D-E|48.96|40.10–57.29|-0.292|-2.260–+1.583|188.01/67.67|
|D-policy|56.77|48.96–64.06|+0.812|-0.865–+2.552|183.37/0.80|

## 2000ms補助（主条件の救済選択には使わない）

新8prefix×先後×3seed=48局。

|比較|point %|95% CI %|平均得点差|得点差95% CI|実ms/手（左/右）|
|---|---:|---|---:|---|---|
|D-A|45.83|35.42–54.17|-1.750|-3.021–-0.146|193.66/193.34|

事前gate通過: **False**。PLAN.mdの条件は変更していない。

確認試験は未実施。gate未達を条件調整で救済せず、未使用確認bankを保持した。

## 計測と監査の読み方

各match JSONに全棋譜、全着手時間、完成depth/root幅、NN呼出数、cache hit、fallback、cold load/warmup、wallを保存。Aは旧実装の統計範囲（depth/nodes/fallback）であり記録されないNN回数を0回と解釈しない。forced passと時間切れfallbackを区別する。予算は同じ上限で実時間同一ではなく、単体gen3も同時間対照ではない。

モデルSHAはmodels.json、分離元はseparation-source.json。全訓練訪問状態indexと過去評価開始状態から開始bankを除外、評価後の訪問状態もaudit.jsonへ記録する。全棋譜を独立座標checkerで再生し合法配置・pass・終局得点を確認、集計とCIを再計算する。詳細はcoordinate-audit.json。前回B/C不採用の記録は保持。

## 再現

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python scripts/check_value_leaf.py
python scripts/pilot_value_leaf.py
# 元の分離index/過去bank参照が必要。既存bankは成績を見て再生成しない。
python scripts/run_value_leaf.py prepare
python scripts/run_value_leaf.py run
python scripts/run_value_leaf.py audit
python scripts/audit_value_leaf_records.py
python -m pytest -q
python -m blokus.ppo_value_leaf_play --method E --budget-ms 500 --human-seat 0
python -m blokus.ppo_value_leaf_play --method D --budget-ms 500 --human-seat 1
```

研究CLIのDは研究条件であり本番推薦とは別。保存済みmatchをrunは再利用するため、再測定時は既存結果を保全し別出力ディレクトリ/新計画を使う。

## 実費と判断

主対戦計測528局の合計wallは1949.98秒（32.50分）、pilot192計測と等価性検証を合わせても90分枠内。追加学習0。E-Aの実対戦は平均65.15対180.36ms/手だが、同じ完成木を揃えた速度比は上記pilotを使う。Eのpoint52.08%（95%50.00–55.21）、得点差+0.302はvalidation所見であり、独立棋力改善の証明にはしない。

D-Aの500ms実時間は187.97対187.66ms/手、point47.92%（95%39.06–55.73）、得点差−0.104。D-Eは48.96%、得点差−0.292。D-policyは56.77%（95%48.96–64.06）、得点差+0.8125。単体に対する期待だけで旧探索への未達を救済しない。2秒補助もD-A45.83%、得点差−1.75で、主条件の変更材料には使わない。

D-AのDは非pass1202/1202手で4ply完成、root幅8以上90.68%、最大500.51ms・超過1/1361手（pass含む）。E-AのEは非pass1203/1203手で4ply完成、最大170.61ms・超過0/1356手。Aは同比較で非pass1197手中13手が2plyまで。全比較の非pass時間切れfallbackは0。集計JSONのfallbackにはforced passが含まれるため、timing-detail.jsonで分離した。2秒補助Dも非pass586手すべて4ply、幅8以上91.13%、平均193.66ms。root12まで完成したら終了する固定設計なので、2秒を常に使い切る方式ではない。

**確認できた原因**: 葉で不要な候補列挙/候補pack/policy headを計算していたことは費用の大きな要因で、Eにより削減できた。Dは十分な頻度で拡幅を完成したのに、旧探索への改善を示さなかった。**未確認の仮説**: 幅を増やすと近似valueの順位誤差に影響される可能性、相手応手幅4や4ply先の評価限界が残る。今回の結果だけでいずれかを主因と断定しない。次に進む場合は固定継続対局でroot追加候補の評価誤差を診断する等、追加幅調整の反復より価値の妥当性検証を優先する。人間棋譜未取得で、人間への強さは不明。

## 検証と統合候補

- 研究CPU-only全テスト: **104 passed, 4 skipped**。小終盤oracle、終局/強制pass、元state不変、途中拡幅fallback、cache有無の完成木値/着手一致を含む。
- 全528棋譜の独立座標合法性/pass/得点再生・算術/CI再計算・モデルhash: PASS。
- 訓練index27,533,566状態、過去開始3598状態を参照して新bankを分離。評価後の全訪問状態も訓練重複0。確認用32prefixは未使用。
- 最小候補 `integration/gen3-value-leaf-E` はmain4f13460基点、追加2ファイル（helper/test）+検索葉変更+説明のみ。Dや前回B/C、研究履歴、モデルを含めない。候補commitは別verification manifestに記録。
- 新cloneのCPU-only検証: E関連10PASS、nativeビルド後全96PASS/4skip。未buildの初回は既存非NNのnative明示依存9件が失敗（84PASS/7skip）、Eの不一致ではない。CPU venvにcompiler/ziglangがなかったため既存開発venvのzigでC backendをビルドし、実行/テストはtorch2.7.1+cpuで行った。初回失敗ログも保持。E自体はPython-onlyで利用可能。
- CPU CLI `ppo_value_leaf_play --method E --budget-ms 500` は完走（−20対−29）、棋力評価用の追加標本ではない。cloneの隔離Webで人間先後の着手/API・gen3単体既定保持を確認。本番サービスには適用しない。

本番main・Frozenタグ/モデル・サービスは保持。Dは研究CLIのみ残し、Web採用候補は用意しない。Eの採否は秘書へ返す。
