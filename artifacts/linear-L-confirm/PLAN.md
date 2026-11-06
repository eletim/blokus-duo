# 固定線形Lの独立確認（2026-11-06、事前計画）
基点b32b89caf483e98d75ba553731dfaca8cd302915。旧Tのgate未達と結果は変更しない。これは32prefixで得た線形Lの探索的示唆を別の固定標本で確認する新計画であり、旧gate救済のholdoutではない。

L=artifacts/linear-tuple/L-3101/2/3.jsonの最終24係数、旧30係数方策を凍結。F=ec64667由来元混成3本。gen3=1751401/2/3を対応固定。全9モデルSHAとpolicy/value配列SHAを評価前に保存。追加学習・seed/checkpoint選択・条件変更なし。非NNは同じCompletedSearch、base6x3/2ply→target24x8/2ply、100ms上限。完成段階fallbackを維持。探索内部に試合をまたぐTT/費用予測状態はなく、各actの費用予測は初期化される。全合法手生成/特徴/valueを実時間に含む。

新prefix系列2480000000から既存訓練index・今回tuple新規全訪問状態・過去評価開始状態/非NN評価全訪問状態を除外し、256個を先に保存/hash固定。L/FとL/gen3各256prefix×先後交換×3seed=1536局、計3072局を全て実施。32prefix旧結果を混ぜない。途中の成績で標本数/モデルを変えない。主対戦60分上限目安、固定全量の見込み約25分。超過なら未完と報告し標本を勝手に縮小しない。対戦中は学習/監査/負荷診断なし。GPU学習なし。

point=(勝ち+引分*.5)/局数。prefixごと6局をclusterとしてbootstrap20000回、固定乱数2490000000。通常95%CIと、2主比較のBonferroni family2 point CI（各97.5%、分位.0125/.9875）を報告。得点差は通常95%CI。L/F補正下限>.5で元非NN超え、L/gen3補正下限>.5でこの固定3seed/評価分布のgen3超え。未達は未確定であり同等性証明ではない。seed別・席別point/得点差/95%CIも記録するが追加の優越主張はしない。

gen3は単体で約.4ms、非NNは約20msの実消費で、同wall優越とは呼ばない。warmupのmake/初回act費用とwarm対戦実時間を分ける。全棋譜の独立座標再生・得点/point/CI算術・SHA・事前事後訓練/旧評価状態非重複・完成段階条件を監査。新評価driverだけをテストし不変モデルの重い学習検証を繰返さない。

L/F確認成功なら最新remote mainから履歴不要モデル3本/既存探索/CPU人間CLI/READMEの最小統合branchを作り新cloneで検証する。main/Web/gen3既定は変更しない。未達なら研究記録だけ保存する。
