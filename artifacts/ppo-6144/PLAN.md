# 4096→6144 有限継続（事前登録）
基点 d29f47a。main gen2は変更しない。3seed最新4096のmodel/optimizer/past/全RNGから継続し、5120/6144を保存。seed1401のgen2基準は3072であり再開元4096と異なる。1402/1403は同一4096。
loss/勝敗終局報酬/モデル/LR .0003/16並行ゲーム/相手比率current .5,past .25,heuristic .25/4iterationごとのpast交換を固定。新しいshaping/探索なし。
既存全評価prefix集合を凍結し、その開始状態を含む新生成prefixを棄却。棄却数と旧評価開始状態への偶然の実訪問重複は別に記録する。旧訓練履歴・state indexはSHA付き参照で保持し、追加分だけ保存する。
生成+更新+保存wallは3seed合計5400秒以内。各seed cap最大1760秒、保存余裕を確保し、全体予算の残りで制限。予算到達なら完了iterationまでを保存し自動延長しない。非有限値は停止。64iteration連続窓でentropy平均<.02かつheuristic対戦point<.10なら崩壊疑いとして停止診断。タイムアウトは有限subprocess timeoutで回収する。
新validation seed1650000/32prefix。候補gen2・latest4096・5120・6144（未完走なら存在する保存checkpointのみ）。選択criterionはgen2 .5 + oldvalue6ms .25 + teacher100ms .25 のpoint、同値時同重み得点差、さらに早いiteration。再開元4096への直接対戦は全候補で別記しcriterionに追加しない。gen2自体は必ず候補に残す。seed選別なし。
選択固定後holdout seed1660000/96prefix・先後交換。主比較はgen2とteacher100ms。seed1401は選択と再開元の違いを明確にするため、成績に依存せずfresh同96prefixでlatest4096への補足を常に行う（他seedはgen2=latestなので重複対戦しない）。全seed/個別のpoint・得点差CIはprefix単位bootstrap。旧holdout成績を選択に使わない。
訓練全訪問・過去評価prefix/開始状態を新bankから事前除外。実訪問重複を事後監査。時間対戦は学習/監査と順次実行し、教師/旧探索を毎試合cold。ネット単体で全合法手、推論全費用を計測。同時間比較ではない。
