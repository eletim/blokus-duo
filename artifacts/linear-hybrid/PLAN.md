# 旧方策＋固定非NN価値：事前計画

前回128prefix確認で新39係数は旧30係数に45.25%、追加特徴は不採用。旧方策＋新価値はこの結果から生まれた新仮説で、旧確認標本は再利用しない。

モデル：a7baba3内old/new各3101/3102/3103、全て1024iteration=16384gamesの終点。方策6本/新価値3本をmodels.jsonでSHA固定。gen3対応1751401/2/3も固定。追加方策学習なし。root6/reply3/depth2/6msを変更しない。旧方策をrootだけでなく相手応手の順位にも使用、葉は前回と同じ24係数・視点・終局WDL。探索法は同一、条件変更は方策部の交換のみ。

第一段階：新validation64prefix、seed base2240000000、3seed先後。混成/前回探索を主比較、混成/旧方策、混成/gen3、前回探索/gen3を測る。全訓練訪問状態SHA、過去評価開始状態と前回calibration開始を除外。6局/prefixを一clusterとする95%bootstrap CI、pointと得点差と実wallを記録。同時間対gen3とは呼ばない。

進行gate：探索fallback<10%、混成/前回探索point>=.48、かつ混成/gen3 point>=前回探索/gen3 point-.03なら、固定混成を新128prefix確認へ進める（僅差順位だけで棄却しない）。混成/前回探索point>.5を方式内前進の点推定、gen3超えは別に検定。全seedを保持、checkpoint選択なし。

第二段階（必要時一度のみ）：主比較point<.5かgate未達の場合、別の新規各64gameで旧固定方策分布の価値MSEを調べる。ゼロ予測MSE以上、または新方策分布MSEより10%以上悪ければ分布ずれ仮説として旧固定方策下の終局WDLから価値のみ再学習。各seed固定8192games、16game更新512回、value LR.01/NumPy Adam、方策・pool50/25/25は固定（current/pastとも今回旧終点、initialは旧heuristic）。最大30分、非有限は停止。同じvalidationは選択用として再利用し報告。追加LR探索なし。新価値を採るには同じgateを満たし、混成/gen3 pointが旧価値以上、混成/前回探索pointが旧価値-.02以上。満たさなければ旧混成のgateで判断。

確認：候補固定後、新128prefix seed base2250000000、選択/calibration/全訓練状態から分離。混成/前回探索、混成/gen3を主family2（Bonferroni同時97.5%point CI、通常95%CIも別表示）。混成/旧方策は診断。各組768局。標本数を途中で増やさない。再学習を採用した場合のみ固定旧価値混成を追加診断、主familyは2のまま。gen3超えは補正point CI下限>.5の場合のみ。実時間・fallback・root完了・nodesを保存。時間対戦と監査/学習を重ねない。

監査：全棋譜再生/合法性/pass/得点/終局報酬/算術/モデルhash/全評価訪問状態の訓練との重複を確認。初期空盤面等のprefix以前は分離対象外。前回評価は開始状態の除外であり、全過去評価軌跡との非重複とは呼ばない。全root同深さ完了またはfallbackを検証。CPU-only CLI/独立小終盤/保存SHAをテスト。main/Webは変更しない。
