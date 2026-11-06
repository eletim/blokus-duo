# 4ch PPOと同量学習基盤高速化

研究a011c78、公開main c901d09を基準。新研究branch。ルール/START(4,4),(9,9)/報酬/モデル世代既定を維持。新モデルだけ4ch（own/opponent occupancy/frontier）。旧6chは明示channel/format識別で専用経路を保持、黙った重み切り詰め禁止。

現行16games/iteration、50%current/25%past/25%heuristic、PPO2epochs/minibatch16/clip.2/Adam.0003を固定。既存profileのCPU生成支配を踏まえ、任意native backendによる順序保存した合法手scan＋候補feature packと、rollout tensor一括転送/同形minibatch再利用を実装候補にする。根拠のない目標倍率は設定しない。旧Python/旧更新経路を参照として保持。nativeはC++共有ライブラリを明示build、未build時は純Python fallback。学習条件を変えずに計測し、効果がなければ採用しない。worker/compileは測定根拠がなければ導入しない。

6ch固定公開gen2で旧Python/旧update、nativeのみ、native+packed updateを交互に3反復、同じ経験・更新量・seedから測る。4chは別初期化でPython/native、channel差を分離。全観測・軌跡・model/optimizer/RNG bit等価を検査（差が出れば誤差を記録し採用判断を返す）。合法生成/feature+board pack/transfer同期/inference/update/saveを記録し、GPU eventのforward/backward時間をwallと区別。初期化・build・warmup・メモリ・候補/learner状態/step数を保存。計測は単独、学習/監査と並行しない。

小検証後4ch新経路を64iterations×16games=1024gamesのみpilot。強さ保証/公開gen2置換なし。CUDA別process連続4iteration vs 2+保存再開2（past交換を含む）とCPU更新/再開、pass/初手/frontier/終局/相手ラベル/入力mixをテスト。GPU計算wallは検証＋pilot合計60分以内、超えそうなら有限checkpointで停止。mainは変更せず、公開互換＋新学習CLIの最小統合候補を別branchに準備して返す。

同量交互3反復の結果、nativeだけで6ch 0.512→0.238秒/16games+更新（約2.15倍）、4ch 0.458→0.218秒（約2.10倍）。全軌跡/入力/model/optimizer/RNGがbit一致。resident rollout一括転送は明確な改善がなくpeak GPU memoryが約50MB増えたため既定に採用しない。`--packed`実験対照として残し、pilot/推奨新経路はnative＋従来minibatch転送。単なる4ch化による速度効果とは別に示す。nativeはC11実装（C++候補より依存を小さくした）、明示build、純Python fallbackを保持。worker/compileは導入しない。
