# CPU生成・推論の等価高速化

基点bd962d7、main変更なし。追加本学習0。先に新seed1730000以降の12ゲームを保存し、合法手/12特徴/39特徴pack/board encode/observe/単体CPU・CUDA推論を分離測定する。初期化・モデルロード・warmupを計時から除外。16ゲームPPO batch収集・更新も別測定する。

改善は最大2系統、状態依存cacheを避けた不変配置情報の再利用と重複pack削減を第一候補とする。実測後に対象を確定し、合法手順序・特徴float32 bit・入力独立所有・乱数を維持。旧コードをreferenceとして保存。任意nativeは必要時のみ、今回まずPython/NumPyで行う。

同一局面・固定gen2重みで旧新を交互に複数測定。短いPPOは2iteration×16gamesを3反復、旧新とも同じ初期model/past/optimizer・seedから開始し、全trajectory/全obs/最終重み/optimizer/RNGを比較。各反復は初期化除外し、収集games/sec・CPU生成pack・GPU同期推論・更新wallを保存。本学習を延長しない。過去履歴は複製しない。

全21駒/向き/盤端と新ゲーム序中終盤・pass・終了状態で旧列挙順、独立座標合法性、特徴、遷移を検査。CPU/CUDA出力・着手比較、配列mutation隔離、全テスト、新clone CPU CLIを確認する。時間制御探索の展開量/着手は速度で変わり得るため同挙動保証対象外。棋力改善を主張しない。有限90分目安、改善が確認できたらmain用最小差分を分離commitして研究branch pushし、統合判断を秘書へ返す。

初期profile後の対象確定: 合法手0.212ms、12特徴0.226ms、39特徴pack0.397ms（12特徴含む）、observe0.707ms、encode0.019ms。対象(1)featuresの不変配置幾何・局面内スカラーの重複計算削減、(2)inputs/observeの同じ配置幾何とcells pack再利用。Moveだけをkeyとする最大32768件LRU、戻り値tupleのみ。出力NumPyは都度新規所有。nativeやstate cacheは追加しない。旧engine/legal_movesを変更しない。

初回training交互測定はforward/収集のみwarmupで更新kernel初回費用が最初の旧runへ偏る可能性があった。結果はtraining-initial.jsonへ保全し、更新も1batch warmup後に同じ3組を再測定する。実装・seed・計測更新量・比較条件は不変。計測対象192ゲーム×2回とwarmup16×2回、合計416ゲームの検証のみ。最終報告は更新warmup済み測定を使用し初回結果も参照可。
