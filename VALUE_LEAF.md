# PPO探索の葉value専用経路（公開済みE）

Frozen gen3の重みは変更せず、探索木の葉ではpolicy候補配置・特徴・logitを作らずvalueだけを計算します。4ch/6chの盤面と49次元state、conv/global/valueの演算は元モデルと同じです。残駒・両者の面積/frontier・得点・passも保持します。terminalは実得点の勝敗符号です。

主変更は `ppo_search.py` のdepth0評価です。幅4、深さ2→4、合法なpolicy fallback、全root完成後のみ結果を採用する規則、公開モデル/既定seedは変えません。Python fallbackを保持し、nativeは必要ありません。

研究 `research/gen3-value-leaf` の事前固定許容差はvalue絶対誤差1e-7（相対誤差0）。4/6ch・Python/nativeの1,188値比較は差0、102完成木は着手一致でした。同じ完成4ply木の保存局面pilot（8局面・交互2反復、初期化除外）はPython A320.3ms→E97.9ms、native A92.1ms→E39.6ms。小さなpilot分布の測定であり一般的速度保証ではありません。native導入自体の速度差とEの効果は別です。

これは計算効率改善です。完成した同一木の結果を維持しますが、時間内に完成する深さが増えれば着手は変わります。棋力改善・人間への優越・hard deadline保証は主張しません。研究のD（root拡幅）はこの最小候補に含みません。

検証例:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q tests/test_value_leaf.py tests/test_value_search.py
python -m blokus.ppo_play --help
```

追加学習なし。2026-11-07、mainへ統合し稼働Webへ反映しました（統合・smoke記録8fb3e9d、artifacts/value-leaf-deployment）。重み・既定単体gen3・非NNモードは維持しています。
