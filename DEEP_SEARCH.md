# 固定gen3・幅4・反復深化6plyの統合候補

このbranchは本番変更ではありません。gen3の重み/タグと単体既定は保持し、CPU研究CLIを追加します。公開済みE（葉value専用、幅4/depth4）と、追加候補H（同幅でdepth2→4→6）を選べます。

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m blokus.ppo_deeper_play --method H --budget-ms 500 --human-seat 0
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m blokus.ppo_deeper_play --method H --budget-ms 500 --human-seat 1
python -m blokus.ppo_deeper_play --method E --budget-ms 500
```

モデルseed既定1751401、対応3seedは1751401/2/3。推論exportは既存mainのモデルを参照し、学習履歴・訓練index・研究moduleは不要です。純Python/CPUで利用可能。追加学習なし。

候補幅は全深度で4、rootのpolicy greedyを保持。各深さの全rootが完成した時だけ選択を更新し、6plyの途中打切りは完成4plyへ戻ります。TTは着手内のみ、状態/手番/残駒/last/pass/深さ/幅とbound種別を区別し、未完成値をexactとして採用しません。alpha-betaは同じ固定木の枝刈りで、全合法手を探索する完全minimaxではありません。NN valueは終局勝敗期待値で、得点差ではありません。

同500ms上限でもEより多く実時間を消費します。hard deadline保証、人間への優越、一般的最強は主張しません。最大2000ms指定はAPI互換の研究用で、推奨比較の根拠は500msのみです。結果・限界は研究 `research/gen3-deeper-value` のREPORTを参照してください。Webやmainへの採否は別途判断します。

独立確認（新32prefix×先後×固定3seed、各192局）: H/E point62.76%、2比較補正97.5%CI57.29–68.23%、得点差+2.42（95%CI1.56–3.30）。H/単体gen3 point68.75%、補正CI61.20–75.78%、得点差+4.69。H/E実時間平均262.69/68.92ms。H最大550.09ms、p99501.19msで、500msのhard保証はない。幅4/6ply・500msでの固定モデル/開始分布の結果で、人間や未知seed全体への優越を証明しない。確認でseedやcheckpointを選び直していない。
