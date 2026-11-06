# 固定線形L：CPUで遊べる非ニューラル候補

既存の公開gen3・グローバルdefault・稼働Webを変更しない追加候補です。NumPyの段階別線形方策30係数と局面価値24係数を使用し、ニューラルネットを方策/価値に使いません。既存基盤の依存としてTorch CPUパッケージは必要です。推論に研究module・訓練履歴・巨大indexは不要です。

方策は以前の終局REINFORCE学習済み旧方策を固定。今回の価値は元の学習済み24係数を初期値に、固定された広幅非NN探索の自己対戦1,024ゲーム/seed・8epochの実終局勝敗から学び直しました（ゼロからの学習ではありません）。gen3の値や着手を教師にしていません。全3seed3101/2/3を提供し、例の既定3101は成績で選ばず先頭固定です。

```bash
python -m blokus.non_neural_play --model-seed 3101 --backend python --human-seat 0
# 後手: --human-seat 1、AI対戦: --human-seatを省く
# nativeを使う場合:
python -m blokus.ppo_native build
python -m blokus.non_neural_play --model-seed 3102 --backend native --human-seat 1
```

盤面開始点(4,4)/(9,9)、合法性、pass、全置き15/最後monomino20は既存ルールのまま。探索は完成6×3/2plyを確保し、24×8/2plyが全root完成した場合だけ置換、100ms soft deadline。未完了値を採用せず完成段階/greedyへfallbackします。完全探索ではありません。Python fallbackは遅く、同じdeadlineで完成範囲/着手が変わり得ます。cold初回の依存import費用とwarm対戦費用を区別してください。

評価数値はmodels/linear-L/confirmation.jsonと研究branch research/linear-L-confirmのREPORTを参照。新256prefix×先後×3seed、各1,536局、L/FとL/gen3の2主比較の補正CIで判定。旧32prefixを混ぜず、固定3seed/この評価分布の結論です。gen3単体より実思考が長く、同wall優越・全seed一般化・一般的最強の証明ではありません。旧局所表Tのgateを変更したものではなく、別の事前計画確認です。

確認結果：L/Fは947勝57分532敗、point63.51%、2比較補正CI60.71–66.24%、得点差+2.809。元の非NN価値に対する改善を確認しました。L/gen3は713勝75分748敗、point48.86%、補正CI46.22–51.47%、得点差−1.507で、gen3超えは未確認です。同等性・非劣性も主張しません。実msはL/Fが20.188/20.061、L/gen3が19.874/0.419。新規訓練なし、旧32prefixは合算していません。
