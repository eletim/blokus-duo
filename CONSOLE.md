# AI観戦・学習コンソール

VPN内URL: http://100.95.30.96:8766/

対戦記録と対局番号を選び「再生」で観戦します。停止、一手送り、一手戻し、手順スライダー、速度変更、次の対局、連続再生に対応します。青が先手、黄が後手です。評価対戦は保存された共通開始局面を含めて再生し、その区間を画面上で区別します。終局時は保存スコアとの一致を検証します。

強さの比較は評価区分と相手を選択します。勝率は勝ち/対戦数、得点率は(勝ち+0.5×引き分け)/対戦数です。異なる評価局面で測定した集計を、改善の証明として扱わないでください。確認評価に36,864局の点がないのは未実施のためで、補間データを生成していません。49,152局版の採用と、24,576局版からの改善未確認を併記しています。

学習欄は保存済みの処理履歴です。「記録を再取得」で更新します。稼働プロセスを監視するライブテレメトリは未取得と表示し、過去の完了記録を現在進行中の学習として見せません。24,768局は今回の追加生成（train/dev合計）、49,152局は累積学習量で、別の指標です。学習表は配置価値Aのseed 1751401の完了epochと採用epochを区別します。

## データ連携

- `/home/eletim/blokus-value-placement/artifacts/value-placement-24576`: 評価summary、圧縮棋譜とprefix bank、学習curve/selection。
- 同`value-placement-49152/n36864`と`n49152`: 同形式の段階評価・確認評価。最新のモデルはgen4。
- 同`value-placement-49152/games/timing.json`と`pipeline-complete.json`: 実際の生成数と完了日時。
- `/home/eletim/.local/share/blokus-web/games.sqlite3`: SQLite `mode=ro`接続。`kind=human`の2件だけを別枠表示。テスト棋譜は表示・勝率集計に含めません。
- 初期配置時点で42組、2,592評価対戦、14評価集計、3学習段階。元の対戦数には段階評価と確認評価が含まれ、グラフでは混合しません。

取得できない値は「未取得」、読み込み・再生検証失敗はエラーにします。任意パス読み込みや学習・対局開始・保存のAPIはありません。AIモデルやTorchはロードしません。対象形式以外の新研究データは自動取り込みしません。

## 独立運用

実装: `/home/eletim/blokus-console`、ブランチ`feature/observation-console`（最新origin/mainから分岐）。既存の`blokus-gen4-frozen`、モデル、棋譜DB、8765番のWebサービスは変更していません。mainマージと一般インターネット公開は行いません。

Python標準ライブラリだけで実行できます:

```sh
python3 -m blokus.console --host 100.95.30.96 --port 8766 \
  --artifacts /home/eletim/blokus-value-placement/artifacts \
  --archive /home/eletim/.local/share/blokus-web/games.sqlite3
```

配置済みのユーザーサービスは`blokus-console.service`。ユニットは`artifacts/console-review/blokus-console.service`にも保存しています。運用確認は`systemctl --user status blokus-console.service`、ログは`journalctl --user -u blokus-console.service`。停止・配置の取り消しは`systemctl --user disable --now blokus-console.service`で独立して行えます。既存8765番には影響しません。データが追加・変更された場合は「記録を再取得」。JSONはmtime/sizeでキャッシュを更新します。

## 検証

`tests/test_console.py`は欠損データ、共通prefixと先後の復元、不正棋譜/スコア不一致、任意パス拒否、人間棋譜の選別、DB非変更を確認します。

`PYTHONPATH=. python3 scripts/check_console_data.py`は42組すべての先頭/末尾84対戦と人間棋譜2件を実ルールで再生し、終局スコア一致とDBハッシュ不変を検証します。対局生成ではありません。

`scripts/check_console_browser.py`は既存のPlaywright/Chrome環境で8766番のみを操作します。再生・停止・一手送り・速度・シーク・終局結果・次局連続再生、グラフの切替、人間棋譜、デスクトップ/390px幅の画面、変更API拒否を検証します。JSON結果とスクリーンショットは`artifacts/console-review/`に保存します。既存8765番へのPOSTは行いません。

VPNアドレスへのHTTP応答とブラウザ表示はE-frontierから確認しています。別のVPN端末からの閲覧は未検証です。
