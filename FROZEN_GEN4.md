# gen4 Frozen — 2026-11-20

固定版は annotated tag `gen4-frozen-2026-11-20`。コードと3seedのモデルを同じcommitで保存する。タグは管理者が移動可能なので、`git rev-parse gen4-frozen-2026-11-20^{commit}` のcommitと [SHA256 manifest](models/gen4/manifest.json) を固定参照にする。旧gen3の重み・タグは変更しない。今後もgen4を上書きせず新世代名を使う。

**gen4はNN方式**。Frozen gen3のPPO方策を固定し、実終局WDLから学習した合法配置value Aを組み合わせる。非NN線形Lとは別方式。49,152訓練局・共通epoch5（開発MSEだけで選択）、3seed 1751401/1751402/1751403を全て保存。提供seedは既存の先頭seed規約に従う1751401で、対戦成績による選別なし。元成果commit `8cc8094ca3bab12ff77ac27af369e05384ee09db` から推論exportをbyteコピーし再保存していない。optimizer/RNGを含む学習再開checkpointではない。

盤面は開始点専用入力なしの4ch×14×14、残駒等49特徴、valueには合法配置252特徴。元policyはheuristic priorありPPO。方策・価値を別ネットとしてCPUで実行し、既検証の幅4、最大6ply、目安500msで2→4→6の完成段階だけを採用する。未完段階の値は採用せず、完成段階がなければ方策greedy。常に6手完成する保証やハード時間上限ではない。CLIとWebは同じ `blokus.gen4.search` を使う。

独立確認は各384局・先後交換・全3seed。旧3,072局Aに60.29%（4比較補正CI56.12–64.58）、Frozen gen3+Hに68.23%（62.89–73.70）、同量Cに56.77%（52.73–61.20）。24,576局Aには51.43%（47.79–54.95）で上積み未確定。**旧の全比較採用gate未達を変更せず、ユーザーの明示採用で公開する**。固定分布での結果で、人間より強いとの証明ではない。詳細は既存研究REPORTを保持する。今回追加学習・棋力再評価なし。

## CPUで復元して遊ぶ

Python3.12、Linux x86_64。CPU依存固定例は`artifacts/gen4-release/cpu-requirements.txt`（torch2.7.1+cpu/NumPy2.5.4）。サービス検証環境は `artifacts/gen4-release/environment.json`。CUDA版torchでも推論deviceはCPUに固定する。

```bash
git clone --branch gen4-frozen-2026-11-20 --single-branch https://github.com/eletim/blokus-duo.git blokus-gen4
cd blokus-gen4
git rev-parse HEAD
python3.12 -m venv .venv
.venv/bin/python -m pip install 'torch==2.7.1+cpu' --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r artifacts/gen4-release/cpu-requirements.txt
python3.12 scripts/verify_frozen_gen4.py
# 評価済みnative経路をビルド（Cコンパイラ、または上記依存のziglangを使用）。未ビルドならautoはPythonへfallback。
.venv/bin/python -m blokus.ppo_native build
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m blokus.gen4_play --human-seat 0
# 後手は --human-seat 1。AI完走は --human-seatを省略。
# Pythonのみで動かす場合 --backend python（速度は別条件）。
.venv/bin/python -m blokus.web_play --host 127.0.0.1 --port 8765 --device cpu
```

モデルはGit実体であり外部model download不要。推論に研究の訓練履歴・巨大state index・optimizerは不要。manifestのsourceは由来情報で、loadはmodels/gen4と既存artifacts/ppo-gen3だけを読む。研究全履歴を省く運用は下記deployment記録のsparse checkoutを参照。

Webでは「gen4・学習済み配置評価＋先読み」を選び、先手/後手を選んで「新しい対局」。旧gen3各モードと非NNは選択肢として維持。自動棋譜保存・手動downloadも維持し、gen4のpolicy/value hash、時間・完成深さを記録する。実人間棋譜はGitへ保存しない。

## サービス・停止・復帰

運用記録とrollbackは [deployment](artifacts/gen4-release/deployment.md)。公開範囲は既存のTailscale内 `http://100.95.30.96:8765` のみ。認証・防火壁変更なし。
