# 固定L確認の再現

基点 `b32b89caf483e98d75ba553731dfaca8cd302915`。新規計画PLAN.md、9モデルmodels.json、固定256prefixはprefixes.json。旧32prefixを結果へ合算しない。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m blokus.ppo_native build
python -m pytest -q tests/test_linear_L_confirm.py
python -m scripts.linear_L_confirm prepare
# prepareはモデル/bank/旧評価indexを固定。保存済み出力の再生成は別worktreeで行う。
timeout 3600 python -m scripts.linear_L_confirm evaluate
python -m scripts.linear_L_audit
```

初期4plyの合法ランダムprefix、256固定×先後×3seed、L/FとL/gen3各1,536局。モデルはL-3101/2/3と元混成F、gen3対応1751401/2/3で固定。nativeを同じhostでbuildし、計測は他負荷を避ける。soft deadlineによりOS負荷で一部着手が変わり得るため、ビット同一の棋譜再生成とは主張しない。

訓練indexは旧研究参照 `/tmp/linear-retrain-training.npy`, `/tmp/ppo-lr64-training.npy`, `/tmp/ppo-minibatch-training.npy`, `/tmp/ppo-four-gpu-all-training.npy` と `artifacts/linear-tuple/new-visited.npy`。巨大過去履歴は複製せずpre-audit.jsonでraw SHAを記録。新規旧評価indexは既存非NN研究の保存棋譜を再生して生成、由来はold-evaluation.json。新cloneで推論する場合はこれらindex/研究履歴は不要。

研究用人間対戦（既存CLI・重み不変）:
```bash
python -m blokus.linear_play2 --checkpoint artifacts/linear-tuple/L-3101.json --backend python --completed-search --search-ms 100 --root-width 24 --reply-width 8 --depth 2 --human-seat 0
```
`--human-seat 1`は後手。native build済みなら`--backend native`。gen3の既定を変更しない。

履歴不要候補: `integration/linear-L-candidate` aa50302772302c6d8c5044d1bdc16ab02eeb22fc（main86d9e51直系、通常2commit、まだmain未適用）。`integration.patch.gz`も同じ最小差分（`gzip -dc integration.patch.gz | git apply`）。新cloneでCPU-onlyテスト88PASS/4skip、モデル出力一致・人間先後CLIを確認済み。`scripts.linear_L_candidate` はdecisionとauditの成功を確認して候補を作る補助スクリプト、`scripts.linear_L_fixture` は元実装の参照fixture生成。`scripts.linear_L_clone_check` は検証環境の固定パスを使うため別環境ではパスを合わせる。

```bash
git clone --branch integration/linear-L-candidate --single-branch https://github.com/eletim/blokus-duo.git
cd blokus-duo
# READMEのCPU用依存セットアップ後
python -m blokus.non_neural_play --model-seed 3101 --backend python --human-seat 0
# native利用時は先に python -m blokus.ppo_native build
```
