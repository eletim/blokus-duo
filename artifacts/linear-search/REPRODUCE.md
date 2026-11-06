# 再現
基点ec64667、固定モデルはmodels.jsonとartifacts/linear-hybrid/exports。再学習値は使わない。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m blokus.ppo_native build
python -m scripts.linear_search_run prepare
python -m scripts.linear_search_run pilot
python -m pytest tests/test_linear_tree.py -q
python -m scripts.linear_search_run validation
# selection.jsonのselectedがnullでない場合だけ:
python -m scripts.linear_search_run confirmation
python -m scripts.linear_search_report
python -m scripts.linear_search_run archive
python -m scripts.linear_search_audit
python -m pytest -q
```
時刻ベース探索はOS負荷で打切り/対局結果が変わり得る。固定木診断の値/候補は再生監査する。既存結果があるmatchはprefix/model/config hash照合後再利用するので、再計測は別コピーのartifacts/linear-searchで実施すること。

訓練全訪問indexは過去履歴を複製しない。`/tmp/linear-retrain-training.npy`（3,478,157 states）はresearch/linear-value-retrain cb25af9のtraining-auditから再生成。入力SHAはexclusions.json。過去評価prefix除外集合は同ファイルに保存。prepare時だけ過去worktreeも参照するが、評価/推論に過去worktreeは不要。

CPU-only人間対戦（研究用、棋力採用はREPORT参照）:
```bash
python -m blokus.linear_play2 --checkpoint artifacts/linear-hybrid/exports/seed-3101.json --backend python --completed-search --search-ms 100 --root-width 24 --reply-width 8 --depth 2 --human-seat 0
# 深さ条件: --root-width 6 --reply-width 3 --depth 4
# 旧6msは --completed-search を付けず --search-ms 6
```
全候補生成を含む時間上限。Python fallbackではnativeと完成率/着手が違う可能性がある。公開gen3やWeb既定は変更しない。

CPU-only CLI検証: `BLOKUS_CPU_PYTHON=/path/to/cpu-python python -m scripts.linear_search_cli_smoke`。全対戦記録はgzipで保存。archive-manifest.jsonに非圧縮内容のSHAがあり、キャッシュ読込/監査はgzipに対応する。追加PPO分離監査は/tmp/ppo-*-training.npyが存在する場合に実行し、利用した全index SHAと件数を保存する。
