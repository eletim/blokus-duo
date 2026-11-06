# 再現・CPU対戦

requirementsを入れたCPU環境でも動作する。測定はnative backend、ビルドは既存READMEの `python -m blokus.ppo_native build`（ccまたは任意依存ziglang）。Python fallbackはビルド不要だが6ms内の探索完了を保証しない。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m blokus.linear_play2 --checkpoint artifacts/linear-value-retrain/P-3101/policy.json --backend python --search-ms 6 --human-seat 0
# 人間後手は --human-seat 1
python -m pytest -q
python -m scripts.check_linear_value_fit_resume
# /tmpの過去索引が無い場合だけ、旧棋譜から再構築
python -m scripts.audit_linear_value
python -m scripts.linear_value_fit_audit train
python -m scripts.linear_value_fit_audit diagnostic
python -m scripts.linear_value_fit_audit evaluation
python -m scripts.linear_value_fit_separation
python -m scripts.linear_value_fit_summary
```

元モデルは `artifacts/linear-hybrid/exports/seed-3101.json`。今回のP推論JSONは同じ旧方策＋新しいvalueで、履歴不要。元方策との2ファイル指定も可能：`--checkpoint artifacts/linear-value/old-3101/policy.json --value-checkpoint artifacts/linear-value-retrain/P-3101/policy.json`。研究・不採用モデルであり公開推奨ではない。

今回の結果ファイルがある評価コマンドはSHA一致する保存結果を再利用する。新規測定は結果を退避せず新しい出力先を指定するコード変更が必要で、旧結果を上書きしない。診断/学習を再実行する場合も新出力先へ、事前設定と除外集合を保存したものから使用する。

学習だけの同設定再実行例（mainに影響なし、保存済み除外集合を保持）：

```python
import shutil
from pathlib import Path
from scripts import linear_value_fit_run as run
old = Path('artifacts/linear-value-retrain')
run.R = Path('artifacts/linear-value-retrain-reproduction')
run.R.mkdir()  # 既存なら停止
for name in ('fixed-plan.json', 'exclusions.json', 'pilot-visited.json'):
    shutil.copy2(old / name, run.R / name)
run.train()
```

新prefixを成績で選び直さない。今回のselected=nullに対してconfirmationコマンドを実行しない。再開の個別CLIは `python -m blokus.linear_value_fit --resume STATE.json --until 256 --output NEW_STATE.json`。saved policy/元生成value/RNG/Adam/除外集合/iterationを保持し、lr変更は許していない。
