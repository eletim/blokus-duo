# 再現と固定モデル

基点a7baba3、方策/価値は `artifacts/linear-value/{old,new}-310{1,2,3}/policy.json`、1024iteration終点。今回の追加方策学習はない。`models.json` が元ファイル全体とvalue配列のSHA。gen3は1751401/2/3対応固定。前回pilotではない。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m blokus.ppo_native build
python -m pytest -q
# 前回の派生索引がない場合だけ、保存棋譜から再構築（過去データを学習し直さない）
python -m scripts.audit_linear_value
python -m scripts.linear_hybrid_run validation
python -m scripts.linear_hybrid_run select
python -m scripts.linear_hybrid_run confirmation
python -m scripts.linear_hybrid_run audit
```

今回の実行ログや結果JSONが存在すると同hashの結果を再利用する。独立再測定には結果を別保存して新出力先にする改修が必要で、結果削除/上書きはしない。`select`は事前gate通過と分布診断不要を確認してから固定する。時間対戦と重い監査を並行しない。

CPU-onlyの人間対戦（先手0/後手1）：

```bash
python -m blokus.linear_play2 --checkpoint artifacts/linear-value/old-3101/policy.json --value-checkpoint artifacts/linear-value/new-3101/policy.json --backend python --search-ms 6 --human-seat 0
```

native測定と同じ速度には `--backend native`（事前ビルド）が必要。Python fallbackも合法動作するが制限内に完了する保証はない。CPUのみの推論でNN教師やNN価値は利用しない。共通の既存基盤がTorchをimportすることはある。初期化費用は定常対戦msから除外、soft deadlineはhard real-time保証ではない。

別ファイル指定は明示的v2形式/shape/finiteを検査する。研究runnerは起動前に固定SHAを全件照合する。`Agent.load_pair(...,policy_sha256=...,value_sha256=...)` は明示hashが不一致なら拒否する。組み合わせた推論JSONも2元ファイルのhashをmetadataに保存し、以後は履歴や元モデルを必要としない。

最小追加候補は `integration-candidate.patch`（load_pair/CLIオプション/テストのみ）。前候補abc41cfへ `git apply --check` が通ることを確認済み。今回は適用・main統合していない。必要なら小さいexports JSONを別途移し、モデルの由来SHAをmanifestと合わせて保持する。
