# 再現と推論

基点5776630。モデル由来はmodels.json、事前条件はPLAN.md/pilot後のfixed-plan.json。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m blokus.ppo_native build
python -m scripts.tuple_cycle prepare
# 各seed3101/2/3でpilot4games
python -m scripts.tuple_cycle generate --kind pilot --seed 3101 --games 4
python -m scripts.tuple_cycle generate --kind pilot --seed 3102 --games 4
python -m scripts.tuple_cycle generate --kind pilot --seed 3103 --games 4
python -m scripts.tuple_cycle freeze
python -m scripts.tuple_jobs
python -m scripts.tuple_eval prepare
python -m scripts.tuple_eval calibration
python -m scripts.tuple_audit training
python -m scripts.tuple_bench --arm T
python -m scripts.tuple_eval validation
# selection.json.proceed=trueの場合だけ
python -m scripts.tuple_eval confirmation
python -m scripts.tuple_audit evaluation
python -m pytest -q
```

過去の巨大履歴は複製しない。事前除外index `/tmp/linear-retrain-training.npy` と `/tmp/ppo-{lr64,minibatch,four-gpu-all}-training.npy` のSHA/件数はindices.json。各元研究branchの履歴監査から再生成する。新規生成は別seed系列で1024games/seed、旧凍結Bの固定木を毎手完了(ms=0)、3process。L/T各8epoch/256batch/.01Adam。既存出力を再生成するなら別worktree/出力を退避して使う。

推論には履歴/index/evaluation moduleは不要:
```bash
python -m blokus.tuple_play --checkpoint artifacts/linear-tuple/T-3101.json --backend native --human-seat 0
# --human-seat 1 は後手。--backend python はnative未ビルド時のfallback。
# L-3101.jsonや元のartifacts/linear-hybrid/exports/seed-3101.jsonも同じCLIで読める。
```

同じ方策・幅24x8/2ply/100ms、完成狭幅fallbackを維持。時間制御で着手が変わり得る。性能採用判断はREPORT参照。main/Web/gen3は未変更。

今回のselection.proceedはfalseなのでconfirmationは未実行。終局対戦1,152局の検証:
```bash
python -m scripts.tuple_audit evaluation
python -m scripts.tuple_resume
python -m scripts.tuple_distribution  # 事後診断のみ、選択/学習へ戻さない
python -m scripts.tuple_result_summary
BLOKUS_CPU_PYTHON=/path/to/cpu-only/python python -m scripts.tuple_cli_smoke
```
CPU-onlyスモークはnative AI対戦とPython人間先後を完走。人間入力promptと最終JSONの行分離を修正し再確認した（旧失敗ログcli-smoke-before-fix.logも保存）。予測表のexportはT-3101/2/3.json、線形対照はL-3101/2/3.json、保存再開stateは各*-state.json。exportは履歴不要。全コマンドはrepo rootから実行。
