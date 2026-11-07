from h_distill import *
rows=[];generation=sum(json.loads((R/f'{p}-summary.json').read_text())['wall_seconds'] for p in ('pilot','train','dev'));learning=0
for seed in SEEDS:
 d=json.loads((R/f'learning-{seed}.json').read_text());learning+=d['wall_seconds'];last=d['curve'][-1];before=d['before_dev'];rows.append(f"|{seed}|{last['train']['labels']} / {last['dev']['labels']}|{before['agreement']:.1%} → {last['dev']['agreement']:.1%}|{before['ce']:.3f} → {last['dev']['ce']:.3f}|{last['updates']}|")
selection=json.loads((R/'selection.json').read_text());val=json.loads((R/'validation-summary.json').read_text())
def table(d):
 lines=['|比較（候補/固定gen3）|point|95%CI|2比較補正CI|平均得点差 [95%CI]|候補/対照 ms|','|---|---:|---|---|---|---|']
 for name,a in d.items():
  ci=lambda xs:'–'.join(f'{x*100:.2f}%' for x in xs)
  lines.append(f"|{name}|{a['point']:.2%}|{ci(a['point_ci95'])}|{ci(a['point_ci_bonferroni2'])}|{a['difference']:+.2f} [{a['difference_ci95'][0]:+.2f}, {a['difference_ci95'][1]:+.2f}]|{a['time']['0']['mean_ms']:.2f}/{a['time']['1']['mean_ms']:.2f}|")
 return '\n'.join(lines)
out=f'''# 固定H探索のpolicy蒸留（有限pilot）

## 結論
事前gate: **{'通過' if selection['gate'] else '未達'}**。これは探索教師の判断を学ぶ教師あり蒸留であり、実終局報酬で更新する追加強化学習ではない。主判定は候補+H対固定gen3+H（双方500ms）で、方策単体の改善とは区別する。本番/main/Frozenモデル/人間棋譜は変更していない。人間棋譜は0件との連絡に基づき一切利用せず、人間の敗因を再現したとは言わない。

## 固定条件と学習
元モデル1751401/2/3それぞれからpolicy MLPだけを8epoch更新。conv/global/position/value/priorは凍結。同一入力のvalueはCPU全ラベル・CUDA抽出ラベルでbit一致（audit.json）。4ch/全合法候補、heuristic priorあり。幅4教師の候補外は教師から発見できず、元方策の盲点を再生産し得る。教師の深さ6の探索値は真のWDL/minimax真値とは扱わない。同値のrootは複数正解。valueに教師値を回帰していない。

各seed128trainゲーム/32devゲーム、各12/20/28plyで最大3root。初期6plyランダム、固定gen3 greedy80%/top8ランダム20%。外生prefixはseed間共有、以降の軌跡はモデル依存。384train軌跡だが独立開始系列は128、96dev軌跡/32開始系列。pilot12ゲームは再利用しない。評価はゲーム開始bankも全て別。学習はAdam1e-4/minibatch16、CE+0.05KL、3更新seed9301/2/3、固定終点。ラベル・設定・モデルhashはmanifest.jsonとmodels.json。

|model seed|train/devラベル|dev教師一致率（前→後）|dev CE（前→後）|実更新数|
|---|---|---|---|---|
'''+'\n'.join(rows)+f'''

生成（pilot+train+dev）{generation:.2f}秒、CPU特徴cache+CUDA更新+診断+保存{learning:.2f}秒、合計{generation+learning:.2f}秒（90分枠内）。更新はRTX5090、float32、policy入力は凍結CPU特徴をcache。教師は幅4/6ply全root完成、60秒は安全上限であり時間打切りラベルは不採用。今回は速度最適化の比較ではない。

## 新validation（選択専用）
12prefix×先後×3seed＝72局/比較、計144局。point=勝1/分0.5。全3seed/両席を同prefix内として20,000回bootstrap。通常95%と2比較Bonferroni97.5%CIを区別。少数prefixで不確実性が大きい。主比較H、policyは補助。

'''+table(val)+'\n'
if (R/'confirmation-summary.json').exists():out+='\n## 独立確認\n新32prefix×先後×3seed＝192局/比較。validationと合算しない。\n\n'+table(json.loads((R/'confirmation-summary.json').read_text()))+'\n'
else:out+='\n独立確認は事前gate未達で実施しない。準備済み32prefixを成績選択に利用していない。追加条件/epoch/seed選択や救済探索なし。\n'
for method,a in val.items():
 out+='\n### '+method+' seed別\n'
 for seed,v in a['by_seed'].items():out+=f"- {seed}: {v['wins']}勝{v['draws']}分{v['losses']}敗、point{v['point']:.2%}、得点差{v['difference']:+.2f}\n"
 out+=f"候補/対照 p99={a['time']['0']['p99_ms']:.2f}/{a['time']['1']['p99_ms']:.2f}ms。H完了深さ/fallback/超過数はvalidation-summary.json。\n"
out+='''
## 解釈と限界
value自体は保全できても、学習したpolicyはrootだけでなく相手応手を含む全深さの候補順位を変える。教師root判断の再現と、狭い候補木の質の改善は同じではない。未使用ゲームの教師一致率/CEと実戦を併記し、どちらかだけで棋力改善を断定しない。CPU Python backendを双方で統一した比較で、Webのauto/native環境と実時間・時間制限下の着手が常に同一とは保証しない。固定3seed・新prefix分布でのAI対戦であり、人間への優越は未検証。

## 再現
```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
# original 27,533,566 training-state indexは既存SHA参照（pre-evaluation-separation.json）。巨大履歴を複製しない。
python scripts/h_distill.py pilot --games 4
python scripts/h_distill.py train --games 128
python scripts/h_distill.py dev --games 32
python scripts/h_distill_train.py
# epoch境界から同じAdam/RNGを復帰
python scripts/h_distill_train.py --seed 1751401 --resume
python scripts/h_distill_eval.py
python scripts/audit_h_distill.py
python -m pytest -q
python scripts/play_h_distill.py --checkpoint artifacts/h-policy-distill/student-1751401.pt --model-seed 1751401 --search --human-seat 0
```
推論exportは学習履歴不要（元gen3と同一アーキテクチャ、研究名を明記）。resume-*.ptはoptimizer/RNGを含む別用途。保存済み評価は再利用し、再計測する場合は別出力先/新計画にする。Web/mainへの自動採用なし。
'''
diag=json.loads((R/'final-diagnostic.json').read_text())
out+="""
## 監査結果・失敗の具体的診断
113 tests PASS（CUDA利用環境）。生成492対局＋評価144対局を独立座標checkerで全再生、合法手/pass/終局得点/集計PASS。教師ラベルの全候補合法性・元モデルhash・保存scoreからのtie/選択整合を確認し、32局面を完成6ply木から再生成して一致。全train/dev valueはCPUでbit一致、CUDA抽出局面でもbit一致。元PPO訓練訪問状態および今回train/devとの評価訪問状態重複0。pilot/train/devは6ply以降の訪問状態重複0（seed内/間の同phase共有を独立標本と誇張しない）。
別process CUDAで12更新連続 vs 5更新+保存再開7更新の重み/Adam/NumPy/Torch/CUDA RNG一致。seed1751402は1候補しかない末尾batchの正常なゼロ勾配を誤って拒否したため停止し、guardだけ修正してepoch1から再開。全192更新の連続再現と再開後endpointもbit一致。ゼロ勾配はupdate48/168、いずれも合法候補1件。条件/gateは変更していない。主処理の時間表は成功経路の記録であり、中断分・再開比較の補助計算は別（全体でも90分枠内）。

学習rootの12/20/28plyは全て手番0であり、**両席を均等に学習していない設計上の制約**があった。自分/相手の入力を正規化しても絶対配置/開始位置の幾何は異なる。対戦は両席で測り、候補Hの先手point51.39%（得点差+2.72）/後手30.56%（−3.94）。これは事後の記述的診断で、先手データ偏りが低下の原因だと証明したわけではない。次回修正を今回の成績へ遡及しない。

dev275rootで旧教師選択が新top4から脱落したのは10件（seed別1/3/6）。元教師は自分のtop4から必ず選ぶ。上位4集合の平均重なりは約90.7–94.9%で、学習が探索の候補集合を変えることを確認した。ただし相手応手の質が悪化したという因果は未検証。教師一致率は全seed合計43.64%→45.82%にとどまる。特にseed3は低下し、都合の良いseed1だけに選別しなかった。

Hのfallback集計には強制pass（候補123/対照115）が含まれる。**非passで即決fallbackした手は両者0**、非pass6ply完了は候補79.98%/対照77.87%。候補の超過176/1002着手、p99500.93ms、最大501.36ms。対照超過197/1005、最大557.20ms。今回は大きな遅延や未完成比較が主因という証拠はない。GPUでのcached policy logitとCPUとの差最大3.82e-6は事前atol2e-6+rtol1e-6内で、logitのbit一致とは言わない。valueと凍結重みのbit一致とは分ける。

## 次の限定的な仮説
まずroot抽出を両席均等（例えばゲームごとに12/20/28と13/21/29を半数ずつ）にした同経験量対照で、この偏りの影響を切り分ける。今回追加実行せず、根拠のないLR/epoch増加もしない。それでもHを超えなければ、幅4教師への一致をさらに追うより、探索後の実終局WDLを使う方策・価値の改善を別計画にする。教師の幅外の手を発見できない制約、凍結特徴の表現力、学習したroot順位を全探索層へ使う分布差は残る。今回候補は不採用、最小本番統合候補は作らない。CPU研究CLI/exportのみ保存する。
"""
(R/'REPORT.md').write_text(out)
manifest={str(p):sha(p) for p in sorted(R.rglob('*')) if p.is_file() and p.name not in ('manifest.json',) and not p.name.endswith('.log')}
manifest.update({str(p):sha(p) for p in Path('scripts').glob('*h_distill*.py')});manifest.update({str(p):sha(p) for p in Path('tests').glob('*h_distill*.py')});write(R/'manifest.json',manifest)
