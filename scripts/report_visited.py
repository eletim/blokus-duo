"""Render the frozen experiment; no model selection occurs here."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from blokus.value_experiment import write


def read(p):return json.loads(Path(p).read_text())
def ci(x):return '['+', '.join(f'{v:+.3f}' for v in x)+']'

def main():
    root=Path('artifacts/visited-diversity');summary=read(root/'summary.json');audit=read(root/'audit.json');diag=read(root/'diagnostics.json');final=read(root/'holdout-results.json');stages=[s for s in ['pilot','full'] if (root/s/'summary.json').exists()]
    lines=['# 実戦局面の多様性・checkpoint選択の結果','',f"最終方式は新validationで固定した **{summary['final_stage']}/{summary['chosen']['condition']}/{summary['chosen']['selector']}**。既存valueの推奨は維持し、この実験モデルは研究用とする。",'','## fresh holdout','', 'seed450000の独立96prefix、各先後交換。更新seedは601/602/603で、データと初期重みは共通。共通6ms予算であり、実消費時間を完全にそろえた比較ではない。','','| 対戦 | 勝/敗/分 | 平均得点差 | prefix CI95 | 左/右 ms/手 |','|---|---:|---:|---|---:|']
    for r in final['results']:
        z=r['summary'];l=r['left']['stats'];rr=r['right']['stats'];lines.append(f"| {' vs '.join(r['names'])} | {z['wins']}/{z['losses']}/{z['draws']} | {z['mean_difference']:+.3f} | {ci(z['difference_ci95'])} | {l['milliseconds_per_move']:.3f}/{rr['milliseconds_per_move']:.3f} |")
    for opp,z in audit['crossed_bootstrap'].items():lines+=['',f"3seed総合 vs {opp}: **{z['mean']:+.3f}点、95%区間 {ci(z['ci95'])}**。seed/prefix交差bootstrap。"]
    prior=audit['crossed_bootstrap']['prior']
    if prior['ci95'][0]<=0<=prior['ci95'][1]:lines+=['','既存valueとの差の区間が0を含み、追加学習による安定した改善は確認できなかった。非学習探索との差があっても、既存モデル由来の強さと今回の追加改善を区別する。']
    elif prior['ci95'][1]<0:lines+=['','既存valueに対して劣化。今回のモデルを推奨へ置き換えない。']
    else:lines+=['','この固定条件の集計では既存valueに正の差が出た。ただし同一データ・初期重みの3更新seedであり、独立データ収集や別思考時間への一般化は未検証。推奨置換は追加レビューまで保留する。']
    lines+=['','## validationとselector','', '| 段階 | 条件 | selector | 対static | 対prior | 選択criterion |','|---|---|---|---:|---:|---:|']
    for stage in stages:
        rs=read(root/stage/'validation-results.json')
        for c in read(root/stage/'selection.json')['choices']:
            vals={o:sum(r['summary']['mean_difference'] for r in rs if r['condition']==c['condition'] and r['selector']==c['selector'] and r['opponent']==o)/3 for o in ['static','prior']}
            lines.append(f"| {stage} | {c['condition']} | {c['selector']} | {vals['static']:+.3f} | {vals['prior']:+.3f} | {c['criterion']:+.3f} |")
    lines+=['','raw/liveは同じ更新列から選択したcheckpoint。重みが同一なら同じ対局結果を再利用する。liveは実戦alpha=.5と終局上書きを含むが、そのdev損失低下だけで棋力改善とは判定しない。pilotで両条件が事前gateを通過したため全量へ拡大した。','', '## 計算量とラベル','', '| 段階/条件 | root×完走 | 継続対局 | 継続ply | CPU生成秒 |','|---|---:|---:|---:|---:|']
    label_seconds=0;continuations=0
    for stage in stages:
        for condition in ['wide','dense']:
            m=read(root/stage/f'{condition}.json');label_seconds+=m['seconds'];continuations+=m['continuations'];lines.append(f"| {stage}/{condition} | {m['groups']}×{m['repeats']} | {m['continuations']} | {m['continuation_plies']} | {m['seconds']:.1f} |")
    steps=sum(t['invocation_steps'] for s in stages for t in read(root/s/'summary.json')['training']);vgames=sum(read(root/s/'summary.json')['actual_validation_games'] for s in stages);hgames=sum(len(r['games']) for r in final['results'])
    lines+=['',f"GPU主学習は計{steps:,}更新/{summary['gpu_training_seconds']:.2f}秒（監査用再開検証別）、3seed×2条件×{len(stages)}段階。30分上限内。準備＋本run {summary['prepare_seconds']+summary['run_seconds']:.1f}秒。validation実対局{vgames}局、最終holdout{hgames}局。CPUラベル生成は6worker、上表合計{label_seconds:.1f}秒/{continuations:,}継続対局。dev/収集/監査は別。root×完走は同数だが、継続長と実CPU時間は完全には一致しない。",'', '| 段階 | 4完走→16完走の最良候補一致 | 独立8対8の最良候補一致 | 独立8対8候補RMSE点 |','|---|---:|---:|---:|']
    for s,d in diag.items():
        z=d['label_stability'];lines.append(f"| {s} | {z['4']['best_choice_agreement_with_16']:.3f} | {z['disjoint_8_vs_8']['best_choice_agreement']:.3f} | {z['disjoint_8_vs_8']['candidate_rmse_points']:.3f} |")
    for s,d in diag.items():
        c=d['coverage'];lines+=['',f"{s}の元ゲーム数: wide {c['source_games']}、dense {c['dense_source_games']}。wideの残駒数0/25/50/75/100%分位は{c['remaining_piece_quantiles']}。"]
    if 'full' in diag:
        ds=diag['full']['checkpoint_diagnostics'];c=diag['full']['coverage']
        regrets={cond:sum(x['metrics']['live']['decision_regret_points'] for x in ds if x['condition']==cond and x['selector']=='live')/3 for cond in ['wide','dense']}
        selected_steps=[x['step'] for x in ds if x['condition']=='dense' and x['selector']=='raw'];initial=read(root/'initial-dev.json')['live']['decision_regret_points']
        lines+=['',f"fullのlive基準checkpointのdev regretはwide平均{regrets['wide']:.3f}点、dense{regrets['dense']:.3f}点（初期{initial:.3f}点）。対局validationの表と比較すると、意思決定proxyの改善が実戦改善へ直結するとは限らない。dense/rawの選択更新はseed順に{selected_steps}であり、更新をさらに延ばす判断には対局での根拠が必要。",'',f"wideはroot数が4倍でも元ゲーム数は{c['source_games']}対{c['dense_source_games']}と近く、主に同一ゲーム内の異なる局面が増えた比較である。独立した展開の多様性を4倍にした結果ではない。次はゲーム単位で収集相手poolと展開を分離し、この違いを検証する。"]
    lines+=['','4対16は標本を共有する楽観的指標。独立8対8も真値ではなく、継続poolでのラベル不確実性を示す。学習対象は4候補を完了できた中終盤であり、実戦全体の分布とは異なる。','', '## 固定重みの探索時間曲線','', '| 予算ms | 勝/敗/分 | 得点差/CI95 | 実ms/手 |','|---|---:|---|---:|']
    for r in read(root/'curve-results.json')['results']:
        z=r['summary'];lines.append(f"| {' vs '.join(r['names'])} | {z['wins']}/{z['losses']}/{z['draws']} | {z['mean_difference']:+.3f} {ci(z['difference_ci95'])} | {r['left']['stats']['milliseconds_per_move']:.2f}/{r['right']['stats']['milliseconds_per_move']:.2f} |")
    lines+=['','開発専用8prefixの暫定診断。全比較の得点差CIは0を含み、長時間での強化はまだ確定していない。25/100msで探索候補は増えたが深さは同じ2ply。次は開発prefixを増やして時間曲線を確認し、有効なら単に候補幅を広げる方式と、候補を絞った追加深さ・終盤完全探索を固定予算で比較する。学習改善と時間増加の効果は混同しない。','', '## 検証と限界・次案','', '全テスト、別process CUDA 4更新対2+保存再開2更新、評価全手再生・得点/勝敗/CI/時間算術・checkpoint hash・選択最大値・ラベル標本再生（5dataset各4rootの先頭4完走）を実施。詳細はverification.json、resume-check.json、audit.json。semantic hashは盤面・残駒・両者得点・手番・passesを含む。学習root/leafと収集ゲーム中終盤、dev、validation、最終holdoutの完全重複を監査した。全ラベル継続の内部局面までは保存/監査しておらず、類似局面の漏洩も排除していない。', '', '訪問局面は実6ms探索から取得したが、継続教師は再生可能な固定4×3探索であり、分布不一致をすべて解消したわけではない。候補はheuristic上位4、相手応手は上位3のため、順位外の有望手には学習信号がない。平均継続得点を相手応手で最小化する教師と、実ゲームの適応的な相手にも差が残る。次の変更候補は損失追加ではなく、実戦で判断が割れる局面の追加ラベルと、候補外探索/終盤のより正確な教師。収集対局自体の相手poolを広げ、同一ゲーム内root増加と独立ゲーム展開の増加も分離する価値がある。今回のholdoutはこれ以降の選択には使わない。', '', '[設計と再現コマンド](METHOD.md)。新規試行は新出力先・新validation/holdoutを使う。']
    (root/'conclusions.md').write_text('\n'.join(lines)+'\n')
    write(root/'compute-summary.json',dict(training_updates=steps,gpu_main_seconds=summary['gpu_training_seconds'],validation_actual_games=vgames,holdout_games=hgames,label_seconds_excluding_dev=label_seconds,label_continuations_excluding_dev=continuations))
if __name__=='__main__':main()
