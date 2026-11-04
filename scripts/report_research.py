"""Render a Japanese report from fixed selection and paired holdout artifacts."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('directory'); args = parser.parse_args()
    root = Path(args.directory)
    report = json.loads((root/'holdout-results.json').read_text())
    selection = json.loads((root/'selection.json').read_text())
    run = json.loads((root/'run-summary.json').read_text())
    lines = ['# 3方式研究 round1 実測レポート', '',
             'モデルはvalidationで固定後、別holdoutで評価した。holdout結果による再選択はしていない。', '',
             f"- Validation集合SHA256: `{selection['validation_sha256']}`",
             f"- Holdout集合SHA256: `{selection['holdout_sha256']}`",
             f"- 全ラウンドrun wall time: {run['wall_seconds']:.2f}秒（実装時間を除く）",
             '- 学習上限は各180秒、ゲーム境界停止。思考時間の同等性は主張しない。', '',
             '## 学習とvalidation選択', '',
             '| 方式 | 学習秒 | 対局/更新 | 学習局面 | 探索node | 選択段階 | validation平均得点差 |',
             '|---|---:|---:|---:|---:|---|---:|']
    for method in ('A','B','C'):
        t = run['training'][method]; s = selection['selected'][method]
        lines.append(f"| {method} | {t['training_seconds']:.2f} | {t['total_games']} / {t['updates']} | {t['learner_positions']} | {t['search_nodes']} | {s['stage']} | {s['criterion']:+.2f} |")
    lines += ['', '選択値はheuristicとv1へのprefix paired平均得点差の平均。中間/最終の比較にのみ使用した。A/Bはv1+heuristic蒸留と小さなon-policy補助、Cは探索教師蒸留。CはBと別重みで実学習した。', '',
              '## Holdout（各対戦24prefix×2局、prefix単位bootstrap 4000回）', '',
              '| 左 | 右 | 勝/敗/分 | 平均得点差 [95%区間] | 勝+0.5分率 [95%区間] |',
              '|---|---|---:|---:|---:|']
    totals = {}; ranking = {m:0. for m in ('A','B','C')}
    for result in report['results']:
        left, right = result['names']; s = result['summary']; dlo,dhi=s['difference_ci95']; wlo,whi=s['point_rate_ci95']
        lines.append(f"| {left} | {right} | {s['wins']}/{s['losses']}/{s['draws']} | {s['mean_difference']:+.2f} [{dlo:+.2f}, {dhi:+.2f}] | {s['point_rate']:.3f} [{wlo:.3f}, {whi:.3f}] |")
        if left in ranking and right in ranking:
            ranking[left] += s['point_rate']; ranking[right] += 1-s['point_rate']
        for side, name in [('left',left),('right',right)]:
            target = totals.setdefault(name, dict(calls=0,total_seconds=0.,network_seconds=0.,search_seconds=0.,legal_seconds=0.,feature_seconds=0.,nodes=0))
            for k in target: target[k] += result[side]['stats'][k]
    order = sorted(ranking, key=ranking.get, reverse=True)
    lines += ['', f"3方式総当たりの観測point率合計順位: {' > '.join(order)}（同点の場合の順序に意味なし）。この順位を新たなcheckpoint選択には使用しない。", '',
              '**24prefix・各1学習seedの暫定結果。** 区間はprefix間変動のみを表す。prefix内2局を独立観測とは扱わない。多数比較や学習seed間変動の補正はなく、小差を強化成功と断定しない。対random全勝でbootstrap区間が[1,1]でも、真の勝率100%を保証するものではない。', '',
              '未学習A/B/Cは固定heuristic priorにゼロ初期化した残差を加えたモデル。randomとは別物。各「方式 vs 方式_untrained」が学習の寄与、「C_search vs C」が同一重みへの探索追加の寄与。B vs Cは探索を使わない蒸留後policy同士。', '',
              '## 推論コスト（全holdout対戦をcall数で重み付け集計）', '',
              '| Agent | decisions | 全体ms/手 | network ms/手 | CPU合法手+特徴ms/手 | 探索ms/手 | 探索node |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for name,t in totals.items():
        scale=1000/t['calls']
        lines.append(f"| {name} | {t['calls']} | {t['total_seconds']*scale:.3f} | {t['network_seconds']*scale:.3f} | {(t['legal_seconds']+t['feature_seconds'])*scale:.3f} | {t['search_seconds']*scale:.3f} | {t['nodes']} |")
    lines += ['', 'network時間は特徴転送・GPU同期を含むwall time。heuristicの場合はCPU採点時間。探索nodesは生成した子状態数で、CPU合法手生成の走査候補数ではない。探索はroot4×reply3の幅制限で、追加時間を使う条件である。', '',
              '## 再現と残課題', '',
              'CLI・方式詳細はREADME、実験時の引数とsource hashは各config.json、固定checkpointとhashはselection.json。全対局スコア・座席・prefix IDはholdout-results.json。', '',
              '次ラウンド候補: CPU対戦成績で重みを直接最適化するA2 CEM（GPU勾配学習とは区別）、複数学習seedとprefix数の拡大、探索leafの学習valueとの混合をvalidationで調整、候補shortlist幅と時間制限の比較、teacher/poolの強化、batched CNN更新とCPU生成の並列化。今回のCは浅い探索なのでexpertの品質がstudentの上限になり得る。', '',
              '本ラウンドのholdoutは次回モデル選択に転用しない。次回は新holdoutを事前固定する。']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    print(root/'report.md')
if __name__ == '__main__': main()
