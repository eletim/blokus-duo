"""Render A2 CPU CEM report without changing any model selection."""
import json
import sys
from pathlib import Path


def main():
    root=Path(sys.argv[1]);run=json.loads((root/'run-summary.json').read_text());selection=json.loads((root/'selection.json').read_text())
    report=json.loads((root/'holdout-results.json').read_text());t=run['training']
    lines=['# A2 CPU CEM 追加改善レポート','',
           '第一ラウンド完了後の追加設計なので、新holdout seed61000を事前固定した。旧holdoutでcheckpointを再選択していない。',
           '学習はGPU勾配法ではなくCPU対戦成績によるcross-entropy method（進化戦略）。推論時の線形特徴重みを直接最適化した。','',
           f"- 学習 {t['training_seconds']:.2f}秒、{t['generations']}完了世代、{t['games']}対局、{t['candidate_evaluations']}候補評価。GPU勾配更新0回。",
           f"- run全体 {run['wall_seconds']:.2f}秒（validationと新holdoutを含む）。",
           '- 1世代population10、train prefix3、heuristic/v1相手に各prefix先後交換。120対局/世代。世代ごとにprefix seed71000+世代を更新。',
           '- 上位3候補で平均/標準偏差を平滑更新。サイズ重み12を固定、局面共通の2特徴を除いた9重みを学習。標準偏差下限0.25、重み範囲[-20,20]。',
           '- 240秒目安、最大100世代。直前世代時間×1.25が残予算を超える場合は次世代を開始しない。世代は途中中断しない。',
           f"- Validation選択: `{selection['source']}`、heuristic/v1への平均得点差criterion {selection['criterion']:+.3f}。",
           f"- Validation SHA256: `{selection['validation_sha256']}`",
           f"- 新holdout SHA256: `{selection['holdout_sha256']}`",
           f"- 選択checkpoint SHA256: `{selection['sha256']}`",'',
           '## 新holdout総当たり（24prefix×2局/組）','',
           '| 左 | 右 | 勝/敗/分 | 平均得点差 [95%区間] | 勝+0.5分率 [95%区間] |',
           '|---|---|---:|---:|---:|']
    totals={}
    results=report['results']
    if (root/'search-results.json').exists():
        extra=json.loads((root/'search-results.json').read_text())
        assert extra['prefix_sha256']==report['prefix_sha256']
        results=results+extra['results']
    for r in results:
        a,b=r['names'];s=r['summary'];dl,dh=s['difference_ci95'];wl,wh=s['point_rate_ci95']
        lines.append(f"| {a} | {b} | {s['wins']}/{s['losses']}/{s['draws']} | {s['mean_difference']:+.2f} [{dl:+.2f},{dh:+.2f}] | {s['point_rate']:.3f} [{wl:.3f},{wh:.3f}] |")
        for side,name in [('left',a),('right',b)]:
            out=totals.setdefault(name,dict(calls=0,seconds=0.,nodes=0))
            st=r[side]['stats'];out['calls']+=st['calls'];out['seconds']+=st['total_seconds'];out['nodes']+=st['nodes']
    lines+=['','## 思考コスト','', '| Agent | calls | ms/手 | 探索node |','|---|---:|---:|---:|']
    for name,t in totals.items():lines.append(f"| {name} | {t['calls']} | {1000*t['seconds']/t['calls']:.3f} | {t['nodes']} |")
    weights=json.loads((root/'selected.json').read_text())['weights']
    lines+=['','選択重み（blokus.ai.featuresの順序）: `'+json.dumps(weights)+'`','',
            '**暫定評価**: prefixを単位に4000回bootstrapした95%区間。各prefix内2局は独立観測と扱わない。各方式1学習seed、多数比較未補正。A2とC-searchは計算方式・予算が違い、時間同等比較ではない。',
            'A2-searchは新holdout閲覧前に追加宣言し、A2-offで選択済みの同一重みと固定探索幅4×3を使用。search-results.jsonに別保存し、holdout再選択は行っていない。',
            'Bは第一ラウンドvalidationで最良だった学習方式として事前に固定。新holdoutの成績でBやCの重みを選び直していない。A2_untrainedは初期heuristic重みで、A2との直接比較が学習の寄与。',
            '次ラウンドでは新しいprefix分割・複数学習seedを用意し、CEMのtrain prefix数拡大と対戦相手pool、C-searchの思考時間制限を検証する。']
    diagnostic=Path('artifacts/secretary-review/search-controls/results.json')
    if diagnostic.exists():
        d=json.loads(diagnostic.read_text())
        lines += ['', '## 秘書独立の探索対照診断（seed81000、選択には不使用）', '',
                  '| 左 | 右 | 勝/敗/分 | 平均得点差 [95%区間] |', '|---|---|---:|---:|']
        for r in d['results']:
            a,b=r['names'];v=r['summary'];lo,hi=v['difference_ci95']
            lines.append(f"| {a} | {b} | {v['wins']}/{v['losses']}/{v['draws']} | {v['mean_difference']:+.2f} [{lo:+.2f},{hi:+.2f}] |")
        lines += ['', '学習なしheuristic-searchも強く、学習した候補順位が同じ探索を明確に上回ったとは言えない。C-searchとA2-searchの優劣も未確定。診断はcheckpoint再選択に使用していない。']
    (root/'report.md').write_text('\n'.join(lines)+'\n');print(root/'report.md')
if __name__=='__main__':main()
