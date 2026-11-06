"""Render final fixed-confirmation numbers without model selection."""
from scripts import linear_L_confirm as e
r=e.R;s=e.read(r/'results.json');cost=e.read(r/'costs.json');d=e.read(r/'decision.json')
lines=['各組1,536局、全3,072局を完走。点推定によるモデル差替えなし。','', '|比較|勝/分/敗|point|通常95%CI|2比較補正CI|平均得点差 [95%CI]|実ms 左/右|','|---|---|---:|---|---|---|---|']
ci=lambda x:f'{100*x[0]:.2f}–{100*x[1]:.2f}%'
for k,v in s.items():
 a=v['difference_ci95'];lines.append(f"|{k}|{v['wins']}/{v['draws']}/{v['losses']}|{100*v['point']:.2f}%|{ci(v['point_ci95'])}|{ci(v['point_ci_bonferroni2'])}|{v['difference']:+.3f} [{a[0]:+.3f},{a[1]:+.3f}]|{v['left_ms']:.3f}/{v['right_ms']:.3f}|")
lines+=['',f"事前判定：L/F優越={d['old_nonNN_superiority']}、L/gen3優越={d['gen3_superiority']}。補正下限50%超だけを優越とする。未達は同等性や非劣性の証明ではない。",'','|比較|seed3101 point/差|seed3102 point/差|seed3103 point/差|L先手 point/差|L後手 point/差|','|---|---|---|---|---|---|']
for k,v in s.items():
 f=lambda a:f"{a['point']*100:.2f}%/{a['difference']:+.3f}"
 lines.append('|'+k+'|'+'|'.join([f(v['by_seed'][str(x)]) for x in e.SEEDS]+[f(v['by_seat'][str(x)]) for x in (0,1)])+'|')
lines+=['','seed別・席別の通常95%CIはresults.json。これらを追加の採用選択に使わず、主2比較以外の同時優越は主張しない。','',f"主対戦wallは{sum(v['wall_seconds'] for v in s.values()):.2f}秒、driver全体{d['wall_seconds']:.2f}秒。追加訓練0。",'','|比較/側|完成率|greedy fallback|超過件数|中央値/p90/p99/max ms|','|---|---:|---:|---:|---|']
for k,sides in cost.items():
 for side,v in sides.items():lines.append(f"|{k}/{side}|{100*v['target_complete']:.4f}%|{100*v['greedy_fallback']:.4f}%|{v['overshoots']}|"+'/'.join(f'{x:.2f}' for x in v['ms_quantiles'])+'|')
lines+=['','全rootの完成段階だけを採用し、部分探索値は破棄した。完成root/reply数、特徴/value費用、超過量はcosts.json。gen3単体と同消費時間の比較ではない。','']
cold=[]
for opp in ('F','gen3'):
 for seed in e.SEEDS:
  row=e.read(r/f'L-{opp}-{seed}.json.gz');cold.append(dict(opponent=opp,seed=seed,cold=row['cold']))
e.write('cold-warm.json',cold)
p=r/'REPORT.md';t=p.read_text();t=t.replace('計測完了後に確定値を記載する。','\n'.join(lines));p.write_text(t)
