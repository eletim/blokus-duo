import json
from pathlib import Path
from blokus.komi_analysis import records,analyze,summary
root=Path('artifacts/margin-pilot');rows=list(records('/home/eletim/blokus-value-placement/artifacts'))
(root/'existing-analysis.json').write_text(json.dumps(analyze(rows),ensure_ascii=False))
(root/'records.json').write_text(json.dumps(rows,ensure_ascii=False))
print(json.dumps(dict(total=len(rows),first=summary(rows),bonus_difference_games=sum(r['margin']!=r['unplaced_advantage'] for r in rows)),ensure_ascii=False,indent=2))
