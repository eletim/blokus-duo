"""Independent mechanical check of the preregistered gate, no new selection."""
import os,json
from pathlib import Path
r=Path(os.environ.get('H_BALANCED_DIR','artifacts/h-distill-balanced'))
v=json.loads((r/'validation-summary.json').read_text());b=v['B-public'];ba=v['B-A'];p=v['policy-B-public']
criteria=dict(public_point_above_half=b['point']>.5,public_score_positive=b['difference']>0,two_seed_points_at_least_half=sum(x['point']>=.5 for x in b['by_seed'].values())>=2,not_below_A_point=ba['point']>=.5,policy_not_clear_loss=p['point_ci95'][1]>=.5,p99_at_most750=b['time']['0']['p99_ms']<=750,depth6_at_least20percent=b['time']['0']['depth6_fraction']>=.2)
expected=all(criteria.values());assert expected==json.loads((r/'selection.json').read_text())['gate']
if not expected:assert not list(r.glob('confirmation-B-*.json'))
(r/'decision-audit.json').write_text(json.dumps(dict(pass_=True,criteria=criteria,gate=expected,unchanged_plan=True),indent=2)+'\n')
print(json.dumps(criteria))
