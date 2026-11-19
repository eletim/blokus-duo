import json
from pathlib import Path
R=Path('artifacts/gpu-async512');x=json.loads((R/'steady-4096/timing.json').read_text());t=json.loads((R/'steady-4096-telemetry.json').read_text());g=x['groups'][0];trace=g['occupancy_trace']
def point(n):return next(row for row in trace if row[3]>=n)
a,b=point(1024),point(3072);window=[row for row in trace if a[0]<=row[0]<=b[0]];intervals=[]
for n in range(1024,3072,256):
 u,v=point(n),point(n+256);intervals.append(dict(start_completed=n,end_completed=n+256,seconds=v[0]-u[0],games_s=(v[3]-u[3])/(v[0]-u[0])))
out=dict(games=4096,seed=g['seed'],outer_seconds=t['outer_seconds'],work_seconds=x['seconds'],outer_games_s=4096/t['outer_seconds'],work_games_s=4096/x['seconds'],interior_from_completed=a[3],interior_to_completed=b[3],interior_seconds=b[0]-a[0],interior_games_s=(b[3]-a[3])/(b[0]-a[0]),interior_active_min=min(row[1] for row in window),interior_active_max=max(row[1] for row in window),intervals=intervals,bank_sha=x['bank_sha'],not_a_paired_cpu_comparison=True)
(R/'steady-summary.json').write_text(json.dumps(out,indent=2)+'\n')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(2,1,figsize=(10,6));axes[0].plot([q[0] for q in trace],[q[1] for q in trace]);axes[0].axvspan(a[0],b[0],alpha=.15,color='green',label='Fixed interior: completions 1024–3072');axes[0].set_ylabel('Actually active games');axes[0].legend();axes[1].plot([q[0] for q in trace],[q[3] for q in trace]);axes[1].set_ylabel('Completed games');axes[1].set_xlabel('Work wall seconds');fig.suptitle('Separate 4096-game replenishment diagnostic, frozen first seed');fig.tight_layout();fig.savefig(R/'steady-occupancy.png',dpi=160)
