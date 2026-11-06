"""Post-evaluation diagnostic on common F-versus-gen3 visited states; no fitting."""
import gzip,json
import numpy as np
from scripts import tuple_eval as ev
from blokus.engine import Move,play
from blokus.linear_rl import prefix
from blokus.tuple_value import load_value,phase
R=ev.R;rows=[]
for seed in ev.SEEDS:
 with gzip.open(R/f'validation-F-gen3-{seed}.json.gz','rt') as f:d=json.load(f)
 agents={l:load_value(ev.source(l,seed)) for l in ('F','L','T')};pred={l:[] for l in agents};ys=[];phases=[]
 for g in d['games']:
  s=prefix(g['prefix_seed']);y=float(np.sign(g['difference']))
  for row in g['moves']:
   if s.turn==g['seat']:
    ys.append(y);phases.append(phase(s))
    for l,a in agents.items():pred[l].append(a.value(s))
   play(s,Move(*row) if row else None,validate=False)
 y=np.array(ys);ph=np.array(phases);stats={}
 for label,v in pred.items():
  v=np.array(v);stats[label]=dict(mse=float(np.mean((v-y)**2)),sign_accuracy=float(np.mean(np.sign(v)==y)),by_phase={str(p):dict(states=int(np.sum(ph==p)),mse=float(np.mean((v[ph==p]-y[ph==p])**2)),mean_prediction=float(np.mean(v[ph==p])),mean_target=float(np.mean(y[ph==p]))) for p in range(3)})
 rows.append(dict(seed=seed,games=len(d['games']),states=len(ys),models=stats))
(R/'gen3-distribution-diagnostic.json').write_text(json.dumps(dict(post_validation_diagnostic=True,used_for_training_or_selection=False,targets='actual terminal WDL from F-seat in fixed F/gen3 games, not NN values or NN move imitation',rows=rows),indent=2)+'\n')
