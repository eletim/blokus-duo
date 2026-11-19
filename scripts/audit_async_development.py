"""Replay all new development/restart trajectories after timing, without pooling banks."""
import gzip,json,hashlib
from pathlib import Path
import numpy as np
from audit_placement import replay
import frontier_generation as f
R=Path('artifacts/gpu-async512');rows={};unique=set()
for directory in sorted(R.rglob('*')):
 if not directory.is_dir() or (R/'final') in directory.parents or directory==R/'final':continue
 files=sorted(directory.glob('[0-9]*-*.json.gz'))
 if not files:continue
 keys=set();rngs=set();moves=0
 for path in files:
  d=json.load(gzip.open(path,'rt'));t=d['task'];k=(t['seed'],t['id']);assert k not in keys and t['rng_seed'] not in rngs;keys.add(k);rngs.add(t['rng_seed'])
  assert d['moves'][:len(t['prefix']['moves'])]==t['prefix']['moves']
  assert replay(d['moves'])==d['scores']
  assert len(d['stats'])==len(d['moves'])-len(t['prefix']['moves'])
  assert all(s['forced_pass'] or (s['depth']==6 and s['stages']==[2,4,6] and not s['fallback']) for s in d['stats'])
  assert all(np.isfinite(v) for s in d['stats'] for v in s.get('root_scores',[]))
  moves+=len(d['stats']);unique.add(hashlib.sha256(json.dumps(t,sort_keys=True).encode()).hexdigest())
 rows[str(directory.relative_to(R))]=dict(game_copies=len(files),moves=moves,all_legal_and_terminal=True,all_nonpass_complete6=True)
f.ref.write(R/'development-replay.json',dict(directories=rows,total_game_copies=sum(r['game_copies'] for r in rows.values()),unique_tasks=len(unique),note='development banks are intentionally reused across implementations and concurrency curves, not independent training/evaluation samples; formal separation is audited separately'))
