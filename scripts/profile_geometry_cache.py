"""Separate cold-cache and retained-memory diagnostic, not instrumented speed benchmark."""
import json,time,tracemalloc
from pathlib import Path
from blokus.placement_geometry import geometry
from blokus.engine import State,PLACEMENTS,play,Move
from blokus.research import clone
from blokus import cpu_reference as ref
try:from blokus.terminal_ppo import observe
except ImportError:from blokus.ppo_policy import observe
states=[]
for game in json.loads(Path('artifacts/ppo-cpu-throughput/games.json').read_text()):
 s=State()
 for ply,row in enumerate(game['moves']):
  if ply in [0,4,12,24,36] or row is None:states.append(clone(s))
  play(s,Move(*row) if row else None,validate=False)
results=[]
for repeat in range(4):
 for mode in ['old','cold','warm'] if repeat%2==0 else ['cold','warm','old']:
  if mode=='cold':geometry.cache_clear()
  t=time.perf_counter()
  for s in states:(ref.observe if mode=='old' else observe)(s)
  results.append(dict(repeat=repeat,mode=mode,ms_per_state=1000*(time.perf_counter()-t)/len(states),cache=geometry.cache_info()._asdict()))
geometry.cache_clear();tracemalloc.start()
for q in PLACEMENTS:geometry(q.move)
current,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
print(json.dumps(dict(timings=results,retained_traced_bytes=current,peak_traced_bytes=peak,placements=len(PLACEMENTS),cache=geometry.cache_info()._asdict(),memory_profiling_separate_from_timings=True)))
