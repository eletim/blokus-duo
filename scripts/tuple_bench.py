"""Fresh-process value timings; run outside generation and timed games."""
import time
BEGIN=time.perf_counter()
import argparse,gzip,json,copy,resource
import numpy as np
from blokus.engine import play,Move
from blokus.linear_rl import prefix
from blokus.tuple_value import load_value,codes
from scripts import tuple_cycle as cycle
p=argparse.ArgumentParser();p.add_argument('--arm',choices=['F','L','T'],required=True);a=p.parse_args();IMPORT=time.perf_counter()
path=cycle.source(3101) if a.arm=='F' else cycle.R/f'{a.arm}-3101.json';agent=load_value(path)
with gzip.open(cycle.R/'calibration-3101/games.json.gz','rt') as f:games=json.load(f)
states=[]
for g in games[:6]:
 s=prefix(g['prefix_seed'])
 for ply,row in enumerate(g['moves'],4):
  if ply in (8,18,28):states.append(copy.deepcopy(s))
  play(s,Move(*row) if row else None,validate=False)
SETUP=time.perf_counter();t=time.perf_counter();first=agent.value(states[0]);cold=(time.perf_counter()-t)*1000;rows=[]
for s in states:
 for _ in range(5):
  t=time.perf_counter();v=agent.value(s);dt=(time.perf_counter()-t)*1000;t=time.perf_counter();codes(s);extract=(time.perf_counter()-t)*1000;rows.append(dict(ms=dt,codes_ms=extract,value=v))
print(json.dumps(dict(arm=a.arm,imports_ms=(IMPORT-BEGIN)*1000,setup_ms=(SETUP-IMPORT)*1000,cold_value_ms=cold,warm_value_ms_quantiles=np.quantile([x['ms'] for x in rows],[.5,.9,.99,1]).tolist(),warm_code_ms_quantiles=np.quantile([x['codes_ms'] for x in rows],[.5,.9,.99,1]).tolist(),peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,states=len(states),rows=rows)))
