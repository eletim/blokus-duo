"""One fresh-process CPU timing sample; run only outside tournament timing."""
import time
START=time.perf_counter()
import argparse,copy,gzip,json,random
from blokus.engine import State
from blokus.linear_value import Agent
from blokus.linear_tree import CompletedSearch
p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=3101);args=p.parse_args()
IMPORT=time.perf_counter()
a=Agent.load(f'artifacts/linear-hybrid/exports/seed-{args.seed}.json');settings=json.load(open('artifacts/linear-search/settings.json'))
with gzip.open('artifacts/linear-search/pilot.json.gz','rt') as f:rows=json.load(f)['states']
s=State(*copy.deepcopy(next(r['state'] for r in rows if r['seed']==args.seed and r['ply']==18)))
b=CompletedSearch(a,ms=settings['ms'],width=24,replies=8);SETUP=time.perf_counter();b.act(s,random.Random(1));first=b.stats[-1]
for i in range(5):b.act(s,random.Random(1))
print(json.dumps(dict(seed=args.seed,imports_ms=(IMPORT-START)*1000,setup_ms=(SETUP-IMPORT)*1000,first=first,warm=b.stats[1:],note='process-cold first act includes lazy native load; initialization outside act separately reported')))
