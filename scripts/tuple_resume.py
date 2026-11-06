"""Real-pilot L/T 4 updates versus 2 + separate-process 2, all three seeds."""
import argparse,json,os,subprocess,sys,tempfile,hashlib
from pathlib import Path
import numpy as np
from blokus.tuple_value import Learner
from blokus.linear_value import Agent
from scripts import tuple_cycle as cycle
p=argparse.ArgumentParser();p.add_argument('--child');p.add_argument('--data');args=p.parse_args()
if args.child:
 t=Learner.load(args.child);d=dict(np.load(args.data))
 for _ in range(2):t.update(d['phi'],d['codes'],d['y'])
 t.save(args.child)
else:
 rows=[]
 for seed in cycle.SEEDS:
  path=cycle.R/f'pilot-{seed}/data.npz';d=dict(np.load(path))
  for arm in ('L','T'):
   a=Learner(Agent.load(cycle.source(seed)),arm,seed+10000);b=Learner(Agent.load(cycle.source(seed)),arm,seed+10000)
   for _ in range(4):a.update(d['phi'],d['codes'],d['y'])
   for _ in range(2):b.update(d['phi'],d['codes'],d['y'])
   with tempfile.TemporaryDirectory() as tmp:
    pa=Path(tmp)/'a.json';pb=Path(tmp)/'b.json';a.save(pa);b.save(pb);subprocess.run([sys.executable,'-m','scripts.tuple_resume','--child',str(pb),'--data',str(path)],check=True,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'});assert json.loads(pa.read_text())==json.loads(pb.read_text());rows.append(dict(seed=seed,arm=arm,continuous_updates=4,first_updates=2,resumed_updates=2,weights_optimizer_rng_order_visits_exact=True,state_sha256=hashlib.sha256(pa.read_bytes()).hexdigest(),data_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
 cycle.write('resume-audit.json',dict(pass_=True,separate_process=True,rows=rows));print('RESUME PASS',flush=True)
