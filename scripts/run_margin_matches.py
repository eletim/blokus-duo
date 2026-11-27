"""Run only the frozen 30-task pilot; checkpoint each game, never regenerate it."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import copy,hashlib,json,random,time
from pathlib import Path
import torch
from blokus.gen4 import load,search
from blokus.margin_value import MarginValue
from blokus.margin_search import MarginSearch
from blokus.engine import State,Move,play,SIZES
R=Path('artifacts/margin-pilot');out=R/'matches';out.mkdir(exist_ok=True)
torch.set_num_threads(1)
# One CPU worker; timed comparisons share the same process and environment.
if hasattr(os,'sched_setaffinity'):os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
manifest=dict(model_sha256=hashlib.sha256((R/'model.json').read_bytes()).hexdigest(),tasks_sha256=hashlib.sha256((R/'tasks.json').read_bytes()).hexdigest(),budget_ms=500,width=4,max_depth=6,torch=torch.__version__,cpu_affinity=sorted(os.sched_getaffinity(0)),rules='no komi, ordinary engine score',selection='12 test components by sorted id; 3 frozen seeds for standard initial selfplay')
mp=R/'match-manifest.json'
if mp.exists():assert json.loads(mp.read_text())==manifest
else:mp.write_text(json.dumps(manifest,indent=2))
pairs={};start=time.monotonic()
for task in json.loads((R/'tasks.json').read_text()):
 path=out/(task['id']+'.json')
 if path.exists():
  old=json.loads(path.read_text());assert old['task']==task and old['model_sha256']==manifest['model_sha256'];continue
 seed=task['seed']
 if seed not in pairs:pairs[seed]=load(seed,'native')
 pair=pairs[seed];baseline=search(pair);value=MarginValue(R/'model.json',task['komi']);experimental=MarginSearch(pair[0],500,4,6,value_evaluator=value)
 s=State()
 for m in task['prefix']:play(s,Move(*m) if m is not None else None,validate=True)
 # Warm both evaluators before measurement, without making game actions.
 pair[1](s);value(s)
 stats=[];moves=list(task['prefix']);predictions=[]
 rngs=[random.Random(task['rng_seed']+p) for p in (0,1)]
 while not s.done:
  player=s.turn
  agent=baseline if task['kind']=='self' or player!=task['seat'] else experimental
  predictions.append(dict(ply=len(moves),player=player,raw=value.raw(s),adjusted=value(s)))
  m=agent.act(s,rngs[player if task['seat']==0 else 1-player]);st=copy.deepcopy(agent.stats)
  stats.append(dict(player=player,agent='margin' if agent is experimental else 'gen4',**st))
  play(s,m,validate=True);moves.append(None if m is None else [m.piece,m.orientation,m.row,m.col])
 scores=[s.score(p) for p in (0,1)];remaining=[sum(SIZES[i] for i in range(21) if s.remaining[p]>>i&1) for p in (0,1)]
 result=dict(task=task,model_sha256=manifest['model_sha256'],moves=moves,stats=stats,scores=scores,remaining=remaining,
 predictions=predictions,seconds=sum(st['elapsed_ms'] for st in stats)/1000,trace_sha256=hashlib.sha256(json.dumps(moves).encode()).hexdigest())
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(result));temp.replace(path)
 print(task['id'],scores,round(time.monotonic()-start,1),flush=True)
