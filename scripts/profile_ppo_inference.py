"""Alternating fixed-state full inference wall, native vs Python, each channel format."""
import copy,json,random,time
from pathlib import Path
from blokus.ppo_policy import PPOAgent
from blokus.engine import State,Move,play
from blokus.ai import seed_all
from blokus.value_experiment import write
ROOT=Path('artifacts/ppo-four-channel-runtime');seed_all(1740500);states=[]
for g in json.loads((ROOT/'benchmark-games.json').read_text())[:8]:
 s=State()
 for ply,row in enumerate(g['prefix']+g['moves']):
  if ply in [4,12,24,32] or row is None:states.append(copy.deepcopy(s))
  play(s,Move(*row) if row else None,validate=False)
rows=[]
for device in ['cpu','cuda']:
 for channels in [6,4]:
  agents={b:PPOAgent(device=device,backend=b,checkpoint=ROOT/'pilot/inference.pt' if channels==4 else None) for b in ['python','native']}
  expected=[agents['python'].act(s,random.Random(i)) for i,s in enumerate(states)]
  for repeat in range(4):
   for backend in (['python','native'] if repeat%2==0 else ['native','python']):
    a=agents[backend]
    for i,s in enumerate(states):assert a.act(s,random.Random(i))==expected[i]
    start=time.perf_counter()
    for _ in range(3):
     for i,s in enumerate(states):a.act(s,random.Random(i))
    rows.append(dict(device=device,channels=channels,backend=backend,repeat=repeat,ms_per_move=(time.perf_counter()-start)*1000/(3*len(states))))
write(ROOT/'inference-profile.json',dict(states=len(states),rows=rows,actions_equal=True,warmup='one bank per repeat outside timing; initialization excluded',complete_inference_wall=True))
