"""Audit oracle provenance, saved games, model/source hashes and benchmark arithmetic."""
import ast,json,subprocess,hashlib
from pathlib import Path
import numpy as np
from blokus.engine import State,Move,play,is_legal,legal_moves
from blokus.research import sha
from blokus.value_experiment import write
R=Path('artifacts/ppo-cpu-throughput')
reference=ast.parse(Path('blokus/cpu_reference.py').read_text());functions={n.name:n for n in reference.body if isinstance(n,ast.FunctionDef)}
for path,names in [('blokus/engine.py',['legal_moves']),('blokus/ai.py',['features']),('blokus/root_policy.py',['inputs']),('blokus/terminal_ppo.py',['observe'])]:
 old=ast.parse(subprocess.check_output(['git','show','bd962d7:'+path]).decode())
 for n in old.body:
  if isinstance(n,ast.FunctionDef) and n.name in names:assert ast.dump(n)==ast.dump(functions[n.name])
assert Path('blokus/engine.py').read_bytes()==subprocess.check_output(['git','show','bd962d7:blokus/engine.py'])
games=json.loads((R/'games.json').read_text());cli=json.loads((R/'fresh-cpu-cli.json').read_text());humans=json.loads((R/'fresh-cpu-human.json').read_text());rows=games+[cli]+humans['games']
for g in rows:
 s=State()
 for row in g['moves']:
  m=Move(*row) if row else None;assert is_legal(s,m) if m else not legal_moves(s);play(s,m,validate=False)
 assert s.done and [s.score(0),s.score(1)]==g['scores']
training=json.loads((R/'training.json').read_text())
for rep in range(3):
 old,new=[next(r for r in training['rows'] if r['repeat']==rep and r['backend']==b) for b in ['old','new']]
 assert old['inputs']==new['inputs'] and old['games_sha256']==new['games_sha256']
 for a,b in zip(old['logs'],new['logs']):
  assert {k:v for k,v in a['update'].items() if k!='seconds'}=={k:v for k,v in b['update'].items() if k!='seconds'}
 for r in [old,new]:assert abs(32/r['generation_seconds']-r['games_per_second'])<1e-10
micro=json.loads((R/'micro-after.json').read_text());assert micro['games_sha256']==sha(R/'games.json');assert micro['model_sha256']==sha('artifacts/ppo-gen2/seed-1401.pt')
measurements={}
for op in sorted({r['operation'] for r in micro['rows']}):
 values={b:float(np.median([r['ms_per_state'] for r in micro['rows'] if r['operation']==op and r['backend']==b])) for b in ['old','new']};values['speedup']=values['old']/values['new'];measurements[op]=values
train={}
for b in ['old','new']:
 rs=[r for r in training['rows'] if r['backend']==b]
 train[b]=dict(wall_median=float(np.median([r['wall'] for r in rs])),games_per_second_median=float(np.median([r['games_per_second'] for r in rs])),CPU_generation_pack_median=float(np.median([sum(l['generation']['cpu_generation_pack'] for l in r['logs']) for r in rs])),GPU_inference_sync_median=float(np.median([sum(l['generation']['gpu_inference_sync'] for l in r['logs']) for r in rs])),update_wall_median=float(np.median([sum(l['update']['seconds'] for l in r['logs']) for r in rs])))
write(R/'audit-summary.json',dict(pass_=True,oracle_AST_matches='bd962d7',engine_unchanged=True,game_replays=len(rows),training_trajectory_inputs_logs_equal=True,training_reward_replay=training['reward_action_legal_replay_pass'],measurements=measurements,training=train,source_hashes={p:sha(p) for p in ['blokus/ai.py','blokus/root_policy.py','blokus/terminal_ppo.py','blokus/placement_geometry.py','blokus/cpu_reference.py','scripts/bench_cpu_throughput.py']},model_sha256=micro['model_sha256']))
print('oracle, games, arithmetic, training equal PASS')
