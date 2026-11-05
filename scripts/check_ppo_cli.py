import sys,subprocess,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from blokus.engine import State,Move,play
from blokus.value_experiment import write
rows=[]
for device in ('cpu','cuda'):
 for seed in (1401,1402,1403):
  cmd=[sys.executable,'-m','blokus.ppo_play','--generation','2048','--device',device,'--model-seed',str(seed)];p=subprocess.run(cmd,text=True,capture_output=True,check=True);r=json.loads(p.stdout.splitlines()[-1]);s=State()
  for m in r['moves']:play(s,Move(*m) if m else None)
  assert s.done and r['scores']==[s.score(0),s.score(1)];rows.append(dict(device=device,seed=seed,scores=r['scores'],plies=len(r['moves']),human=False))
for seat in (0,1):
 p=subprocess.run([sys.executable,'-m','blokus.ppo_play','--generation','2048','--device','cpu','--human-seat',str(seat)],input='-1\n99999\nnonsense\n'+'0\n'*50,text=True,capture_output=True,check=True);r=json.loads(p.stdout.splitlines()[-1]);assert '表示された合法手番号' in p.stdout;s=State()
 for m in r['moves']:play(s,Move(*m) if m else None)
 assert s.done;rows.append(dict(device='cpu',seed=1401,human=True,human_seat=seat,scores=r['scores'],plies=len(r['moves'])))
write('artifacts/ppo-2048/cli-check.json',dict(pass_=True,games=rows));print('CLI PASS',rows)
