"""CLI game replay checks for both public PPO generations."""
import argparse,json,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from blokus.engine import State,Move,play
p=argparse.ArgumentParser();p.add_argument('--output',default='artifacts/ppo-gen2/cli-check.json');p.add_argument('--device',choices=['cpu','cuda'],default='cpu');a=p.parse_args();rows=[]
def game(args,stdin=None):
 cmd=[sys.executable,'-m','blokus.ppo_play','--device',a.device]+args
 r=subprocess.run(cmd,input=stdin,text=True,capture_output=True,check=True);v=json.loads(r.stdout.splitlines()[-1]);s=State()
 for m in v['moves']:play(s,Move(*m) if m else None)
 assert s.done and v['scores']==[s.score(0),s.score(1)];rows.append(dict(arguments=args,plies=len(v['moves']),scores=v['scores'],generation=v['generation'],iteration=v['iteration']))
for gen in ('gen2','2048'):
 for seed in (1401,1402,1403):game(['--generation',gen,'--model-seed',str(seed)])
 for seat in (0,1):game(['--generation',gen,'--human-seat',str(seat)],'-1\n99999\nx\n'+'0\n'*50)
game(['--generation','gen2','--opponent','ppo','--opponent-generation','2048'])
for op in ('oldvalue','teacher100'):game(['--opponent',op])
Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(dict(pass_=True,device=a.device,games=rows),indent=2)+'\n');print('CLI replay PASS',len(rows))
