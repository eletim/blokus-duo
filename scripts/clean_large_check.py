"""Clean Git checkout of new CPU exports without histories/indexes; CUDA hidden."""
import os,subprocess,sys,tempfile,json,shutil
from pathlib import Path
R=Path('artifacts/value-placement-24576');root=Path.cwd();commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();target=Path(tempfile.mkdtemp(prefix='placement-large-clean-'));env=dict(os.environ,PYTHONPATH='.:scripts',CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1');rows=[]
paths=['blokus','scripts','tests','models','pyproject.toml','requirements.lock.txt','README.md','FROZEN_GEN3.md','artifacts/ppo-gen3','artifacts/ppo-gen2','artifacts/ppo-2048','artifacts/best-trained.pt','artifacts/research-a2/selected.json','artifacts/ppo-cpu-throughput/games.json','artifacts/value-search/compact/seed-201/predictive-best.pt']+[str(R/f'{v}-{s}.pt') for v in 'AC' for s in (1751401,1751402,1751403)]
archive=subprocess.Popen(['git','archive',commit,*paths],stdout=subprocess.PIPE);subprocess.run(['tar','-x','-C',str(target)],stdin=archive.stdout,check=True);archive.stdout.close();assert archive.wait()==0
commands=[[sys.executable,'-m','blokus.ppo_native','build'],[sys.executable,'scripts/verify_frozen_gen3.py'],[sys.executable,'-m','pytest','-q']]
for seed in (1751401,1751402,1751403):
 for v in 'AC':commands.append([sys.executable,'scripts/placement_value_play.py','--checkpoint',str(R/f'{v}-{seed}.pt'),'--backend','python','--max-depth','2','--budget-ms','500'])
for i,cmd in enumerate(commands):
 log=R/f'clean-{i}.log'
 with log.open('w') as h:subprocess.run(cmd,cwd=target,env=env,stdout=h,stderr=subprocess.STDOUT,check=True)
 rows.append(dict(command=cmd,log=str(log)))
# Human input is deterministic legal index0, both seats; no server/production data.
for seat in (0,1):
 cmd=[sys.executable,'scripts/placement_value_play.py','--checkpoint',str(R/'A-1751401.pt'),'--backend','python','--max-depth','2','--human-seat',str(seat)];log=R/f'clean-human-{seat}.log'
 with log.open('w') as h:subprocess.run(cmd,cwd=target,env=env,input='0\n'*50,text=True,stdout=h,stderr=subprocess.STDOUT,check=True)
 rows.append(dict(command=cmd,log=str(log),scripted_human=True))
from audit_placement import replay
replayed=0
for row in rows:
 if 'scripts/placement_value_play.py' not in row['command']:continue
 lines=Path(row['log']).read_text().splitlines();game=json.loads(lines[-1]);assert replay(game['moves'])==game['scores'];replayed+=1
assert replayed==8
(R/'clean-check.json').write_text(json.dumps(dict(pass_=True,commit=commit,commands=rows,paths=paths,cuda_hidden=True,history_free=True,independently_replayed_cli_games=replayed,environment='existing torch 2.7.1+cu128 wheel with CUDA hidden; not an isolated CPU wheel installation'),indent=2));shutil.rmtree(target)
