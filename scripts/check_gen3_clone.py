import subprocess,json,time,os,sys
from pathlib import Path
clone=Path(sys.argv[1]).resolve();rows=[]
for device,exe in [('cpu',sys.argv[2]),('cuda',sys.argv[3])]:
 if device=='cuda':subprocess.run([exe,'-m','blokus.ppo_native','build'],cwd=clone,check=True,capture_output=True)
 else:
  subprocess.run([exe,'-c','import torch; assert torch.version.cuda is None; from blokus.ppo_native import library; assert library() is None'],cwd=clone,check=True)
 for gen in ('gen3','gen2','2048'):
  for human in (None,0,1):
   cmd=[exe,'-m','blokus.ppo_play','--generation',gen,'--device',device,'--backend','auto','--opponent','ppo','--opponent-generation','gen2']
   if human is not None:cmd+=['--human-seat',str(human)]
   t=time.perf_counter();p=subprocess.run(cmd,cwd=clone,input='0\n' if human is not None else '',text=True,capture_output=True,check=True)
   if human is None:
    result=json.loads(p.stdout.splitlines()[-1]);assert len(result['moves'])>20;details=dict(scores=result['scores'],moves=len(result['moves']),model_seed=result['model_seed'])
   else:
    assert '合法手番号' in p.stdout and '対局を終了しました' in p.stdout;details=dict(valid_move_entered=True,eof_exit=True)
   rows.append(dict(device=device,python=exe,generation=gen,human_seat=human,wall=time.perf_counter()-t,pass_=True,**details))
# Verify errors are concise and legacy seed aliases still load.
exe=sys.argv[2]
p=subprocess.run([exe,'-m','blokus.ppo_play','--generation','gen2','--model-seed','1751401'],cwd=clone,capture_output=True,text=True);assert p.returncode!=0 and 'gen2 requires model seed' in p.stderr and 'Traceback' not in p.stderr
p=subprocess.run([exe,'-m','blokus.ppo_play','--generation','gen3','--model-seed','1401','--device','cpu'],cwd=clone,capture_output=True,text=True,check=True);assert json.loads(p.stdout.splitlines()[-1])['model_seed']==1751401
Path('artifacts/ppo-gen3/clone-smoke.json').write_text(json.dumps(dict(rows=rows,cpu_only=True,history_required=False,invalid_seed_clear=True,legacy_alias=True),indent=2)+'\n')
