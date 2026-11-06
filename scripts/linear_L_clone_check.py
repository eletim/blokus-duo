"""Fresh-clone CPU smoke, no history/index or training modules required."""
import gzip,json,os,subprocess,time
from pathlib import Path
from scripts import linear_L_confirm as ev
root=Path('/tmp/blokus-linear-L-fresh-clone');python='/tmp/blokus-cpu-only-venv312/bin/python';rows=[]
assert not (root/'blokus/linear_train2.py').exists() and not (root/'scripts/tuple_cycle.py').exists() and not (root/'artifacts/linear-tuple').exists()
for backend,seed,seat in [('native',3101,None),('python',3102,None),('native',3103,0),('python',3101,1)]:
 cmd=[python,'-m','blokus.non_neural_play','--backend',backend,'--model-seed',str(seed)]+([] if seat is None else ['--human-seat',str(seat)]);t=time.perf_counter();p=subprocess.run(cmd,cwd=root,input='invalid\n'+'0\n'*100,text=True,capture_output=True,timeout=120,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'});assert p.returncode==0,p.stderr;d=json.loads(p.stdout.splitlines()[-1]);assert len(d['scores'])==2 and len(d['moves'])>0
 if seat is not None:assert '表示された合法手番号' in p.stdout
 name=f'clone-cli-{backend}-{seed}-{seat}.log.gz';(ev.R/name).write_bytes(gzip.compress(p.stdout.encode(),mtime=0));rows.append(dict(command=cmd,seconds=time.perf_counter()-t,scores=d['scores'],plies=len(d['moves']),log=name))
p=subprocess.run([python,'-m','blokus.non_neural_play','--model-seed','1401'],cwd=root,text=True,capture_output=True);assert p.returncode!=0 and 'invalid choice' in p.stderr
(ev.R/'clone-invalid-seed.log').write_text(p.stderr)
ev.write('clone-audit.json',dict(pass_=True,clone=str(root),commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),cpu_environment=subprocess.check_output([python,'-c','import torch;print(torch.__version__,torch.version.cuda)'],text=True).strip(),training_history_required=False,rows=rows,invalid_seed_rejected=True,model_hashes=ev.read(root/'models/linear-L/manifest.json')))
print('CLONE CLI PASS',flush=True)
