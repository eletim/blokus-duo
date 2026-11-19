"""History-free clean checkout with CUDA hidden; no research training indexes needed."""
import os,subprocess,sys,tempfile,json
from pathlib import Path
import frontier_generation as f
root=Path.cwd();target=Path(tempfile.mkdtemp(prefix='async-clean-'));commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();env=dict(os.environ,PYTHONPATH='.:scripts',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1');rows=[]
paths=['blokus','scripts','tests','models','pyproject.toml','requirements.lock.txt','README.md','FROZEN_GEN3.md','artifacts/ppo-gen3','artifacts/ppo-gen2','artifacts/ppo-2048','artifacts/best-trained.pt','artifacts/value-search/compact/seed-201/predictive-best.pt']+[f'artifacts/value-placement-scale/n3072/A-{seed}.pt' for seed in f.ref.e.SEEDS]
archive=subprocess.Popen(['git','archive',commit,*paths],stdout=subprocess.PIPE);subprocess.run(['tar','-x','-C',str(target)],stdin=archive.stdout,check=True);archive.stdout.close();assert archive.wait()==0
commands=[[sys.executable,'-m','blokus.ppo_native','build'],[sys.executable,'scripts/verify_frozen_gen3.py'],[sys.executable,'scripts/check_generation_exports.py'],[sys.executable,'-m','pytest','-q']]+[[sys.executable,'-m','blokus.ppo_play','--generation','gen3','--model-seed',str(seed),'--device','cpu','--backend','python'] for seed in f.ref.e.SEEDS]
for i,cmd in enumerate(commands):
 log=root/f'artifacts/gpu-async512/clean-{i}.log'
 with log.open('w') as handle:subprocess.run(cmd,cwd=target,env=env,stdout=handle,stderr=subprocess.STDOUT,check=True)
 rows.append(dict(command=cmd,log=str(log.relative_to(root))))
f.ref.write(root/'artifacts/gpu-async512/clean-check.json',dict(commit=commit,commands=rows,cuda_hidden=True,environment='same torch2.7.1+cu128 wheel with CUDA hidden; not a separately installed CPU-only wheel',history_free=True,archive_paths=paths))
# This is an exclusively created disposable archive, not a user checkout.
import shutil
shutil.rmtree(target)
