"""Fresh main clone with genuinely CPU-only PyTorch, plus CUDA CLI smoke."""
import subprocess,tempfile,json,sys
from pathlib import Path
repo=Path(__file__).resolve().parents[1];out=repo/'artifacts/ppo-generation2';clone=Path(tempfile.mkdtemp(prefix='blokus-gen2-fresh-'))
with (out/'fresh-clone.log').open('w') as log:
 def run(cmd):return subprocess.run(cmd,cwd=clone,stdout=log,stderr=subprocess.STDOUT,check=True)
 subprocess.run(['git','clone','--depth','1','--single-branch','--branch','main','file://'+str(repo),str(clone)],check=True,stdout=log,stderr=subprocess.STDOUT)
 assert not (clone/'artifacts/ppo-learning-curve').exists() and not (clone/'blokus/terminal_ppo.py').exists()
 run(['uv','venv','--python','3.12',str(clone/'.venv')]);py=str(clone/'.venv/bin/python')
 run(['uv','pip','install','--python',py,'torch==2.7.1','--index-url','https://download.pytorch.org/whl/cpu']);run(['uv','pip','install','--python',py,'numpy','pytest'])
 run([py,'-m','pytest','-q','tests/test_ppo_policy.py'])
 # Copy only the validation harness, no training module/history.
 import shutil
 shutil.copyfile(repo/'scripts/check_ppo_generations.py',clone/'check_generations.py')
 # The harness's path insertion is harmless; cwd still resolves this clone's package.
 cpu_out=out/'fresh-clone-cpu-games.json';run([py,str(clone/'check_generations.py'),'--device','cpu','--output',str(cpu_out)])
 run([str(repo/'.venv/bin/python'),'-m','pytest','-q'])
 gpu_out=out/'fresh-clone-cuda-games.json';run([str(repo/'.venv/bin/python'),str(clone/'check_generations.py'),'--device','cuda','--output',str(gpu_out)])
 info=subprocess.check_output([py,'-c','import torch,blokus;print(torch.__version__,torch.cuda.is_available(),blokus.__file__)'],cwd=clone,text=True).strip()
 assert '+cpu False ' in info and str(clone/'blokus/__init__.py') in info
 gpu_module=subprocess.check_output([str(repo/'.venv/bin/python'),'-c','import blokus;print(blokus.__file__)'],cwd=clone,text=True).strip();assert gpu_module==str(clone/'blokus/__init__.py')
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=clone,text=True).strip()
(out/'fresh-clone.json').write_text(json.dumps(dict(pass_=True,clone=str(clone),commit=commit,cpu_environment=info,no_training_module=True,no_history=True,cpu_games=13,cuda_games=13),indent=2)+'\n');print(info,flush=True)
