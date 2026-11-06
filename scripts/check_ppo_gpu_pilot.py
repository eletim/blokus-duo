import json,subprocess,sys,statistics
from pathlib import Path
import torch
from scripts.bench_ppo_runtime import equal
root=Path('artifacts/ppo-four-gpu');paths={}
for mode in ('baseline','fast'):
 path=Path('/tmp/ppo-gpu-pilot-'+mode);subprocess.run([sys.executable,'-m','blokus.ppo_gpu_train','--output',str(path),'--seed','1750500','--until','16','--mode',mode,'--cap','60'],check=True);paths[mode]=path
x=torch.load(paths['baseline']/'iteration-16.pt',weights_only=False);y=torch.load(paths['fast']/'iteration-16.pt',weights_only=False)
keys=['model','past','optimizer','python_rng','numpy_rng','torch_rng','cuda_rng'];assert all(equal(x[k],y[k]) for k in keys);assert [l['games_sha256'] for l in x['logs']]==[l['games_sha256'] for l in y['logs']]
results={mode:json.loads((path/'result-16.json').read_text()) for mode,path in paths.items()}
(root/'pilot-comparison.json').write_text(json.dumps(dict(bit_equal=keys+['games'],results=results,warmup_note='first iteration initialization separately visible in curve; compare iterations2-16'),indent=2));print('PILOT BIT EQUAL')
# Separate processes, including past swap at 4; resume graph has no hidden state.
p=Path('/tmp/ppo-gpu-resume')
subprocess.run([sys.executable,'-m','blokus.ppo_gpu_train','--output',str(p),'--seed','1750500','--until','2','--cap','60'],check=True)
subprocess.run([sys.executable,'-m','blokus.ppo_gpu_train','--output',str(p),'--seed','1750500','--until','16','--resume',str(p/'iteration-2.pt'),'--cap','60'],check=True)
z=torch.load(p/'iteration-16.pt',weights_only=False);assert all(equal(y[k],z[k]) for k in keys);assert [l['games_sha256'] for l in y['logs']]==[l['games_sha256'] for l in z['logs']]
(root/'resume-audit.json').write_text(json.dumps(dict(pass_=True,separate_process=True,continuous=16,first=2,additional=14,past_swaps=[4,8,12,16],bit_equal=keys+['games'],workers=None),indent=2));print('RESUME BIT EQUAL')
