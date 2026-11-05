"""Read-only checkpoint/history integrity and CPU/CUDA inference after reboot."""
import sys,json,datetime
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from blokus.terminal_ppo import TerminalNet,observe,collate
from blokus.engine import State
from blokus.research import sha
from blokus.ppo_6144_eval import lock
R=Path('artifacts/ppo-6144')
torch.set_num_threads(1)
lock(); selected=json.loads((R/'selection.json').read_text())
assert sha(R/'selection.json')==json.loads((R/'holdout-preflight.json').read_text())['selection_sha256']
checked={};rows=[]
for seed in (1401,1402,1403):
 for iteration in (5120,6144):
  path=R/f'seed-{seed}'/f'iteration-{iteration}.pt'
  c=torch.load(path,map_location='cpu',weights_only=False)
  assert (c['seed'],c['iteration'])==(seed,iteration)
  assert c['optimizer']['state'] and all(k in c for k in ('past','python_rng','numpy_rng','torch_rng','cuda_rng'))
  assert sum(e['games'] for e in c['history_chain'])==iteration*16
  for entry in c['history_chain']:
   hp=(path.parent/entry['path']).resolve()
   if str(hp) not in checked:checked[str(hp)]=sha(hp)
   assert checked[str(hp)]==entry['sha256']
  model=TerminalNet();model.load_state_dict(c['past']);model.load_state_dict(c['model']);model.eval()
  obs=[observe(State())[0]]
  with torch.no_grad():
   cpu=model(*collate(obs,'cpu'));gpu=model.cuda()(*collate(obs,'cuda'));torch.cuda.synchronize()
  for a,b in zip(cpu,gpu):
   assert torch.isfinite(a).all() and torch.isfinite(b).all()
   torch.testing.assert_close(a,b.cpu(),rtol=1e-4,atol=1e-4)
  if selected[str(seed)]['iteration']==iteration:assert sha(path)==selected[str(seed)]['sha256']
  rows.append(dict(seed=seed,iteration=iteration,sha256=sha(path),games=iteration*16,cpu_cuda_close=True))
out=dict(pass_=True,time=datetime.datetime.now().astimezone().isoformat(),torch=torch.__version__,gpu=torch.cuda.get_device_name(),checkpoints=rows,unique_history_files=len(checked),history_sha256=checked,note='No training or timed games rerun. All referenced history bytes checked against checkpoint hashes; prior full legal/reward replay retained. CPU/CUDA inference tolerance 1e-4.')
(R/'reboot-recovery.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ('history_sha256','checkpoints')}))
