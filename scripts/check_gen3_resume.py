import subprocess,torch,json,tempfile,sys
from pathlib import Path
root=Path(tempfile.mkdtemp(prefix='gen3-training-check-'))
def equal(a,b):
 import numpy as np
 if torch.is_tensor(a):return torch.equal(a,b)
 if isinstance(a,np.ndarray):return np.array_equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return a==b
rows=[]
for device in (['cpu','cuda'] if torch.cuda.is_available() else ['cpu']):
 for name,until,resume in [('full',4,None),('split',2,None),('split',4,str(root/f'{device}-split/latest.pt'))]:
  out=root/f'{device}-{name}';cmd=[sys.executable,'-m','blokus.ppo_runtime','--output',str(out),'--iterations',str(until),'--seed','1911','--channels','4','--device',device,'--backend','native','--packed']
  if resume:cmd+=['--resume',resume] # omit settings: must inherit64/3e-4
  else:cmd+=['--minibatch','64','--learning-rate','0.0003']
  subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL)
 a=torch.load(root/f'{device}-full/latest.pt',weights_only=False);b=torch.load(root/f'{device}-split/latest.pt',weights_only=False)
 keys=['model','past','optimizer','python_rng','numpy_rng','torch_rng','cuda_rng'];assert all(equal(a[k],b[k]) for k in keys)
 assert b['minibatch']==64 and b['learning_rate']==.0003 and b['optimizer']['param_groups'][0]['lr']==.0003
 base=[sys.executable,'-m','blokus.ppo_runtime','--output',str(root/f'{device}-change'),'--iterations','4','--seed','1911','--channels','4','--device',device,'--backend','native','--packed','--resume',str(root/f'{device}-full/latest.pt'),'--minibatch','16','--learning-rate','.0006']
 q=subprocess.run(base,capture_output=True,text=True);assert q.returncode!=0 and 'allow-training-change' in q.stderr
 subprocess.run(base+['--allow-training-change'],check=True,stdout=subprocess.DEVNULL)
 c=torch.load(root/f'{device}-change/latest.pt',weights_only=False);assert c['training_changes'][-1]['new_minibatch']==16 and all(g['lr']==.0006 for g in c['optimizer']['param_groups'])
 rows.append(dict(device=device,bit_equal=True,keys=keys,settings_inherited=True,implicit_change_rejected=True,explicit_change_recorded=True,comparison='4 vs 2 + separate process resume2; past swap4'))
Path('artifacts/ppo-gen3/resume-audit.json').write_text(json.dumps(rows,indent=2)+'\n')
