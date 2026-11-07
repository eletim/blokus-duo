"""Run as separate CUDA processes; no model selection."""
from h_distill_train import *
p=argparse.ArgumentParser();p.add_argument('mode',choices=['full','part','resume','audit']);a=p.parse_args();setup()
path=R/'resume-test';path.mkdir(exist_ok=True)
if a.mode=='audit':
 x=torch.load(path/'full.pt',weights_only=False,map_location='cpu');y=torch.load(path/'resume.pt',weights_only=False,map_location='cpu')
 def eq(a,b):
  if torch.is_tensor(a):return torch.equal(a,b)
  if isinstance(a,dict):return a.keys()==b.keys() and all(eq(v,b[k]) for k,v in a.items())
  if isinstance(a,(list,tuple)):return len(a)==len(b) and all(eq(v,w) for v,w in zip(a,b))
  return a==b
 assert eq(x,y)
 write(path/'result.json',dict(pass_=True,separate_process=True,device='cuda',comparison='12 updates vs 5 + save/load + 7 updates',model_optimizer_pythonless_numpy_torch_cuda_rng_equal=True))
else:
 data=cache(PPOAgent(1751401,'cpu'),json.loads((R/'pilot-1751401.json').read_text())['labels']);t=Trainer(1751401,data)
 torch.manual_seed(11901);torch.cuda.manual_seed_all(11901)
 if a.mode=='resume':t.load(path/'part.pt')
 for _ in range({'full':12,'part':5,'resume':7}[a.mode]):t.step()
 t.save(path/f'{a.mode}.pt')
