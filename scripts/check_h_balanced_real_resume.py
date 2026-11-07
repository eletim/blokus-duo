"""Run as separate CUDA processes; no model selection."""
from h_distill_train import *
R=Path("artifacts/h-distill-balanced")
p=argparse.ArgumentParser();p.add_argument('mode',choices=['full','part','resume','audit']);p.add_argument('--arm',choices=['A','B'],required=True);a=p.parse_args();setup()
path=R/('resume-real-'+a.arm);path.mkdir(exist_ok=True)
if a.mode=='audit':
 x=torch.load(path/'full.pt',weights_only=False,map_location='cpu');y=torch.load(path/'resume.pt',weights_only=False,map_location='cpu')
 def eq(a,b):
  if torch.is_tensor(a):return torch.equal(a,b)
  if isinstance(a,dict):return a.keys()==b.keys() and all(eq(v,b[k]) for k,v in a.items())
  if isinstance(a,(list,tuple)):return len(a)==len(b) and all(eq(v,w) for v,w in zip(a,b))
  return a==b
 assert eq(x,y)
 write(path/'result.json',dict(pass_=True,separate_process=True,device='cuda',comparison='20 updates vs 7 + save/load + 13 updates (crosses epoch boundary)',model_optimizer_pythonless_numpy_torch_cuda_rng_equal=True))
else:
 data=cache(PPOAgent(1751401,'cpu'),json.loads((R/f'{a.arm}-train-1751401.json').read_text())['labels']);t=Trainer(1751401,data)
 torch.manual_seed(11901);torch.cuda.manual_seed_all(11901)
 if a.mode=='resume':t.load(path/'part.pt')
 for _ in range({'full':20,'part':7,'resume':13}[a.mode]):t.step()
 t.save(path/f'{a.mode}.pt')
