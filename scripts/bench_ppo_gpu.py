import copy,json,time
from pathlib import Path
import torch,numpy as np
from blokus import ppo_runtime as r
from blokus.ppo_gpu import update_fast
from blokus.ppo_policy import TerminalNet
from scripts.bench_ppo_runtime import equal,records_hash,tensor_hash
root=Path('artifacts/ppo-four-gpu');rows=[]
for rep in range(3):
 r.seed_all(1750100+rep);base=TerminalNet(4).cuda();past=copy.deepcopy(base);records,games,gen=r.collect(base,past,0,1750100+rep,backend='native');outputs={}
 for mode in (['baseline','sync'] if rep%2==0 else ['sync','baseline']):
  m=copy.deepcopy(base);opt=torch.optim.Adam(m.parameters(),lr=.0003);r.seed_all(1750200+rep)
  fn=r.update if mode=='baseline' else update_fast
  start=time.perf_counter();metrics=fn(m,opt,records);torch.cuda.synchronize();wall=time.perf_counter()-start
  outputs[mode]=(copy.deepcopy(m.state_dict()),copy.deepcopy(opt.state_dict()),torch.get_rng_state(),torch.cuda.get_rng_state_all());rows.append(dict(rep=rep,mode=mode,wall=wall,generation=gen,update=metrics,states=len(records)))
 assert equal(outputs['baseline'],outputs['sync'])
print(rows);(root/'sync-benchmark.json').write_text(json.dumps(dict(rows=rows,bit_equal=True),indent=2))
