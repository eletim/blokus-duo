"""Fresh-process startup and first iteration; deliberately no warmup."""
import time
start=time.perf_counter()
import copy,json
from pathlib import Path
import torch
from blokus import ppo_runtime as r,ppo_native as n
imported=time.perf_counter();n.tables();tables=time.perf_counter();r.seed_all(1740600);m=r.TerminalNet(4).cuda();past=copy.deepcopy(m);opt=torch.optim.Adam(m.parameters(),lr=.0003);torch.cuda.synchronize();initialized=time.perf_counter();rec,games,gen=r.collect(m,past,0,1740600,backend='native');metrics=r.update(m,opt,rec);end=time.perf_counter();saving=r.save('/tmp/ppo-runtime-cold.pt',m,past,opt,1,1740600,'native',False,[])
Path('artifacts/ppo-four-channel-runtime/cold-start.json').write_text(json.dumps(dict(import_seconds=imported-start,table_initialization_seconds=tables-imported,CUDA_model_optimizer_initialization_seconds=initialized-tables,first_iteration_seconds=end-initialized,whole_process_through_first_iteration=end-start,save_seconds=saving,generation=gen,update=metrics,warmup=False,torch_compile=False),indent=2))
