"""One diagnostic update with CUDA events; synchronization instrumentation is NOT throughput benchmark."""
import copy,json,time
from pathlib import Path
import torch,numpy as np
from blokus import ppo_runtime as r
from blokus.ppo_policy import TerminalNet,collate
R=Path('artifacts/ppo-four-gpu');r.seed_all(1750400);m=TerminalNet(4).cuda();records,_,_=r.collect(m,copy.deepcopy(m),0,1750400,backend='native');opt=torch.optim.Adam(m.parameters(),lr=.0003)
r.update(m,opt,records);r.seed_all(1750401);ret=torch.tensor([x['return'] for x in records],device='cuda');oldv=torch.tensor([x['value'] for x in records],device='cuda');adv=ret-oldv;adv=(adv-adv.mean())/adv.std(unbiased=False).clamp_min(1e-6);lp0=torch.tensor([x['log_prob'] for x in records],device='cuda');actions=torch.tensor([x['action'] for x in records],device='cuda');rows=[]
for ix in torch.randperm(len(records)).split(16):
 ids=ix.tolist();obs=[records[i]['obs'] for i in ids];t=time.perf_counter();batch=collate(obs,'cuda');torch.cuda.synchronize();transfer=time.perf_counter()-t;ev=[torch.cuda.Event(enable_timing=True) for _ in range(6)];ev[0].record();logits,v=m(*batch);ev[1].record();dist=torch.distributions.Categorical(logits=logits,validate_args=False);lp=dist.log_prob(actions[ids]);ratio=(lp-lp0[ids]).exp();pl=-torch.minimum(ratio*adv[ids],ratio.clamp(.8,1.2)*adv[ids]).mean();vl=(v-ret[ids]).square().mean();entropy=dist.entropy().mean();loss=pl+.5*vl-.01*entropy;ev[2].record();opt.zero_grad();loss.backward();ev[3].record();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),.5);ev[4].record();opt.step();ev[5].record();ev[5].synchronize();t=time.perf_counter();metrics=torch.stack((loss,pl,vl,entropy,norm)).detach().cpu().tolist();logwall=time.perf_counter()-t
 rows.append(dict(batch=len(ids),real_candidates=sum(len(o[2]) for o in obs),padded_candidates=len(ids)*batch[2].shape[1],collate_h2d_sync_ms=transfer*1000,forward_gpu_ms=ev[0].elapsed_time(ev[1]),loss_gpu_ms=ev[1].elapsed_time(ev[2]),backward_gpu_ms=ev[2].elapsed_time(ev[3]),norm_gpu_ms=ev[3].elapsed_time(ev[4]),optimizer_gpu_ms=ev[4].elapsed_time(ev[5]),log_d2h_wall_ms=logwall*1000))
(R/'phase-events.json').write_text(json.dumps(dict(rows=rows,diagnostic_only=True,throughput_use='update-benchmark.json',padding_ratio=sum(r['padded_candidates'] for r in rows)/sum(r['real_candidates'] for r in rows)),indent=2))
