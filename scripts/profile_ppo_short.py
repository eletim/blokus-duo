"""Separate microprofile after timed games; no model or search tuning."""
import json,time,torch
from blokus.ppo_short_experiment import R,model
from blokus.ppo_short_search import ShortSearch
from blokus.terminal_ppo import observe,collate
from blokus.engine import play,Move
from blokus.research import clone
from blokus.tournament import read_prefixes,prefix_state
from blokus.ai import seed_all
from blokus.value_experiment import write
seed_all(1690000);a=ShortSearch(model(1401));f=read_prefixes(R/'dev-prefixes.json');d=json.loads((R/'dev.json').read_text());byid={p['id']:p for p in f['prefixes']};rows=[]
for game in d['games'][:4]:
 s=prefix_state(byid[game['prefix_id']])
 for ply,row in enumerate(game['moves'],start=len(byid[game['prefix_id']]['moves'])):
  if ply in [8,20,32]:
   t=time.perf_counter();obs,qs=observe(s);cpu_root=(time.perf_counter()-t)*1000
   if qs:
    logits=a.policies([obs])[0];order=sorted(range(len(qs)),key=lambda i:-logits[i])[:8];observations=[];t=time.perf_counter()
    for i in order:
     child=clone(s);play(child,qs[i].move,validate=False);observations.append(observe(child)[0])
    cpu_children=(time.perf_counter()-t)*1000;torch.cuda.reset_peak_memory_stats();t=time.perf_counter();batch=collate(observations,a.device);torch.cuda.synchronize();transfer=(time.perf_counter()-t)*1000
    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record()
    with torch.no_grad():out=a.model(*batch)
    end.record();end.synchronize();gpu=start.elapsed_time(end)
    rows.append(dict(game=game['game'],ply=ply,root_candidates=len(qs),children=len(order),child_max_candidates=max(len(o[2]) for o in observations),CPU_root_observe_ms=cpu_root,CPU_children_observe_ms=cpu_children,collate_transfer_sync_ms=transfer,GPU_forward_kernel_ms=gpu,peak_allocated_bytes=torch.cuda.max_memory_allocated()))
  play(s,Move(*row) if row else None,validate=False)
write(R/'microprofile.json',dict(rows=rows,scope='separate post-evaluation diagnostic; CUDA event is kernel-only, tournament time includes all work',device=torch.cuda.get_device_name(),torch=torch.__version__))
