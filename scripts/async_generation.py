"""Arrival-driven shared-slot GPU generation; one outstanding request per game."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,json,time,multiprocessing as mp
from multiprocessing.connection import wait
from collections import deque
from pathlib import Path
import numpy as np
import torch
import frontier_generation as f
from blokus.ppo_frontier import FrontierEvaluator
from blokus.ppo_policy import PPOAgent

CAP=4096
class Slots:
 def __init__(self,ctx,n):
  self.n=n
  self.raw=[ctx.RawArray('B',n*4*196),ctx.RawArray('f',n*49),ctx.RawArray('f',n*CAP*39),ctx.RawArray('i',n*CAP*5),ctx.RawArray('f',n*CAP),ctx.RawArray('d',n)]
 def arrays(self):
  shapes=[(self.n,4,14,14),(self.n,49),(self.n,CAP,39),(self.n,CAP,5),(self.n,CAP),(self.n,)]
  types=[np.uint8,np.float32,np.float32,np.int32,np.float32,np.float64]
  return [np.frombuffer(r,dtype=t).reshape(s) for r,t,s in zip(self.raw,types,shapes)]

def worker(wid,pipe,slots,offset,count,directory,fast):
 try:
  os.sched_setaffinity(0,{wid});torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
  b,st,x,c,l,v=slots.arrays()
  if fast:
   import blokus.ppo_frontier as frontier
   from blokus.generation_encoding import encode,child
   frontier.value_inputs=encode;frontier.child=child
  pipe.send(('ready',))
  while True:
   cmd=pipe.recv()
   if cmd[0]=='stop':return
   _,seed,tasks=cmd;p=PPOAgent(seed,'cpu',backend='native')
   if fast:
    from blokus.generation_encoding import observe
    p._observation=lambda state:observe(state,p.channels)
   todo=iter(tasks);active={};serial=0;completed=0;pack=0.;cpu=time.process_time();start=time.perf_counter();peak=0
   def advance(slot,g,answer=None,initial=False):
    nonlocal serial,completed,pack,peak
    while True:
     try:req=next(g) if initial else g.send(answer)
     except StopIteration:
      completed+=1
      try:g=f.game(next(todo),p,directory);initial=True;continue
      except StopIteration:active.pop(slot,None);return
     tick=time.perf_counter();kind,obs=req;n=len(obs[2]) if kind=='policy' else 0
     if n>CAP:raise RuntimeError(f'candidate capacity exceeded: {n}>{CAP}')
     b[slot]=obs[0];st[slot]=obs[1]
     if n:x[slot,:n]=obs[2];c[slot,:n]=obs[3]
     serial+=1;active[slot]=(g,serial,kind,n);peak=max(peak,len(active));pack+=time.perf_counter()-tick
     pipe.send(('request',slot,serial,kind,n,time.perf_counter()));return
   for slot in range(offset,offset+count):
    try:advance(slot,f.game(next(todo),p,directory),initial=True)
    except StopIteration:break
   pipe.send(('occupancy',len(active),completed))
   while active:
    msg=pipe.recv()
    if msg[0]!='reply':raise RuntimeError(msg)
    for slot,rid in msg[1]:
     g,expected,kind,n=active[slot];assert rid==expected
     answer=(l[slot,:n].copy(),float(v[slot])) if kind=='policy' else float(v[slot])
     previous=completed;advance(slot,g,answer)
     if completed!=previous:pipe.send(('occupancy',len(active),completed))
   pipe.send(('done',dict(games=completed,cpu_seconds=time.process_time()-cpu,seconds=time.perf_counter()-start,pack_seconds=pack,peak_active=peak)))
 except BaseException as exc:
  pipe.send(('error',repr(exc)));raise
 finally:pipe.close()

def bench(bank_path,out,concurrency=512,batch=128,fast=False,graphs=False):
 assert concurrency%8==0
 out=Path(out);out.mkdir(parents=True,exist_ok=True);assert not (out/'timing.json').exists()
 bank=json.loads(Path(bank_path).read_text());torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 assert all(f.ref.e.sha(f'artifacts/ppo-gen3/seed-{s}.pt')==h for s,h in bank['models'].items())
 ctx=mp.get_context('spawn');slots=Slots(ctx,concurrency);arrays=slots.arrays();b,st,x,c,l,v=arrays
 pairs=[ctx.Pipe() for _ in range(8)];ps=[];init=time.perf_counter();groups=[]
 try:
  for i in range(8):
   p=ctx.Process(target=worker,args=(i,pairs[i][1],slots,i*concurrency//8,concurrency//8,out,fast));p.start();ps.append(p)
  pipes=[q[0] for q in pairs]
  for p in pipes:assert p.recv()[0]=='ready'
  models={s:PPOAgent(s,'cuda',backend='native') for s in sorted({g['seed'] for g in bank['games']})}
  init=time.perf_counter()-init;start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
  for seed,model in models.items():
   tasks=[g for g in bank['games'] if g['seed']==seed];ev=FrontierEvaluator(model)
   if graphs:
    from blokus.generation_graph import GraphEvaluator
    ev=GraphEvaluator(model)
   pending=deque();live=set(pipes);results=[];queue_wait=[];hist={};occupancy={p:0 for p in pipes};trace=[];tick=time.perf_counter();server_cpu=time.process_time()
   for i,p in enumerate(pipes):p.send(('start',seed,tasks[i::8]))
   while live or pending:
    ready=wait(list(live),timeout=0 if pending else None) if live else []
    for p in ready:
     # Bounded draining allows GPU dispatch while other workers continue advancing.
     for _ in range(batch):
      msg=p.recv()
      if msg[0]=='error':raise RuntimeError(msg)
      if msg[0]=='occupancy':
       occupancy[p]=msg[1];trace.append([time.perf_counter()-tick,sum(occupancy.values()),len(pending)])
       if not p.poll():break
       continue
      if msg[0]=='done':results.append(msg[1]);live.remove(p);break
      assert msg[0]=='request';pending.append((p,msg))
      if not p.poll():break
    if not pending:continue
    selected=[pending.popleft() for _ in range(min(batch,len(pending)))];requests=[]
    for p,(_,slot,rid,kind,n,created) in selected:
     queue_wait.append(time.perf_counter()-created)
     requests.append((kind,(b[slot],st[slot],x[slot,:n],c[slot,:n]) if n else (b[slot],st[slot])))
    answers=ev.batch(requests);replies={};hist[len(selected)]=hist.get(len(selected),0)+1
    for (p,(_,slot,rid,kind,n,_)),answer in zip(selected,answers):
     if kind=='policy':l[slot,:n]=answer[0];v[slot]=answer[1]
     else:v[slot]=answer
     replies.setdefault(p,[]).append((slot,rid))
    for p,reply in replies.items():p.send(('reply',reply))
   groups.append(dict(seed=seed,seconds=time.perf_counter()-tick,server_cpu_seconds=time.process_time()-server_cpu,workers=results,occupancy_trace=trace,graph_capture_seconds=getattr(ev,"capture_seconds",0),graph_count=len(getattr(ev,"graphs",{})),h2d_seconds=getattr(ev,"h2d_seconds",0),padding_real=getattr(ev,"padding_real",0),padding_slots=getattr(ev,"padding_slots",0),dispatch_hist=hist,batch_hist=ev.hist,pack_seconds=ev.pack_seconds,forward_seconds=ev.forward_seconds,return_seconds=ev.return_seconds,queue_wait_quantiles=np.quantile(queue_wait,[0,.5,.95,.99,1]).tolist() if queue_wait else []))
  result=dict(seconds=time.perf_counter()-start,initialization_seconds=init,concurrency=concurrency,batch_limit=batch,fast_encoding=fast,graphs=graphs,groups=groups,games=len(bank['games']),bank_sha=f.ref.e.sha(bank_path),peak_memory=torch.cuda.max_memory_allocated())
  f.ref.write(out/'timing.json',result);print(json.dumps(result),flush=True)
 finally:
  for p,_ in pairs:
   try:p.send(('stop',))
   except (EOFError,BrokenPipeError):pass
  for p in ps:
   p.join(5)
   if p.is_alive():p.terminate();p.join()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--bank',required=True);p.add_argument('--out',required=True);p.add_argument('--concurrency',type=int,default=512);p.add_argument('--batch',type=int,default=128);p.add_argument('--fast-encoding',action='store_true');p.add_argument('--graphs',action='store_true');a=p.parse_args();bench(a.bank,a.out,a.concurrency,a.batch,a.fast_encoding,a.graphs)
