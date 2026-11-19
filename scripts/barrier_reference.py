"""Eight CPU coroutine cohorts, one GPU ready-frontier batch per wave, no RPC/node."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,json,multiprocessing as mp,time
import torch
import frontier_generation as f
from blokus.ppo_frontier import FrontierEvaluator,run_frontier
from blokus.ppo_policy import PPOAgent

class BulkEvaluator:
 def __init__(self,pipe):self.pipe=pipe;self.wait=0.;self.calls=0;self.requests=0
 def batch(self,requests):
  t=time.perf_counter();self.pipe.send(('frontier',requests));out=self.pipe.recv();self.wait+=time.perf_counter()-t;self.calls+=1;self.requests+=len(requests)
  if out[0]!='values':raise RuntimeError(out)
  return out[1]

def worker(wid,pipe,directory):
 try:
  os.sched_setaffinity(0,{wid});torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
  ps={s:PPOAgent(s,'cpu',backend='native') for s in f.ref.e.SEEDS};pipe.send(('ready',wid))
  while True:
   command=pipe.recv()
   if command[0]=='stop':break
   _,seed,tasks=command;ev=BulkEvaluator(pipe);cpu=time.process_time();t=time.perf_counter()
   rows=run_frontier([f.game(g,ps[seed],directory) for g in tasks],ev)
   pipe.send(('done',dict(worker=wid,games=len(rows),seconds=time.perf_counter()-t,cpu_seconds=time.process_time()-cpu,wait_seconds=ev.wait,bulk_messages=ev.calls,nn_requests=ev.requests)))
 except BaseException as exc:
  pipe.send(('error',repr(exc)));raise
 finally:pipe.close()

def bench(repeat,root):
 R=__import__("pathlib").Path(root);outpath=R/f'parallel-{repeat}.json';assert not outpath.exists()
 bank=json.loads((R/'bank.json').read_text());directory=R/f'parallel-{repeat}';directory.mkdir(exist_ok=True)
 assert all(f.ref.e.sha(f'artifacts/ppo-gen3/seed-{s}.pt')==h for s,h in bank['models'].items())
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);start=time.perf_counter()
 ctx=mp.get_context('spawn');pairs=[ctx.Pipe() for _ in range(8)];workers=[ctx.Process(target=worker,args=(i,pairs[i][1],directory)) for i in range(8)]
 for p in workers:p.start()
 pipes=[x[0] for x in pairs];agents={s:PPOAgent(s,'cuda',backend='native') for s in f.ref.e.SEEDS}
 try:
  for p in pipes:assert p.recv()[0]=='ready'
  init=time.perf_counter()-start;start=time.perf_counter();cpu=time.process_time();groups=[];torch.cuda.reset_peak_memory_stats()
  for seed,block in [(s,i) for s in f.ref.e.SEEDS for i in range(0,len([g for g in bank["games"] if g["seed"]==s]),64)]:
   tasks=[x for x in bank['games'] if x['seed']==seed][block:block+64];ev=FrontierEvaluator(agents[seed]);t=time.perf_counter();results=[];wait=0.;ipc=0.
   for i,p in enumerate(pipes):p.send(('start',seed,tasks[i::8]))
   active=pipes[:]
   while active:
    frontier=[];slots=[];pending=[];tick=time.perf_counter()
    for p in active:
     msg=p.recv()
     if msg[0]=='error':raise RuntimeError(msg)
     if msg[0]=='done':results.append(msg[1]);continue
     assert msg[0]=='frontier';slots.append((p,len(msg[1])));frontier.extend(msg[1]);pending.append(p)
    wait+=time.perf_counter()-tick
    if frontier:
     answers=ev.batch(frontier);offset=0;tick=time.perf_counter()
     for p,n in slots:p.send(('values',answers[offset:offset+n]));offset+=n
     ipc+=time.perf_counter()-tick
    active=pending
   groups.append(dict(seed=seed,block=block,seconds=time.perf_counter()-t,workers=results,batches=ev.hist,pack_seconds=ev.pack_seconds,forward_seconds=ev.forward_seconds,return_seconds=ev.return_seconds,frontier_wait_receive_seconds=wait,reply_send_seconds=ipc))
  out=dict(seconds=time.perf_counter()-start,initialization_seconds=init,server_cpu_seconds=time.process_time()-cpu,groups=groups,bank_sha=f.ref.e.sha(R/'bank.json'),games=len(bank['games']),peak_memory=torch.cuda.max_memory_allocated())
  f.ref.write(outpath,out);print(json.dumps(out),flush=True)
 finally:
  for p in pipes:
   try:p.send(('stop',))
   except (EOFError,BrokenPipeError):pass
  for p in workers:
   p.join(timeout=5)
   if p.is_alive():p.terminate();p.join()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--repeat',type=int,default=1);p.add_argument('--root',required=True);a=p.parse_args();bench(a.repeat,a.root)
