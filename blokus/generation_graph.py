"""Opt-in inference-only CUDA graph buckets. Frozen network, float32, no training."""
import time
import numpy as np
import torch
from .ppo_frontier import FrontierEvaluator

def bucket(n):return max(8,1<<(n-1).bit_length())
class GraphEvaluator(FrontierEvaluator):
 def __init__(self,policy):
  super().__init__(policy);self.graphs={};self.capture_seconds=0.;self.h2d_seconds=0.;self.padding_real=0;self.padding_slots=0;self.pool=torch.cuda.graph_pool_handle();self.transfer=torch.cuda.Event(enable_timing=True);self.end=torch.cuda.Event(enable_timing=True);self.begin=torch.cuda.Event(enable_timing=True);self.sync_dispatch_seconds=0.
 def make(self,kind,b,n):
  t=time.perf_counter();spec=[((b,4,14,14),torch.uint8),((b,49),torch.float32)]
  if kind=='policy':spec += [((b,n,39),torch.float32),((b,n,5),torch.int64),((b,n),torch.bool)]
  host=[torch.zeros(shape,dtype=dtype,pin_memory=True) for shape,dtype in spec];device=[h.to('cuda') for h in host]
  if kind=='policy':host[3].fill_(196);device[3].fill_(196)
  def forward():
   if kind=='policy':return self.policy.model(*device)
   board,st=device;m=self.policy.model;g=m.global_net(m.conv(board.float()));return m.value(torch.cat((g,st),-1)).squeeze(-1)
  stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(stream),torch.inference_mode():
   for _ in range(3):forward()
  torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph,pool=self.pool),torch.inference_mode():output=forward()
  self.capture_seconds+=time.perf_counter()-t
  return host,device,graph,output
 def batch(self,requests):
  answers=[None]*len(requests)
  for kind in ('policy','value'):
   ix=[i for i,r in enumerate(requests) if r[0]==kind]
   if not ix:continue
   actual=len(ix);b=bucket(actual);n=bucket(max(len(requests[i][1][2]) for i in ix)) if kind=='policy' else 0;k=(kind,b,n)
   if k not in self.graphs:self.graphs[k]=self.make(*k)
   host,device,graph,output=self.graphs[k];arr=[h.numpy() for h in host];t=time.perf_counter()
   if kind=='policy':arr[4].fill(False)
   for j,i in enumerate(ix):
    obs=requests[i][1];arr[0][j]=obs[0];arr[1][j]=obs[1]
    if kind=='policy':
     count=len(obs[2]);arr[2][j,:count]=obs[2];arr[3][j,:count]=obs[3];arr[4][j,:count]=True;self.padding_real+=count
   if kind=='policy':self.padding_slots+=b*n
   self.pack_seconds+=time.perf_counter()-t;t=time.perf_counter()
   self.begin.record()
   for d,h in zip(device,host):d.copy_(h,non_blocking=True)
   # Events separate device transfer / replay without a synchronization per input.
   transfer=self.transfer;end=self.end
   transfer.record();graph.replay();end.record();end.synchronize()
   self.forward_seconds+=transfer.elapsed_time(end)/1000
   self.h2d_seconds+=self.begin.elapsed_time(transfer)/1000
   self.sync_dispatch_seconds+=time.perf_counter()-t-self.begin.elapsed_time(end)/1000
   t=time.perf_counter()
   if kind=='policy':ll=output[0][:actual].cpu().numpy();vv=output[1][:actual].cpu().numpy()
   else:vv=output[:actual].cpu().numpy()
   for j,i in enumerate(ix):answers[i]=(ll[j,:len(requests[i][1][2])].copy(),float(vv[j])) if kind=='policy' else float(vv[j])
   self.return_seconds+=time.perf_counter()-t;self.hist[actual]=self.hist.get(actual,0)+1
  return answers
