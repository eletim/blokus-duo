"""Frozen-model inference broker. CPU engines send arrays, never CUDA tensors.

One outstanding request per worker; response ids detect stale/mixed replies.
No learning, random sampling, rule transitions or search decisions in this module.
"""
import time
import queue
from collections import Counter, defaultdict
import numpy as np
import torch
from .ppo_policy import collate
from .ppo_value_leaf import value_inputs

class InferenceClient:
    def __init__(self, worker, seed, requests, replies):
        self.worker, self.seed = worker, seed
        self.requests, self.replies = requests, replies
        self.serial = 0
        self.wait_seconds = 0.
    def call(self, kind, arrays):
        self.serial += 1
        start = time.perf_counter()
        self.requests.put((self.worker, self.serial, self.seed, kind, arrays))
        serial, result, error = self.replies.get(timeout=120)
        self.wait_seconds += time.perf_counter() - start
        if serial != self.serial: raise RuntimeError('stale inference response')
        if error: raise RuntimeError(error)
        return result
    def __call__(self, board, state, x, cells, mask):
        # Existing single-state collate and argmax/tie handling remain in the worker.
        if len(board) != 1: raise ValueError('client expects one search node')
        n = int(mask[0].sum())
        obs = tuple(a.detach().numpy().copy() for a in
                    (board[0], state[0], x[0,:n], cells[0,:n]))
        logits, value = self.call('policy', obs)
        return torch.from_numpy(logits[None]), torch.tensor([value], dtype=torch.float32)
    def leaf(self, state, channels):
        if state.done:
            d = state.score(state.turn) - state.score(1-state.turn)
            return float((d>0)-(d<0))
        return self.call('value', value_inputs(state, channels))

class BatchServer:
    def __init__(self, models, requests, replies, max_batch=32, wait_ms=1.):
        self.models, self.requests, self.replies = models, requests, replies
        self.max_batch, self.wait = max_batch, wait_ms/1000
        self.hist = Counter(); self.counts = Counter()
        self.times = defaultdict(float)
        self.real_candidates = self.padded_candidates = 0
    def step(self, timeout=.1):
        try: first = self.requests.get(timeout=timeout)
        except queue.Empty: return False
        pending=[first]; deadline=time.perf_counter()+self.wait
        while len(pending)<self.max_batch:
            try: pending.append(self.requests.get(timeout=max(0,deadline-time.perf_counter())))
            except queue.Empty: break
        groups=defaultdict(list)
        for request in pending: groups[request[2:4]].append(request)
        for (seed,kind), batch in groups.items():
            start=time.perf_counter(); self.hist[f'{kind}:{len(batch)}']+=1
            self.counts[kind]+=len(batch)
            try:
                model=self.models[seed]; obs=[r[4] for r in batch]
                device=next(model.parameters()).device
                events=[torch.cuda.Event(enable_timing=True) for _ in range(3)]
                events[0].record()
                if kind=='policy':
                    args=collate(obs,device)
                    self.real_candidates+=sum(len(o[2]) for o in obs)
                    self.padded_candidates+=len(obs)*max(len(o[2]) for o in obs)
                else:
                    args=[torch.as_tensor(np.stack([o[i] for o in obs]),device=device) for i in (0,1)]
                events[1].record()
                with torch.inference_mode():
                    if kind=='policy': logits,values=model(*args)
                    else:
                        b,st=args
                        values=model.value(torch.cat((model.global_net(model.conv(b.float())),st),-1)).squeeze(-1)
                events[2].record()
                values=values.cpu().numpy()
                if kind=='policy': logits=logits.cpu().numpy()
                self.times['transfer_pack_cuda_ms']+=events[0].elapsed_time(events[1])
                self.times['forward_cuda_ms']+=events[1].elapsed_time(events[2])
                for i,(worker,serial,_,_,o) in enumerate(batch):
                    result=(logits[i,:len(o[2])].copy(),float(values[i])) if kind=='policy' else float(values[i])
                    self.replies[worker].put((serial,result,None))
            except Exception as exc:
                for worker,serial,*_ in batch: self.replies[worker].put((serial,None,repr(exc)))
                raise
            self.times['service_wall_seconds']+=time.perf_counter()-start
        return True
    def summary(self):
        return dict(batch_histogram=dict(self.hist),requests=dict(self.counts),times=dict(self.times),
                    real_candidates=self.real_candidates,padded_candidates=self.padded_candidates,
                    peak_cuda_bytes=torch.cuda.max_memory_allocated())
