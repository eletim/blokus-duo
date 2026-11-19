"""Finite, resumable fixed-work CPU8 / central-GPU generation benchmark."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse, copy, gzip, hashlib, json, multiprocessing as mp, random, time
from pathlib import Path
import numpy as np
import torch
import placement_experiment as e
from blokus.engine import State, Move, play
from blokus.ppo_policy import PPOAgent
from blokus.ppo_search_deeper import DeepSearch
from blokus.ppo_runtime import state_hash
from blokus.batched_inference import InferenceClient, BatchServer
R=Path('artifacts/gpu-generation')

def write(path,obj):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2)+'\n'); tmp.replace(path)
def prepare():
    R.mkdir(exist_ok=True)
    if (R/'bank.json').exists(): raise RuntimeError('bank already fixed')
    ex=set()
    for base in ('value-placement-ab','value-placement-scale','value-placement-6144'):
        for name in ('exclusions.json','historical-starts.json','train-labels.json','dev-labels.json'):
            p=Path('artifacts')/base/name
            if p.exists():
                d=json.loads(p.read_text());ex.update(d.get('states',[]));ex.update(d.get('visited',[]))
        for p in (Path('artifacts')/base).rglob('*bank.json'):
            ex.update(x['state_hash'] for x in json.loads(p.read_text()).get('prefixes',[]))
    idx=np.load(e.INDEX,mmap_mode='r'); bank=[]
    for seed in e.SEEDS:
        for gid in range(8):
            rng_seed=981000000+seed*10000+gid
            rng=random.Random(rng_seed);pr=e.prefix(rng,idx,ex,n=8)
            ex.add(pr['state_hash'])
            bank.append(dict(seed=seed,id=gid,rng_seed=rng_seed,prefix=pr,rng_state=rng.getstate()))
    write(R/'bank.json',dict(games=bank,models={str(s):e.sha(f'artifacts/ppo-gen3/seed-{s}.pt') for s in e.SEEDS}))
    write(R/'excluded.json',dict(states=sorted(ex-set(g['prefix']['state_hash'] for g in bank))))

def rebuild_exclusions():
    manifest=json.loads((R/'exclusion-manifest.json').read_text());ex=set()
    for name,expected in manifest['sources'].items():
        assert e.sha(name)==expected,name
        d=json.loads(Path(name).read_text());ex.update(d.get('states',[]));ex.update(d.get('visited',[]))
        ex.update(p['state_hash'] for p in d.get('prefixes',[]))
    write(R/'excluded.json',dict(states=sorted(ex)))
    assert e.sha(R/'excluded.json')==manifest['union_sha256']

def tuples(x): return tuple(tuples(y) for y in x) if isinstance(x,list) else x

def worker(wid,tasks,mode,request,replies,ready,go,finished,directory):
    try:
        os.sched_setaffinity(0,{wid})
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        agents={}; clients={};pack_seconds=0.
        for seed in e.SEEDS:
            p=PPOAgent(seed,'cpu',backend='native')
            if mode=='gpu':
                client=InferenceClient(wid,seed,request,replies[wid]);clients[seed]=client;p.model=client
                p.value_backend=lambda z,c=client,ch=p.channels:c.leaf(z,ch)
                a=DeepSearch(p,60000)
            else:a=DeepSearch(p,60000)
            original=p._observation
            def observed(z,original=original):
                nonlocal pack_seconds
                start=time.perf_counter();out=original(z);pack_seconds+=time.perf_counter()-start;return out
            p._observation=observed;agents[seed]=a
        ready.put(wid);go.wait()
        start=time.perf_counter();cpu=time.process_time();games=0
        for g in tasks:
            path=Path(directory)/f"{g['seed']}-{g['id']}.json.gz"
            if path.exists():
                old=json.load(gzip.open(path,'rt'));assert old['task']==g;continue
            rng=random.Random();rng.setstate(tuples(g['rng_state']));s=e.state(g['prefix']['moves']);moves=list(g['prefix']['moves']);stats=[]
            while not s.done:
                a=agents[g['seed']];m=a.act(s,rng)
                assert a.stats['forced_pass'] or a.stats['depth']==6, a.stats
                stats.append(copy.deepcopy(a.stats));play(s,m,validate=True);moves.append(e.raw(m))
            out=dict(task=g,moves=moves,scores=[s.score(0),s.score(1)],stats=stats)
            tmp=path.with_suffix('.tmp')
            with gzip.open(tmp,'wt') as f:json.dump(out,f)
            tmp.replace(path);games+=1
        finished.put(dict(worker=wid,games=games,seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,
                          observation_seconds=pack_seconds,wait_seconds=sum(c.wait_seconds for c in clients.values())))
    except BaseException as exc:
        finished.put(dict(worker=wid,error=repr(exc)));raise

def benchmark(mode,repeat):
    if (R/f'{mode}-{repeat}.json').exists():raise RuntimeError('completed measurement exists; use a new repeat number')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    bank=json.loads((R/'bank.json').read_text());directory=R/f'{mode}-{repeat}';directory.mkdir(exist_ok=True)
    assert all(e.sha(f'artifacts/ppo-gen3/seed-{s}.pt')==h for s,h in bank['models'].items())
    ctx=mp.get_context('spawn');request=ctx.Queue();replies=[ctx.Queue() for _ in range(8)];ready=ctx.Queue();go=ctx.Event();finished=ctx.Queue()
    init=time.perf_counter();server=None
    if mode=='gpu':
        models={s:PPOAgent(s,'cuda',backend='native').model for s in e.SEEDS}
        server=BatchServer(models,request,replies);torch.cuda.reset_peak_memory_stats()
    workers=[ctx.Process(target=worker,args=(i,bank['games'][i::8],mode,request,replies,ready,go,finished,str(directory))) for i in range(8)]
    try:
        for p in workers:p.start()
        for _ in workers:ready.get(timeout=120)
        initialization=time.perf_counter()-init;start=time.perf_counter();cpu=time.process_time();go.set();results=[]
        while len(results)<8:
            if server:server.step(.02)
            else:time.sleep(.02)
            while not finished.empty():
                r=finished.get();results.append(r)
                if 'error' in r:raise RuntimeError(r)
            if any(p.exitcode not in (None,0) for p in workers):raise RuntimeError('worker failure')
        elapsed=time.perf_counter()-start
        for p in workers:p.join(timeout=10)
        out=dict(mode=mode,repeat=repeat,seconds=elapsed,initialization_seconds=initialization,
                 server_cpu_seconds=time.process_time()-cpu,workers=results,server=server.summary() if server else None,
                 bank_sha=e.sha(R/'bank.json'),new_games=sum(r['games'] for r in results))
        write(R/f'{mode}-{repeat}.json',out);print(json.dumps(out),flush=True)
    finally:
        for p in workers:
            if p.is_alive():p.terminate()
            p.join()
        for q in [request,ready,finished,*replies]:q.close()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare','rebuild-exclusions','cpu','gpu']);ap.add_argument('--repeat',type=int,default=1);a=ap.parse_args()
    if a.command=='prepare':prepare()
    elif a.command=='rebuild-exclusions':rebuild_exclusions()
    else:benchmark(a.command,a.repeat)
