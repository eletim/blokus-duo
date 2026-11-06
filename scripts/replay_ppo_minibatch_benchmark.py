"""Recover the full fixed benchmark trace without rerunning any update/benchmark."""
import gzip,json,torch
from blokus import ppo_runtime as r
from blokus.ppo_minibatch import R,write
from scripts.bench_ppo_runtime import records_hash
r.seed_all(0);m,p,o,ck=r.restore('artifacts/ppo-four-gpu/seed-1751401/iteration-4096.pt','cuda');rec,games,_=r.collect(m,p,4096,1751401,backend='native');saved=torch.load(R/'benchmark-rollout.pt',weights_only=False);assert records_hash(rec)==records_hash(saved)
with gzip.open(R/'benchmark-games.jsonl.gz','wt') as f:f.write(json.dumps(dict(iteration=4097,games=games))+'\n')
write(R/'benchmark-replay.json',dict(pass_=True,records_sha256=records_hash(rec),games=16,updates=0,note='Exact saved-input replay for full visited-state exclusion; no second timing benchmark.'))
