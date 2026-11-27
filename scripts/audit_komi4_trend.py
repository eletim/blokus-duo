"""Audit saved artifacts without repeating generation or optimization."""
import hashlib
import json
from pathlib import Path
import torch
import time
from blokus.engine import State,Move,play,SIZES,legal_moves
from scripts.komi4_trend import read,sha,state_hash,write,baseline
R=Path('artifacts/komi4-trend')

def run():
    started=time.monotonic()
    events=[json.loads(line) for line in (R/'operations.jsonl').read_text().splitlines()]
    if not any(e['stage']=='computation-complete' for e in events):
        raise SystemExit('Computation incomplete: final audit deferred')
    torch.set_num_threads(1)
    assert read(R/'plan.json')['tasks_sha256']==sha(R/'tasks.json')
    metas={p.stem:read(p) for p in (R/'features').glob('*.json')}
    assert all('train-g'+str(g) in metas for g in range(6,16))
    keys={n:set(m['keys']) for n,m in metas.items()}
    overlaps={a+' / '+b:len(keys[a]&keys[b]) for i,a in enumerate(keys) for b in list(keys)[i+1:]}
    # Across training generations overlap would be reported; held-out overlap is forbidden.
    assert all(v==0 for k,v in overlaps.items() if 'validation' in k or 'test' in k),overlaps
    train=set().union(*(keys['train-g'+str(g)] for g in range(6,16)))
    traces=[t for g in range(6,16) for t in metas['train-g'+str(g)]['traces']]
    assert len(traces)==len(set(traces)) and len(traces)>=21000
    old=Path('artifacts/komi4-scale')
    old_train=set().union(*(set(read(p)['keys']) for p in (old/'features').glob('train-g*.json')))
    assert not train & old_train
    old_traces=set().union(*(set(read(p)['traces']) for p in (old/'features').glob('train-g*.json')))
    assert not set(traces)&old_traces
    train_with_prior=train|old_train
    counts={}; models={}
    for p in (R/'models').glob('*.pt'):
        ck=torch.load(p,map_location='cpu',weights_only=False)
        assert ck['komi']==4 and all(torch.isfinite(v).all() for v in ck['model'].values())
        if p.stem.endswith('last'):
            assert all(k in ck for k in ['optimizer','scheduler','numpy_rng','torch_rng','cuda_rng'])
        models[p.name]=dict(sha256=sha(p),epoch=ck['epoch'],generation=ck['generation'])
    for seed,t in read(R/'selection.json')['models'].items():
        assert sha(Path(t['checkpoint']))==t['sha256']
        candidates=[dict(generation=5,checkpoint=baseline(int(seed)),sha256=sha(baseline(int(seed))),validation_mae=read(R/'baseline-validation.json')[seed]['mae'])]+[read(R/f'training/g{g}-{seed}.json') for g in range(6,16)]
        assert t==min(candidates,key=lambda r:(r['validation_mae'],r['generation']))
    shared={}; manifest={}; alltraces=set(); learned=train_with_prior|keys['validation']|keys['test']; prefix_by_group={}; benchmarks={}
    for p in sorted((R/'games').glob('*.json.gz')):
        d=read(p);s=State();hit=0;t=d['task']
        assert d['moves'][:len(d['prefix'])]==d['prefix']
        if t['split']=='benchmark':
            ident=t['group']
            assert prefix_by_group.setdefault(ident,d['prefix'])==d['prefix']
            benchmarks[t['generation']]=benchmarks.get(t['generation'],0)+1
        for ply,m in enumerate(d['moves']):
            if t['split'] in ['final','standard','benchmark'] and state_hash(s) in learned:hit+=1
            if m is None:assert not legal_moves(s), 'illegal pass'
            play(s,Move(*m) if m is not None else None,validate=m is not None)
        assert s.done and d['scores']==[s.score(i) for i in [0,1]]
        assert d['adjusted_first_margin']==s.score(0)-s.score(1)-4
        assert d['remaining']==[sum(SIZES[i] for i in range(21) if s.remaining[p]>>i&1) for p in [0,1]]
        assert d['trace_sha256']==hashlib.sha256(json.dumps(d['moves']).encode()).hexdigest()
        if hit:shared[t['id']]=hit
        counts[t['split']]=counts.get(t['split'],0)+1
        manifest[p.name]=sha(p);alltraces.add(d['trace_sha256'])
    assert counts['final']==1536 and counts['standard']==12
    assert benchmarks=={g:1536 for g in range(5,16)}
    assert len(prefix_by_group)==128
    result=dict(audit_wall_seconds=time.monotonic()-started,unique_training_games=len(traces),unique_training_states=len(train),counts=counts,
        all_unique_trajectories=len(alltraces),state_intersections=overlaps,
        final_comparison_training_validation_overlap=shared,models=models,
        datasets={n:{k:v for k,v in m.items() if k not in ['keys','traces','tasks','game_ids']} for n,m in metas.items()},
        audit='All placements independently coordinate-checked, passes verified by empty legal-move list; all games terminal, scores/komi/remaining exact; finite models; resumable optimizer/RNG; validation-only selection.')
    write(R/'game-manifest.json',manifest);write(R/'audit.json',result)
    print(json.dumps({k:result[k] for k in ['unique_training_games','unique_training_states','counts','final_comparison_training_validation_overlap']}))
    assert not shared, 'Evaluation trajectories overlap learned or prediction-test states; report contamination before using as held-out evidence.'

if __name__=='__main__':run()
