"""Read-only evaluation of the frozen campaign; never generates or trains games."""
import collections
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np

R = Path('artifacts/komi4-scale')

def read(path):
    with (gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as f:
        return json.load(f)

def cluster_interval(rows, field, seed=260526):
    """Resample opening groups, keeping both seats and repeated conditions together."""
    groups = collections.defaultdict(list)
    for r in rows:
        groups[r['group']].append(r[field])
    if len(groups) < 2:
        return None
    sums = np.array([sum(v) for v in groups.values()])
    sizes = np.array([len(v) for v in groups.values()])
    ix = np.random.default_rng(seed).integers(0, len(groups), (4000, len(groups)))
    return np.quantile(sums[ix].sum(1) / sizes[ix].sum(1), [.025, .975]).tolist()

def summarize(rows):
    if not rows:
        return None
    ds = np.array([r['margin'] for r in rows])
    wins, draws, losses = int(sum(ds > 0)), int(sum(ds == 0)), int(sum(ds < 0))
    enriched = [dict(r, win=float(r['margin'] > 0), points=float(r['margin'] > 0) + .5 * (r['margin'] == 0)) for r in rows]
    return dict(games=len(rows), groups=len({r['group'] for r in rows}), unique_trajectories=len({r['trace'] for r in rows}),
                wins=wins, draws=draws, losses=losses, win_rate=wins/len(rows), point_rate=(wins+.5*draws)/len(rows),
                mean_margin=float(ds.mean()), median_margin=float(np.median(ds)),
                quantiles=dict(zip(['min','p10','p25','p75','p90','max'], np.quantile(ds,[0,.1,.25,.75,.9,1]).tolist())),
                mean_winning_margin=float(ds[ds>0].mean()) if wins else None,
                mean_unplaced_advantage=float(np.mean([r['unplaced_advantage'] for r in rows])),
                bonus_difference_games=sum(r['bonus_delta'] != 0 for r in rows),
                ci95={k:cluster_interval(enriched,k) for k in ['margin','win','points']})

def row(game):
    t=game['task']; sign=1 if t['seat']==0 else -1
    # All standard-start seeds share the same initial position: one opening group.
    return dict(id=t['id'], group='standard-initial' if t['split']=='standard' else t['group'],
                seed=t['seed'], seat=t['seat'], opponent=t['opponent'], split=t['split'],
                trace=game['trace_sha256'], margin=sign*game['adjusted_first_margin'],
                unplaced_advantage=sign*(game['remaining'][1]-game['remaining'][0]),
                bonus_delta=sign*(game['scores'][0]-game['scores'][1]-(game['remaining'][1]-game['remaining'][0])))

def run():
    events=[json.loads(s) for s in (R/'operations.jsonl').read_text().splitlines()]
    end=next((e for e in reversed(events) if e['stage']=='computation-complete'),None)
    if end is None:
        raise SystemExit('Campaign still running: final results are not available.')
    progress=read(R/'learning-progress.json'); selection=read(R/'selection.json')
    assert progress['unique_train_games'] >= 10000 and progress['generation']==5
    games=[read(p) for p in sorted((R/'games').glob('*.json.gz')) if p.name.startswith(('comparison-','standard-'))]
    assert len(games)==786
    rows=[row(g) for g in games]
    comparisons={}
    for opp in ['ridge','prior','gen4']:
        rr=[r for r in rows if r['opponent']==opp]
        comparisons[opp]={}
        for split in ['comparison','standard']:
            subset=[r for r in rr if r['split']==split]
            comparisons[opp][split]=dict(overall=summarize(subset),
                seats={str(s):summarize([r for r in subset if r['seat']==s]) for s in [0,1]},
                seeds={str(s):summarize([r for r in subset if r['seed']==s]) for s in [1751401,1751402,1751403]})
    search=collections.defaultdict(list)
    for g in games:
        for s in g['stats']:
            search[s['agent']].append(s)
    search_summary={}
    for agent, stats in search.items():
        numeric={k for s in stats for k,v in s.items() if isinstance(v,(float,int)) and k!='player'}
        search_summary[agent]=dict(moves=len(stats), metrics={k:dict(mean=float(np.mean([float(s[k]) for s in stats if k in s])),p95=float(np.quantile([float(s[k]) for s in stats if k in s],.95))) for k in sorted(numeric)})
    generation_matches={}
    for generation in range(1,6):
        gs=[read(p) for p in sorted((R/'games').glob(f'selection-g{generation}-*.json.gz'))]
        generation_matches[str(generation)]=summarize([row(g) for g in gs])
    training=progress['training']
    curve=[{k:v for k,v in t.items() if k!='history'} for t in training]
    result=dict(komi=4, unique_train_games=progress['unique_train_games'], unique_positions=progress['unique_positions'],
        generated_train_games=progress['generated_train_games'], excluded=progress['excluded_duplicates'],
        training_seconds=sum(t['seconds'] for t in training), optimizer_updates=sum(t['updates'] for t in training),
        epochs=sum(t['epochs'] for t in training), elapsed_seconds=end['at']-read(R/'plan.json')['created_at'],
        comparisons=comparisons, search=search_summary, learning_curve=curve, generation_matches=generation_matches,
        selected={s:{k:v for k,v in t.items() if k!='history'} for s,t in selection['models'].items()},
        prediction_test=read(R/'prediction-test.json'),
        interpretation=['Win rate counts draws as non-wins; point rate gives draws 0.5.',
            'Margins include bonuses and second-player komi +4. Unplaced advantage excludes both.',
            '95% percentile bootstrap resamples opening groups (4000 draws); descriptive, not multiplicity-adjusted.',
            'Standard-start results have one initial-position group: no independent-opening CI.',
            'Validation selects checkpoints. Final test and matches do not select models.',
            'Gen4 intermediate value was trained at komi0. Equal search caps do not imply equal achieved depth.'])
    tmp=R/'results.json.tmp';tmp.write_text(json.dumps(result,ensure_ascii=False,indent=2));tmp.replace(R/'results.json')
    print(json.dumps({k:result[k] for k in ['unique_train_games','training_seconds','optimizer_updates','elapsed_seconds']}))

if __name__=='__main__':
    run()
