"""Summarize completed campaign only; never generate games or select models."""
import json
import time
from collections import defaultdict
import numpy as np
from blokus.trend_console import compute_benchmark
from scripts.summarize_komi4_scale import read, row, summarize
from scripts.trend_paired_intervals import R, run as paired_run

def run():
    started=time.monotonic()
    events=[json.loads(line) for line in (R/'operations.jsonl').read_text().splitlines()]
    complete=next((e for e in reversed(events) if e['stage']=='computation-complete'),None)
    if complete is None:
        raise SystemExit('Computation incomplete; no final results emitted')
    progress=read(R/'learning-progress.json')
    assert progress['generation']==15 and progress['additional_unique_train_games']>=21000
    benchmarks=[compute_benchmark(g) for g in range(5,16)]
    paired_run()
    games=[read(p) for p in sorted((R/'games').glob('*.json.gz')) if p.name.startswith(('final-','standard-'))]
    assert len(games)==1548
    rows=[row(g) for g in games]
    final={}
    for opponent in ['gen4','prior']:
        final[opponent]={}
        for split in ['final','standard']:
            subset=[r for r in rows if r['opponent']==opponent and r['split']==split]
            assert len(subset)==(768 if split=='final' else 6)
            final[opponent][split]=dict(overall=summarize(subset),
                seats={str(s):summarize([r for r in subset if r['seat']==s]) for s in [0,1]},
                seeds={str(s):summarize([r for r in subset if r['seed']==s]) for s in [1751401,1751402,1751403]})
    search=defaultdict(list)
    for game in games:
        for stat in game['stats']:search[stat['agent']].append(stat)
    metrics={}
    for agent,stats in search.items():
        names={k for s in stats for k,v in s.items() if isinstance(v,(int,float)) and k!='player'}
        metrics[agent]={k:dict(mean=float(np.mean([s[k] for s in stats if k in s])),p95=float(np.quantile(np.asarray([s[k] for s in stats if k in s],dtype=float),.95))) for k in sorted(names)}
    training=progress['training']
    result=dict(komi=4,additional_unique_train_games=progress['additional_unique_train_games'],
        cumulative_unique_train_games=progress['cumulative_unique_train_games'],new_training_positions=progress['new_training_positions'],
        epochs=sum(t['epochs'] for t in training),optimizer_updates=sum(t['updates'] for t in training),
        training_validation_wall_seconds=sum(t['seconds'] for t in training),
        campaign_elapsed_seconds=complete['at']-read(R/'plan.json')['created_at'],
        stage_timings={p.stem:read(p) for p in sorted(R.glob('*-timing.json'))},
        benchmarks=benchmarks,paired_changes=read(R/'paired-intervals.json'),final=final,search=metrics,
        selection=read(R/'selection.json'),prediction_test=read(R/'prediction-test.json'),
        learning_curve=[{k:v for k,v in t.items() if k!='history'} for t in training],
        interpretation=['Win rate is wins/all; point rate assigns draws 0.5.',
            'Benchmark is repeatedly observed, not an unknown test; validation alone selects models.',
            'Opening-cluster percentile intervals are descriptive without multiple-testing correction.',
            'Standard-start games share one opening group; no independent-opening confidence interval.',
            'Point margins include bonuses and second-player +4 komi; unplaced squares exclude these.',
            'Gen4 intermediate value was trained at komi0; equal search caps do not ensure equal achieved depth.',
            'Training timing is epoch training plus validation wall time, not pure GPU utilization.'])
    result['summary_wall_seconds']=time.monotonic()-started
    tmp=R/'results.json.tmp'
    tmp.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');tmp.replace(R/'results.json')
    print('Saved final results; independent audit still required')

if __name__=='__main__':run()
