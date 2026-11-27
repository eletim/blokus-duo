"""Paired descriptive intervals on the pre-fixed repeated benchmark (not a test)."""
import json
from pathlib import Path
from scripts.summarize_komi4_scale import read, row, cluster_interval

R = Path('artifacts/komi4-trend')

def paired(baseline, current):
    def key(r):
        return (r['group'], r['seed'], r['seat'], r['opponent'])
    before = {key(r): r for r in baseline}
    after = {key(r): r for r in current}
    if len(before) != len(baseline) or len(after) != len(current) or before.keys() != after.keys():
        raise ValueError('Paired conditions must be unique and identical')
    differences = []
    for k, a in after.items():
        b = before[k]
        win = lambda r: float(r['margin'] > 0)
        points = lambda r: win(r) + .5 * (r['margin'] == 0)
        differences.append(dict(group=a['group'], win=win(a)-win(b),
                                points=points(a)-points(b), margin=a['margin']-b['margin']))
    if not differences:
        raise ValueError('Empty comparison')
    return dict(games=len(differences), groups=len({r['group'] for r in differences}),
                change={k:sum(r[k] for r in differences)/len(differences) for k in ['win','points','margin']},
                ci95={k:cluster_interval(differences,k) for k in ['win','points','margin']})

def run():
    def rows(g):
        paths=sorted((R/'games').glob(f'benchmark-g{g}-*.json.gz'))
        if len(paths)!=1536:
            return None
        return [row(read(p)) for p in paths]
    baseline=rows(5)
    if baseline is None:
        raise SystemExit('Baseline incomplete')
    results=[]
    for g in range(5,16):
        current=rows(g)
        if current is None:
            continue
        comparisons={}
        for opponent in ['gen4','prior']:
            b=[r for r in baseline if r['opponent']==opponent]
            c=[r for r in current if r['opponent']==opponent]
            comparisons[opponent]=dict(overall=paired(b,c),
                seats={str(s):paired([r for r in b if r['seat']==s],[r for r in c if r['seat']==s]) for s in [0,1]},
                seeds={str(s):paired([r for r in b if r['seed']==s],[r for r in c if r['seed']==s]) for s in [1751401,1751402,1751403]})
        results.append(dict(generation=g,versus_generation=5,opponents=comparisons))
    out=dict(interpretation='Current minus baseline; paired opening-group bootstrap, 4000 draws. Repeated descriptive benchmark, no multiple-comparison correction; not model selection or an unknown test.',generations=results)
    tmp=R/'paired-intervals.json.tmp'
    tmp.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    tmp.replace(R/'paired-intervals.json')
    print('Completed paired generations:',[r['generation'] for r in results])

if __name__=='__main__':
    run()
