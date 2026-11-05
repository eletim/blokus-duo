"""Summarize saved diagnostics and paired results without reselecting anything."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from collections import defaultdict
import numpy as np
from blokus.value_experiment import write
ROOT=Path('artifacts/terminal-rollouts')

def ci(x,unit):
    a=np.asarray(x,float);idx=np.random.default_rng(906).integers(len(a),size=(4000,len(a)))
    return dict(mean=float(a.mean()),ci95=np.quantile(a[idx].mean(1),[.025,.975]).tolist(),unit=unit,count=len(a))

def main():
    data=json.loads((ROOT/'diagnostic.json').read_text());exact=json.loads((ROOT/'exact-diagnostic.json').read_text());out={}
    for div in [False,True]:
        for kind in ['mid','end']:
            ds=[d for d in data if d['kind']==kind];means={}
            for method in ['uniform','halving']:
                a=[]
                for d in ds:
                    best=max(r['mean'] for r in d['references']);values={tuple(r['move']) if r['move'] else None:r['mean'] for r in d['references']}
                    a.append(float(np.mean([best-values[tuple(z['move']) if z['move'] else None] for z in d['experiments'] if z['method']==method and z['diverse']==div])))
                means[method]=a;out[f'{kind}/diverse={div}/{method}']=ci(a,'source game (3 sampling seeds averaged within game)')
            out[f'{kind}/diverse={div}/halving_minus_uniform_loss']=ci(np.asarray(means['halving'])-means['uniform'],'source game')
    for div in [False,True]:
        for method in ['uniform','halving']:
            out[f'exact/diverse={div}/{method}']=ci([np.mean([x['regret'] for x in r['experiments'] if x['method']==method and x['diverse']==div]) for r in exact],'solved source game (capped cases excluded)')
    write(ROOT/'diagnostic-uncertainty.json',out)
    evaluations=[]
    for r in json.loads((ROOT/'holdout-results.json').read_text()):
        groups=defaultdict(list)
        for g in r['games']:groups[g['prefix_id']].append(g)
        rates=[np.mean([g['difference']>0 for g in gs]) for gs in groups.values()]
        phases={}
        for who in [0,1]:
            for phase in ['early','mid','late']:
                z=[x for g in r['games'] for x in g['decisions'] if x['who']==who and ('early' if x['remaining_pieces']>24 else 'mid' if x['remaining_pieces']>8 else 'late')==phase]
                if not z:continue
                phases[f'{who}/{phase}']=dict(calls=len(z),mean_ms=float(np.mean([x['ms'] for x in z])),mean_completed=float(np.mean([x.get('completed',0) for x in z])),mean_committed=float(np.mean([x.get('committed',0) for x in z])),fallbacks=sum(x.get('fallback',False) for x in z))
        evaluations.append(dict(name=r['name'],raw_win_rate=ci(rates,'prefix pair'),phases=phases))
    write(ROOT/'evaluation-uncertainty.json',evaluations)
    print(json.dumps(dict(diagnostic=out,evaluation=evaluations),indent=2))
if __name__=='__main__':main()
