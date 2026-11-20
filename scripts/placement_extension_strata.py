"""Descriptive seed/seat intervals after fixed confirmation; never selection."""
import json,gzip
from pathlib import Path
import numpy as np
R=Path('artifacts/value-placement-49152/n49152');SEEDS=(1751401,1751402,1751403)
def summarize(rows):
 groups=sorted({g['prefix'] for g in rows});ix=np.random.default_rng(772).integers(len(groups),size=(20000,len(groups)));out=dict(games=len(rows),prefixes=len(groups),wins=sum(g['difference']>0 for g in rows),draws=sum(g['difference']==0 for g in rows),losses=sum(g['difference']<0 for g in rows))
 for key in ('point','difference'):
  means=np.array([np.mean([g[key] for g in rows if g['prefix']==i]) for i in groups]);out[key]=float(means.mean());out[key+'_ci95']=np.quantile(means[ix].mean(1),[.025,.975]).tolist()
 return out
def main():
 out={}
 for pair in ('AQ','AR','AF','AC'):
  rows=[dict(seed=s,**g) for s in SEEDS for g in json.load(gzip.open(R/f'confirmation-{pair}-{s}.json.gz','rt'))['games']]
  out[pair]=dict(by_seed={str(s):summarize([g for g in rows if g['seed']==s]) for s in SEEDS},by_seat={str(s):summarize([g for g in rows if g['seat']==s]) for s in (0,1)},by_seed_seat={f'{s}/{seat}':summarize([g for g in rows if g['seed']==s and g['seat']==seat]) for s in SEEDS for seat in (0,1)})
 (R/'confirmation-strata.json').write_text(json.dumps(dict(results=out,note='Descriptive ordinary95% prefix-cluster intervals for fixed seed/seat strata, not simultaneous subgroup claims, not used for model/seed selection or the preregistered aggregate gate.'),indent=2))
if __name__=='__main__':main()
