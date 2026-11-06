import hashlib,json
from pathlib import Path
import numpy as np
from scripts.linear_value_fit_eval import R,SEEDS,model_sha

def main():
 result={};train=[]
 for seed in SEEDS:
  d=json.loads((R/f'P-{seed}/training.json').read_text());logs=d['logs'];train.append(dict(seed=seed,seconds=d['seconds'],first16_mse=float(np.mean([x['mse'] for x in logs[:16]])),last16_mse=float(np.mean([x['mse'] for x in logs[-16:]])),policy_sha256=hashlib.sha256(np.array(json.loads((R/f'P-{seed}/policy.json').read_text())['weights']).tobytes()).hexdigest()))
 point={};score={}
 for label in ('P','original'):
  for s in SEEDS:
   d=json.loads((R/f'validation-{label}-gen3-{s}.json').read_text())
   for g in d['games']:
    point.setdefault(label,{}).setdefault(g['prefix_seed'],[]).append(g['point']);score.setdefault(label,{}).setdefault(g['prefix_seed'],[]).append(g['difference'])
 keys=sorted(point['P']);dp=np.array([np.mean(point['P'][k])-np.mean(point['original'][k]) for k in keys]);ds=np.array([np.mean(score['P'][k])-np.mean(score['original'][k]) for k in keys]);ix=np.random.default_rng(23400).integers(len(keys),size=(10000,len(keys)))
 result=dict(training=train,paired_gen3_difference=dict(point=float(dp.mean()),point_ci95=np.quantile(dp[ix].mean(1),[.025,.975]).tolist(),score=float(ds.mean()),score_ci95=np.quantile(ds[ix].mean(1),[.025,.975]).tolist(),note='diagnostic paired validation difference; not independent confirmation, no family correction'))
 selection=json.loads((R/'selection.json').read_text());summary=json.loads((R/'validation-summary.json').read_text());assert selection['selected'] is None;assert summary['P/gen3']['point']<summary['original/gen3']['point']-.03
 assert all(model_sha(label,int(seed))==h for label,hashes in selection['hashes'].items() for seed,h in hashes.items())
 result['selection_recomputed']=True;result['confirmation_run']=False
 (R/'final-summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
