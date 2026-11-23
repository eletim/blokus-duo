"""Descriptive pretraining root-width diagnosis; no outcome or minimax claims."""
import json
import numpy as np
from candidate_data import R,save

def main():
 groups={str(i):[] for i in range(3)};games=[]
 for path in sorted((R/'diagnosis').glob('*.json')):
  d=json.loads(path.read_text());gs=[]
  for row in d['rows']:
   w,n,s=[row['results'][x] for x in ['wide','narrow','shallow']]
   common=[abs(n['root_scores'][i]-w['root_scores'][w['root_moves'].index(m)]) for i,m in enumerate(n['root_moves'])]
   # Root-only expansion retains identical internal trees. Common-root values must agree.
   assert max(common,default=0)==0
   omitted=w['move'] not in row['top4'];stable=w['move']==s['move'];best_narrow=max(w['root_scores'][w['root_moves'].index(m)] for m in row['top4'])
   r=dict(omitted=omitted,stable=stable,stable_omitted=omitted and stable,changed=w['move']!=n['move'],value_gain=max(w['root_scores'])-best_narrow,common_error=max(common,default=0))
   for name,z in [('wide',w),('narrow',n),('shallow',s)]:
    for k in ['nodes','nn_policy_calls','nn_value_calls','elapsed_ms']:r[name+'_'+k]=z[k]
   groups[str(row['stage'])].append(r);gs.append(r)
  games.append(dict(seed=d['seed'],id=d['id'],roots=len(gs),omitted=sum(r['omitted'] for r in gs),stable_omitted=sum(r['stable_omitted'] for r in gs)))
 summary={}
 for stage,rs in groups.items():
  summary[stage]=dict(n=len(rs),means={k:float(np.mean([r[k] for r in rs])) for k in rs[0]},value_gain_quantiles=np.quantile([r['value_gain'] for r in rs],[0,.5,.95,1]).tolist())
 save(R/'diagnostic-details.json',dict(stages=summary,game_clusters=games,note='Six potential paired roots per game are correlated. Descriptive proportions only; game-cluster structure retained. Wider completed tree predictions are not true WDL/minimax. Root-only widening holds replies/value/depth fixed; unchanged common roots exactly match.'))
if __name__=='__main__':main()
