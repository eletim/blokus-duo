import copy,json,time,random,resource
from pathlib import Path
import numpy as np
from blokus.engine import State,play
from blokus.linear_value import Agent,Search,enriched_inputs,value_features,geometry_tables
from blokus.linear_rl import inputs,prefix
R=Path('artifacts/linear-value')
def main():
 a=Agent.load(R/'new-3101/policy.json');states=[];rng=random.Random(24131)
 for game in range(8):
  s=prefix(2410000000+game);ply=4
  while not s.done:
   if ply in (4,16,28):states.append((ply,copy.deepcopy(s)))
   play(s,a.act(s,rng),validate=False);ply+=1
 rows=[]
 for _,s in states:
  enriched_inputs(s);value_features(s)
 for rep in range(5):
  for ply,s in states:
   order=['old_inputs','new_inputs','value'] if rep%2==0 else ['value','new_inputs','old_inputs']
   for name in order:
    begin=time.perf_counter();result=inputs(s,'native') if name=='old_inputs' else enriched_inputs(s) if name=='new_inputs' else value_features(s);ms=(time.perf_counter()-begin)*1000
    rows.append(dict(rep=rep,ply=ply,name=name,ms=ms,candidates=len(result[1]) if name!='value' else None))
 feature=[]
 for ply,s in states:
  x,qs,p=enriched_inputs(s)
  feature.append(dict(ply=ply,candidates=len(qs),extra_range=np.ptp(x[:,-3:],axis=0).tolist() if len(qs) else [0,0,0]))
 tables=geometry_tables();d=dict(rows=rows,feature_ranges=feature,parameter_bytes=a.weights.nbytes+a.value_weights.nbytes,geometry_array_bytes=sum(x.nbytes for x in tables[:3]),process_peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,warmup=True,alternating=True,torch_shared_import_not_neural_inference=True)
 (R/'profile.json').write_text(json.dumps(d,indent=2)+'\n');print('PROFILE DONE')
if __name__=='__main__':main()
