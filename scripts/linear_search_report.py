"""Derive diagnostic cost/choice statistics without rerunning timed matches."""
import gzip,json
import numpy as np
from scripts import linear_search_run as ev
R=ev.R
with gzip.open(R/'pilot.json.gz','rt') as f:p=json.load(f)
rows=[r for r in p['states'] if r['results']['6x3']['move'] is not None]
metrics=[]
for r in rows:
 a=r['results']['6x3'];b=r['results']['24x3'];c=r['results']['6x8'];d=r['results']['24x8'];v=a['stats']['values'];wide=b['stats']['values'];reply=c['stats']['values']
 assert max(wide)>=max(v)-1e-12 and all(y<=x+1e-12 for x,y in zip(v,reply))
 metrics.append(dict(game=r['game'],ply=r['ply'],root_change=b['move']!=a['move'],reply_change=c['move']!=a['move'],both_change=d['move']!=a['move'],outside=d['move'] not in a['stats']['roots'],root_value_gain=max(wide)-max(v),reply_value_drop=max(v)-max(reply),any_reply_changes=any(abs(x-y)>1e-12 for x,y in zip(v,reply)),strict_root_gain=max(wide)>max(v)+1e-12,changed_without_root_gain=b['move']!=a['move'] and max(wide)<=max(v)+1e-12,narrow_all_tied=max(v)-min(v)<1e-12))
bygame={}
for x in metrics:bygame.setdefault(x['game'],[]).append(x)
rates={k:float(np.mean([x[k] for x in metrics])) for k in ['root_change','reply_change','both_change','outside','root_value_gain','reply_value_drop','any_reply_changes','strict_root_gain','changed_without_root_gain','narrow_all_tied']}
phase={str(ply):{k:float(np.mean([x[k] for x in metrics if x['ply']==ply])) for k in rates} for ply in (8,18,28)}
timing={}
for ms in (25,100):
 for label in ('A','B','C'):
  for warm in (False,True):
   stats=[x['stats'] for x in p['timing'] if x['budget']==ms and x['label']==label and x['warm']==warm and not x['stats'].get('forced_pass')]
   timing[f'{ms}/{label}/'+('warm' if warm else 'instance-cold')]=ev.summarize_stats(stats)
micro={str(ply):{k:float(np.mean([x[k] for x in p['micro'] if x['ply']==ply])) for k in p['micro'][0] if k!='ply'} for ply in (8,18,28)}
ev.write('diagnostic-derived.json',dict(independent_games=len(bygame),active_states=len(metrics),rates=rates,by_phase=phase,costs=timing,micro=micro,cold_definition='new search instance; native module/model already warm; process cold separately measured',micro_note='Python legal/feature alternatives and native fused legal+feature are overlapping measurements; value includes two scans and region features',wall=p['wall']))
# Correct the descriptive outside-root denominator: pass has no candidate set.
summary=json.loads((R/'pilot-summary.json').read_text());summary['wide_choice_outside_narrow']=rates['outside'];summary['outside_denominator_active_states']=len(metrics);summary['forced_pass_states']=len(p['states'])-len(metrics);summary['metric_note']='outside-root rate excludes forced passes; settings/budget unchanged';ev.write('pilot-summary.json',summary)
# Enrich timing summaries from saved per-move records; no new timed matches.
for phase_name in ('validation','confirmation'):
 if not (R/f'{phase_name}-summary.json').exists():continue
 groups={}
 for path in R.glob(f'{phase_name}-*.json*'):
  obj=ev.read_match(path)
  if not isinstance(obj.get('games'),list):continue
  key=obj['left']+'/'+obj['right'];groups.setdefault(key,{'left':[],'right':[]})
  for side in ('left','right'):groups[key][side]+=obj[side+'_search']
 ev.write(f'{phase_name}-costs.json',{k:{side:ev.summarize_stats(st) for side,st in sides.items() if st} for k,sides in groups.items()})
