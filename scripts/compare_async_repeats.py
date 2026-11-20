import gzip,json
from pathlib import Path
import frontier_generation as f
R=Path('artifacts/gpu-async512/final');bank=json.loads((R/'bank.json').read_text());pairs={}
for left,right in [('cpu-1','cpu-2'),('async-1','async-2')]:
 equal=0;score_equal=0;actions=0;agree=0;examples=[]
 for task in bank['games']:
  name=f"{task['seed']}-{task['id']}.json.gz";a=json.load(gzip.open(R/left/name,'rt'));b=json.load(gzip.open(R/right/name,'rt'));assert a['task']==b['task']==task
  equal+=a['moves']==b['moves'];score_equal+=a['scores']==b['scores']
  for i,(x,y) in enumerate(zip(a['moves'],b['moves'])):
   if i>=len(task['prefix']['moves']):actions+=1;agree+=x==y
   if x!=y:
    if len(examples)<32:examples.append(dict(game=name,ply=i,left=x,right=y,scores_left=a['scores'],scores_right=b['scores']))
    break
 pairs[f'{left}/{right}']=dict(games=len(bank['games']),equal_trajectories=equal,equal_scores=score_equal,same_state_actions=actions,action_agreement=agree/actions,examples=examples)
f.ref.write(R/'repeat-agreement.json',pairs)
