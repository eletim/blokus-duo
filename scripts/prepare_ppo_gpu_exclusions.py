"""Read only prior prefix files from git objects; do not duplicate histories."""
import json,subprocess,hashlib
from pathlib import Path
from blokus.tournament import prefix_state
from blokus.ppo_runtime import state_hash
root=Path('artifacts/ppo-four-gpu');ref='a011c78';paths=subprocess.check_output(['git','ls-tree','-r','--name-only',ref],text=True).splitlines();states=set();sources=[]
for path in paths:
 if not path.endswith('.json') or not any(w in path.lower() for w in ('prefix','bank')):continue
 raw=subprocess.check_output(['git','show',ref+':'+path])
 try:d=json.loads(raw)
 except ValueError:continue
 prefixes=d.get('prefixes') if isinstance(d,dict) else None
 if not isinstance(prefixes,list):continue
 count=0
 for p in prefixes:
  if not isinstance(p,dict) or 'moves' not in p:continue
  states.add(state_hash(prefix_state(p)));count+=1
 if count:sources.append(dict(path=path,sha256=hashlib.sha256(raw).hexdigest(),count=count))
(root/'excluded-starts.json').write_text(json.dumps(dict(ref=ref,sources=sources,count=len(states),states=sorted(states)),indent=2));print('excluded starts',len(states),'files',len(sources))
