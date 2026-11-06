"""Export fixed hybrid without original model/history dependencies at inference."""
import json,hashlib
from pathlib import Path
from blokus.linear_value import Agent
from scripts.linear_hybrid_run import locked,OLD,R,SEEDS

def main():
 locked();folder=R/'exports';folder.mkdir(exist_ok=True);rows=[]
 for seed in SEEDS:
  p=OLD/f'old-{seed}/policy.json';v=OLD/f'new-{seed}/policy.json';h=lambda x:hashlib.sha256(x.read_bytes()).hexdigest()
  a=Agent.load_pair(p,v,'python',h(p),h(v));target=folder/f'seed-{seed}.json'
  a.save(target,dict(**a.provenance,seed=seed,policy_iteration=1024,value_iteration=1024,experimental=True))
  rows.append(dict(path=str(target),sha256=h(target),source_policy_sha256=h(p),source_value_sha256=h(v)))
 (folder/'manifest.json').write_text(json.dumps(rows,indent=2)+'\n')
if __name__=='__main__':main()
