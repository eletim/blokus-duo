"""Postselection descriptive candidate turnover; no new selection criterion."""
import json
import torch
from candidate_data import R,SEEDS,save
from candidate_train import dataset
from blokus.gen4 import load
from blokus.ppo_policy import PPOAgent,collate

def main():
 torch.set_num_threads(1);result=[]
 for seed in SEEDS:
  original,value=load(seed,'native');candidate=PPOAgent(seed,'cpu',checkpoint=R/f'policy-{seed}.pt',backend='native');rows=dataset('dev',seed,original)
  st={str(i):dict(n=0,teacher_old=0,teacher_new=0,old_top4_removed=0,old_greedy_removed=0,new_entries=0,value_max_error=0) for i in range(3)}
  with torch.inference_mode():
   for r in rows:
    inputs=collate([r['obs']],'cpu');old,ov=original.model(*inputs);new,nv=candidate.model(*inputs);assert torch.equal(ov,nv)
    a=set(old[0].argsort(descending=True)[:4].tolist());b=set(new[0].argsort(descending=True)[:4].tolist());z=st[str(r['stage'])];z['n']+=1;z['teacher_old']+=r['teacher'] in a;z['teacher_new']+=r['teacher'] in b;z['old_top4_removed']+=len(a-b);z['new_entries']+=len(b-a);z['old_greedy_removed']+=int(old[0].argmax()) not in b
  result.append(dict(seed=seed,stages=st))
 save(R/'dev-turnover.json',dict(results=result,note='Postselection descriptive audit only; old top4 removed counts are not known-good moves. Fixed gen4 value tensors independently hash-verified; old network value output exactly preserved.'))
if __name__=='__main__':main()
