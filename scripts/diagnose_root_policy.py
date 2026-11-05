"""Development-only mechanism diagnostic; never used for checkpoint selection."""
import sys,json,gzip,random
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from blokus.root_data import ROOT
from blokus.root_policy import RootPolicy,inputs
from blokus.root_experiment import agent
from blokus.root_train import metrics
from blokus.engine import State,legal_moves,Move
from blokus.research import clone
from blokus.ai import features,heuristic,seed_all
from blokus.reply_search import reset_game
from blokus.value_experiment import write

def main():
    seed_all(901);records=json.load(gzip.open(ROOT/'teacher-traces.json.gz','rt'));dev=[r for r in records if r['trainable'] and r['split']==1];out=[]
    for budget in [6,25]:
        for kind,policy in [('heuristic',None),('untrained',str(ROOT/'seed-901/untrained.pt')),('student',str(ROOT/'seed-901/step-1000.pt'))]:
            a=agent(dict(method=kind,policy=policy,budget_ms=budget))
            for r in dev:
                s=clone(State(**r['state']));reset_game(a);m=a.act(s,random.Random(91));d=dict(a.decisions[-1]);qs=legal_moves(s);x,st=features(s,qs);members=np.argsort(-heuristic(x),kind='stable')[:128];scores=heuristic(x)[members]
                if a.policy is not None:
                    xx,ss=inputs(s,[qs[int(i)] for i in members],x[members],st)
                    with torch.no_grad():scores+=a.policy(torch.as_tensor(xx,device='cuda'),torch.as_tensor(ss,device='cuda')).cpu().numpy()
                order=np.argsort(-scores,kind='stable');target=r['target'];n=d['completed_roots'];teacher=Move(*r['moves'][target])
                out.append(dict(game=r['game'],state_hash=r['state_hash'],budget_ms=budget,kind=kind,teacher_rank=int(np.where(order==target)[0][0])+1,teacher_evaluated=bool(target in order[:n]),selected_teacher=m==teacher,all_roots_complete=n==len(members),teacher_reverses_initial=bool(r['initial'][target]<max(v for v in r['initial'] if v is not None)-4e-6),remaining=sum(v.bit_count() for v in s.remaining),**d))
    raw=np.load(ROOT/'data.npz');data={k:torch.as_tensor(raw[k],device='cuda') for k in ['x','state','mask','legal_mask','target','split','prior']};fit=[]
    for seed in [901,902,903]:
        for step in [1000,3000]:
            model=RootPolicy().cuda();model.load_state_dict(torch.load(ROOT/f'seed-{seed}/step-{step}.pt',weights_only=False,map_location='cpu')['model']);fit.append(dict(seed=seed,step=step,train=metrics(model,data,0),dev=metrics(model,data,1)))
    summary=[]
    for b in [6,25]:
        for kind in ['heuristic','untrained','student']:
            ds=[d for d in out if d['budget_ms']==b and d['kind']==kind]
            summary.append(dict(budget_ms=b,kind=kind,roots=len(ds),games=len({d['game'] for d in ds}),teacher_coverage=float(np.mean([d['teacher_evaluated'] for d in ds])),teacher_agreement=float(np.mean([d['selected_teacher'] for d in ds])),all_roots_fraction=float(np.mean([d['all_roots_complete'] for d in ds])),median_policy_ms=float(np.median([d['policy_ms'] for d in ds]))))
    result=dict(development_only=True,checkpoint='seed901/1000 representative, not selected from holdout',rows=out,summary=summary,fit=fit,teacher_reversals=sum(r['initial'][r['target']]<max(v for v in r['initial'] if v is not None)-4e-6 for r in dev),dev_roots=len(dev))
    write(ROOT/'mechanism.json',result);print(summary,flush=True)
if __name__=='__main__':main()
