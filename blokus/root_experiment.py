"""Reconfirmation, training, matched ordering controls and frozen fresh evaluation."""
import argparse,json,random,time
from pathlib import Path
import numpy as np
from .root_policy import StudentAgent
from .aligned_data import BASE
from .value_search import BudgetAgent
from .reply_search import reset_game
from .engine import play,legal_moves,State
from .visited_data import state_hash
from .node_diagnostic import row
from .ai import seed_all
from .research import sha
from .tournament import make_prefixes,prefix_state,summarize
from .value_experiment import write
ROOT=Path('artifacts/root-policy-distillation')

def agent(spec):
    if spec['method']=='old':return BudgetAgent('value',BASE,.5,24,spec['budget_ms'],'cuda')
    return StudentAgent(BASE,spec['budget_ms'],spec.get('policy'))

def match(left,right,fixture):
    agents=[agent(left),agent(right)];games=[];begin=time.perf_counter()
    for i,p in enumerate(fixture['prefixes']):
        for seat in [0,1]:
            for a in agents:reset_game(a)
            s=prefix_state(p);rngs=[random.Random(950000+2*i+j) for j in (0,1)];moves=[];hashes=[state_hash(s)];decisions=[]
            while not s.done:
                who=0 if s.turn==seat else 1;a=agents[who];before=a.roots;m=a.act(s,rngs[s.turn])
                d=dict(a.decisions[-1]) if isinstance(a,StudentAgent) else dict(ms=a.times[-1]*1000,completed_roots=a.roots-before)
                d['who']=who;decisions.append(d)
                if m is None:assert not legal_moves(s)
                play(s,m,validate=m is not None);moves.append(row(m));hashes.append(state_hash(s));assert len(moves)<=86
            games.append(dict(prefix_id=p['id'],seat=seat,cold=True,scores=[s.score(0),s.score(1)],difference=s.score(seat)-s.score(1-seat),moves=moves,state_hashes=hashes,decisions=decisions))
    return dict(left=dict(spec=left,stats=agents[0].stats(),policy_sha256=sha(left['policy']) if left.get('policy') else None),right=dict(spec=right,stats=agents[1].stats(),policy_sha256=sha(right['policy']) if right.get('policy') else None),games=games,prefix_sha256=fixture['sha256'],value_sha256=sha(BASE),summary=summarize(games),seconds=time.perf_counter()-begin)

def validation():
    fixture=make_prefixes(921000,16);write(ROOT/'validation-prefixes.json',fixture);results=[]
    for budget in [6,25]:
        old=dict(method='old',budget_ms=budget)
        for name,policy in [('heuristic',None),('untrained',str(ROOT/'seed-901/untrained.pt'))]:
            r=match(dict(method=name,budget_ms=budget,policy=policy),old,fixture);results.append(r);write(ROOT/'validation-results.json',results);print(name,budget,r['summary'],flush=True)
        for step in [1000,3000]:
            for seed in [901,902,903]:
                spec=dict(method='student',budget_ms=budget,policy=str(ROOT/f'seed-{seed}/step-{step}.pt'),step=step,seed=seed)
                r=match(spec,old,fixture);results.append(r);write(ROOT/'validation-results.json',results);print('student',seed,step,budget,r['summary'],flush=True)
    scores={}
    for b in [6,25]:
        for st in [1000,3000]:
            rows=[r for r in results if r['left']['spec'].get('step')==st and r['left']['spec']['budget_ms']==b]
            scores[f'{b}/{st}']=float(np.mean([r['summary']['mean_difference'] for r in rows]))
    chosen=max(scores,key=scores.get);b,st=map(int,chosen.split('/'))
    selection=dict(budget_ms=b,step=st,seeds=[901,902,903],criterion='mean score difference across all three update seeds on new validation; no seed selection',criteria=scores,prefix_sha256=fixture['sha256'],policies={str(seed):dict(path=str(ROOT/f'seed-{seed}/step-{st}.pt'),sha256=sha(ROOT/f'seed-{seed}/step-{st}.pt')) for seed in [901,902,903]})
    # Pooled paired-prefix CI, not six independent observations per prefix.
    chosen_rows=[r for r in results if r['left']['spec'].get('step')==st and r['left']['spec']['budget_ms']==b]
    selection['validation_pooled']=pooled(chosen_rows);selection['advance']=scores[chosen]>=-1 and selection['validation_pooled']['difference_ci95'][1]>=0
    write(ROOT/'selection.json',selection);print('SELECTION',selection,flush=True)

def pooled(matches):
    by={}
    for m in matches:
        for g in m['games']:by.setdefault(g['prefix_id'],[]).append(g['difference'])
    d=np.array([np.mean(x) for x in by.values()]);w=np.array([np.mean(np.array(x)>0) for x in by.values()]);p=np.array([np.mean((np.array(x)>0)+.5*(np.array(x)==0)) for x in by.values()])
    idx=np.random.default_rng(9817).integers(len(d),size=(4000,len(d)))
    return dict(prefixes=len(d),games=sum(len(x) for x in by.values()),mean_difference=float(d.mean()),difference_ci95=np.quantile(d[idx].mean(1),[.025,.975]).tolist(),win_rate=float(w.mean()),win_ci95=np.quantile(w[idx].mean(1),[.025,.975]).tolist(),point_rate=float(p.mean()),point_ci95=np.quantile(p[idx].mean(1),[.025,.975]).tolist(),unit='prefix: average paired seats and all shared-data update seeds before bootstrap')

def holdout():
    selection=json.loads((ROOT/'selection.json').read_text());assert selection['advance']
    fixture=make_prefixes(922000,96);write(ROOT/'holdout-prefixes.json',fixture);results=[];budget=selection['budget_ms']
    for seed in selection['seeds']:
        spec=dict(method='student',budget_ms=budget,seed=seed,step=selection['step'],policy=selection['policies'][str(seed)]['path']);assert sha(spec['policy'])==selection['policies'][str(seed)]['sha256']
        for right in [dict(method='old',budget_ms=budget),dict(method='untrained',budget_ms=budget,policy=str(ROOT/'seed-901/untrained.pt'))]:
            r=match(spec,right,fixture);results.append(r);write(ROOT/'holdout-results.json',results);print('HOLDOUT',seed,right['method'],r['summary'],flush=True)
    write(ROOT/'holdout-pooled.json',{name:pooled([r for r in results if r['right']['spec']['method']==name]) for name in ['old','untrained']})

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['teacher','collect','validation','holdout','play']);p.add_argument('--seed',type=int,default=901);a=p.parse_args();seed_all(901)
    if a.command=='teacher':
        from .refinement_experiment import match as teacher_match
        f=make_prefixes(920000,96);write(ROOT/'teacher-prefixes.json',f);spec=json.load(open('artifacts/minimax-refinement/selection.json'))['spec'];return write(ROOT/'teacher-results.json',teacher_match(spec,dict(method='old',budget_ms=100),f))
    if a.command=='collect':
        from .root_data import collect,prepare_labels
        collect();return prepare_labels()
    if a.command=='validation':return validation()
    if a.command=='holdout':return holdout()
    sel=json.loads((ROOT/'selection.json').read_text());bot=agent(dict(method='student',budget_ms=sel['budget_ms'],policy=sel['policies'][str(a.seed)]['path']));s=State();rng=random.Random(0)
    while not s.done:
        print(s.render())
        if s.turn==0:
            qs=legal_moves(s);m=None
            if qs:
                for i,q in enumerate(qs):print(i,q.move)
                try:m=qs[int(input('index> '))].move
                except (ValueError,IndexError):continue
        else:m=bot.act(s,rng)
        play(s,m,validate=m is not None)
    print('scores',s.score(0),s.score(1))
if __name__=='__main__':main()
