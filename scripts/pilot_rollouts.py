"""Finite development-only pilot; stop criterion precedes validation/holdout."""
import sys,json,random,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from blokus.rollout_search import candidates,terminal_rollout,allocate,RolloutAgent
from blokus.node_diagnostic import ordered,row
from blokus.engine import State,play,legal_moves
from blokus.research import clone,sha,source_hashes
from blokus.visited_data import state_hash,state_dict
from blokus.ai import seed_all
from blokus.aligned_data import BASE
from blokus.value_search import BudgetAgent
from blokus.reply_search import reset_game
from blokus.value_experiment import write
from tests.test_deep_search import oracle

ROOT=Path('artifacts/terminal-rollouts')

def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'pilot.json').exists():raise ValueError('existing pilot')
    seed_all(901);begin=time.perf_counter();states=[]
    write(ROOT/'config.json',dict(games=12,game_seed=720000,sample_seeds=[731000,732000,733000],reference_seed=750000,reference_per_candidate=64,candidates=8,fixed_samples=32,methods=['uniform','halving'],budgets=[25,100],criterion='100ms mid initial coverage>=.8 and median completed>=16 required',checkpoint=BASE,checkpoint_sha256=sha(BASE),training_updates=0,source_hashes=source_hashes()))
    for gid in range(12):
        s=State();history=[];moves=[];rng=random.Random(720000+gid)
        while not s.done:
            history.append(clone(s));m=rng.choice(ordered(s)[:8]);moves.append(row(m));play(s,m,validate=False)
        late=next(z for z in reversed(history) if 2<=len(legal_moves(z))<=8)
        for kind,z in [('mid',history[min(18+(gid%2)*6,len(history)//2)]),('end',late)]:
            states.append(dict(game=gid,kind=kind,state=state_dict(z),state_hash=state_hash(z),game_moves=moves))
    write(ROOT/'states.json',states)
    # Timing first. No oracle/label generation overlaps timed decisions.
    old={ms:BudgetAgent('value',BASE,.5,24,ms,'cuda') for ms in [25,100]};timing=json.loads((ROOT/'timing.json').read_text()) if (ROOT/'timing.json').exists() else []
    for d in ([] if timing else states):
        s=State(**d['state'])
        for ms in [25,100]:
            for diverse in [False,True]:
                for method in ['uniform','halving']:
                    a=RolloutAgent(ms,method,diverse,seed=740000+d['game']);a.reset_game(740000+d['game']);m=a.act(s,random.Random(0))
                    timing.append(dict(game=d['game'],kind=d['kind'],budget_ms=ms,diverse=diverse,method=method,move=row(m),**a.decisions[-1]))
            a=old[ms];reset_game(a);n=a.roots;m=a.act(s,random.Random(0));timing.append(dict(game=d['game'],kind=d['kind'],budget_ms=ms,method='old',move=row(m),ms=a.times[-1]*1000,completed_roots=a.roots-n))
        print('TIMING',d['game'],d['kind'],flush=True)
    write(ROOT/'timing.json',timing)
    # Reference means use independent seeds; they are not minimax ground truth.
    diagnostics=json.loads((ROOT/'diagnostic.json').read_text()) if (ROOT/'diagnostic.json').exists() else []
    resumed_states=len(diagnostics)
    for d in states:
        if any(x['game']==d['game'] and x['kind']==d['kind'] for x in diagnostics):continue
        s=State(**d['state']);sets={str(div):candidates(s,8,div) for div in [False,True]};union=list(dict.fromkeys(sets['False']+sets['True']));refs=[]
        for m in union:
            labels=[terminal_rollout(s,m,750000+d['game']*10000+i,trace=i==0) for i in range(64)]
            refs.append(dict(move=row(m),mean=float(np.mean([z['value'] for z in labels])),labels=labels))
        exact=None
        if d['kind']=='end':
            try:
                v,best,n=oracle(s);exact=dict(value=v,best_moves=[row(m) for m in sorted(best,key=lambda m:str(m))],nodes=n,complete=True)
            except RuntimeError as e:
                if str(e)!='oracle bound':raise
                exact=dict(complete=False,reason='independent oracle exceeded20000 nodes; no exact claim')
        experiments=[]
        for div,ms in sets.items():
            ref=[next(r['mean'] for r in refs if r['move']==row(m)) for m in ms]
            for method in ['uniform','halving']:
                for seed in [731000,732000,733000]:
                    z=allocate(s,ms,seed+d['game']*100,32,method,trace=seed==731000)
                    z.update(diverse=div=='True',method=method,seed=seed+d['game']*100,mean_regret=max(ref)-ref[z['candidate']],candidate_reference=ref)
                    experiments.append(z)
        diagnostics.append(dict(game=d['game'],kind=d['kind'],references=refs,exact=exact,experiments=experiments))
        write(ROOT/'diagnostic.json',diagnostics);print('DIAG',d['game'],d['kind'],flush=True)
    eligible=[];summary=[]
    for div in [False,True]:
        for method in ['uniform','halving']:
            for ms in [25,100]:
                ds=[t for t in timing if t['kind']=='mid' and t['method']==method and t['budget_ms']==ms and t['diverse']==div]
                coverage=float(np.mean([not t['fallback'] for t in ds]));med=float(np.median([t['completed'] for t in ds]));summary.append(dict(diverse=div,method=method,budget_ms=ms,initial_coverage=coverage,median_completed=med,mean_ms=float(np.mean([t['ms'] for t in ds])),overruns=sum(t['ms']>ms for t in ds)))
                if ms==100 and coverage>=.8 and med>=16:eligible.append(dict(diverse=div,method=method))
    write(ROOT/'pilot.json',dict(seconds=time.perf_counter()-begin,resumed_states=resumed_states,eligible=eligible,timing_summary=summary,stop=not eligible,training_updates=0));print('PILOT',summary,'ELIGIBLE',eligible,flush=True)
if __name__=='__main__':main()
