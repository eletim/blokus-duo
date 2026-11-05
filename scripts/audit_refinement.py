"""Independent legal replay, arithmetic, pairing, time and frozen-weight audit."""
import json,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from blokus.engine import State,Move,play,legal_moves
from blokus.visited_data import state_hash
from blokus.tournament import read_prefixes,prefix_state,summarize
from blokus.research import sha,clone
from blokus.aligned_data import BASE
from blokus.value_experiment import write
ROOT=Path('artifacts/minimax-refinement')
def replay(s,row):
    m=Move(*row) if row else None
    if m is None:assert not legal_moves(s)
    play(s,m,validate=m is not None)
def run():
    out=[];phase_hashes={};games_total=0
    selected=json.loads((ROOT/'selection.json').read_text())
    validation=json.loads((ROOT/'validation-results-0.json').read_text())[0]
    assert selected['spec']==validation['left']['spec'] and selected['validation_sha256']==validation['prefix_sha256']
    assert selected['gate_pass'] and selected['frozen_before_holdout']
    assert validation['summary']['mean_difference']>=-1 and validation['summary']['difference_ci95'][1]>=0
    assert validation['left']['stats']['overruns']/validation['left']['stats']['calls']<.1
    assert all(m['left']['spec']==selected['spec'] for m in json.loads((ROOT/'holdout-results.json').read_text()))
    for path in sorted(ROOT.glob('*results*.json')):
        fixture=read_prefixes(str(path).replace('results','prefixes'));phase_hashes[path.name]=set()
        prefix_by={p['id']:p for p in fixture['prefixes']}
        for match in json.loads(path.read_text()):
            assert match['checkpoint_sha256']==sha(BASE) and match['prefix_sha256']==fixture['sha256']
            pairs=defaultdict(list);times=[[],[]];roots=[0,0];nodes=[0,0]
            for g in match['games']:
                s=prefix_state(prefix_by[g['prefix_id']]);assert state_hash(s)==g['state_hashes'][0];assert g['cold'];pairs[g['prefix_id']].append(g['seat'])
                assert len(g['moves'])==len(g['decisions'])==len(g['state_hashes'])-1
                for i,(m,d) in enumerate(zip(g['moves'],g['decisions'])):
                    who=0 if s.turn==g['seat'] else 1;assert who==d['who'];times[who].append(d['ms']);roots[who]+=d['completed_roots']
                    phase_hashes[path.name].add(state_hash(s));replay(s,m);assert state_hash(s)==g['state_hashes'][i+1]
                    if 'refined' in d:
                        assert d['refined']<=d['attempted'] and d['refined']<=d['completed_roots']<=d['candidate_count']
                        assert d['baseline_retained']==(d['refined']==0)
                        assert d['fallback']==bool(d['candidate_count'] and not d['completed_roots'])
                        assert abs(sum(d['sections_ms'].values())-d['ms'])<.1
                        nodes[who]+=d['nodes']
                assert s.done and g['scores']==[s.score(0),s.score(1)] and g['difference']==s.score(g['seat'])-s.score(1-g['seat'])
            assert len(pairs)==len(prefix_by) and all(sorted(p)==[0,1] for p in pairs.values())
            summary=match['summary'];diffs=[g['difference'] for g in match['games']]
            assert summary['wins']==sum(d>0 for d in diffs) and summary['losses']==sum(d<0 for d in diffs) and summary['draws']==sum(d==0 for d in diffs)
            assert abs(summary['mean_difference']-sum(diffs)/len(diffs))<1e-12
            assert abs(summary['point_rate']-(summary['wins']+.5*summary['draws'])/len(diffs))<1e-12
            grouped=defaultdict(list)
            for g in match['games']:grouped[g['prefix_id']].append(g['difference'])
            pairmeans=[sum(ds)/2 for ds in grouped.values()];rng=np.random.default_rng(9817)
            indices=rng.integers(len(pairmeans),size=(4000,len(pairmeans)))
            independently_bootstrapped=[sum(pairmeans[int(i)] for i in ids)/len(pairmeans) for ids in indices]
            assert np.allclose(np.quantile(independently_bootstrapped,[.025,.975]),summary['difference_ci95'],atol=1e-12)
            assert summarize(match['games'])==summary
            for who,k in enumerate(['left','right']):
                st=match[k]['stats'];assert st['calls']==len(times[who]) and abs(sum(times[who])/1000-st['total_seconds'])<1e-8
                assert st['overruns']==sum(t>st['allowance_ms'] for t in times[who]) and st['completed_roots']==roots[who]
                if match[k]['spec']['method']!='old':assert st['nodes']==nodes[who]
            # Raw win-rate interval, cluster resampling paired seats, separate from point rate.
            clusters=defaultdict(list)
            for g in match['games']:clusters[g['prefix_id']].append(float(g['difference']>0))
            means=np.array([np.mean(x) for x in clusters.values()]);rng=np.random.default_rng(9817);boot=means[rng.integers(len(means),size=(4000,len(means)))].mean(1)
            out.append(dict(file=path.name,left=match['left']['spec'],right=match['right']['spec'],games=len(match['games']),win_rate=float(means.mean()),win_ci=np.quantile(boot,[.025,.975]).tolist(),summary=match['summary']))
            games_total+=len(match['games'])
    source_hashes=set();states=json.loads((ROOT/'states.json').read_text())
    for r in states:
        s=State();history=[]
        for m in r['game_moves']:history.append(state_hash(s));replay(s,m)
        assert s.done and history[r['ply']]==r['state_hash']==state_hash(clone(State(**r['state'])))
        source_hashes.add(r['state_hash'])
    from tests.test_deep_search import oracle
    oracle_checks=0
    for pilot in ROOT.glob('pilot-[0-9].json'):
        for d in json.loads(pilot.read_text()):
            if 'value' not in d.get('oracle',{}):continue
            source=next(r for r in states if r['game']==d['game'] and r['kind']==d['kind'])
            v,optimal,_=oracle(clone(State(**source['state'])))
            assert v==d['oracle']['value']
            saved={Move(*m) if m else None for m in d['oracle']['best']};assert saved==optimal
            oracle_checks+=1
    overlap={name:len(hs&source_hashes) for name,hs in phase_hashes.items()}
    # Initial positions and forced end states can coincide; report all cross-phase hashes.
    cross={a+' / '+b:len(phase_hashes[a]&phase_hashes[b]) for a in phase_hashes for b in phase_hashes if a<b}
    result=dict(pass_=True,games=games_total,diagnostic_states=len(states),independent_oracle_rechecks=oracle_checks,diagnostic_root_overlap=overlap,cross_phase_state_overlap=cross,matches=out,checkpoint_sha256=sha(BASE))
    write(ROOT/'audit.json',result);print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':run()
