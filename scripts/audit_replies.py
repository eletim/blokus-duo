"""Independent replay/accounting and exact endgame checks, after timed runs."""
import sys,json,time
from pathlib import Path
from collections import defaultdict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from blokus.engine import State,Move,play,legal_moves
from blokus.tournament import read_prefixes,prefix_state,summarize
from blokus.visited_data import state_hash
from blokus.research import sha
from blokus.ai import seed_all
from blokus.value_experiment import write
from blokus.node_diagnostic import NodeSearch
from blokus.value_search import ValueEvaluator
from blokus.aligned_data import BASE
from tests.test_deep_search import oracle


def audit_match(r,fixture):
    assert r['prefix_sha256']==fixture['sha256'] and r['checkpoint_sha256']==sha(BASE)
    assert r['budget_ms']==25 and r['resets_per_agent']==len(r['games'])
    prefixes={p['id']:p for p in fixture['prefixes']};pairs=defaultdict(list);states=set();times=[[],[]];counts=[defaultdict(int),defaultdict(int)]
    assert len({state_hash(prefix_state(p)) for p in fixture['prefixes']})==len(prefixes)
    for g in r['games']:
        pairs[g['prefix_id']].append(g['seat']);s=prefix_state(prefixes[g['prefix_id']]);hashes=[state_hash(s)]
        assert len(g['moves'])==len(g['decisions'])
        for move,d in zip(g['moves'],g['decisions']):
            assert d['who']==(0 if s.turn==g['seat'] else 1)
            m=Move(*move) if move else None
            if m is None:assert not legal_moves(s)
            play(s,m,validate=m is not None);hashes.append(state_hash(s))
            times[d['who']].append(d['ms'])
            for k in ['completed_roots','nodes','fallbacks']:counts[d['who']][k]+=d[k]
        assert s.done and hashes==g['state_hashes'] and [s.score(0),s.score(1)]==g['scores'] and s.score(g['seat'])-s.score(1-g['seat'])==g['difference']
        states.update(hashes)
    assert set(pairs)==set(prefixes) and all(sorted(v)==[0,1] for v in pairs.values())
    assert summarize(r['games'])==r['summary']
    for i,side in enumerate(['left','right']):
        z=r[side];t=np.array(times[i]);assert z['calls']==len(t) and z['overruns']==sum(t>25)
        assert abs(z['milliseconds_per_move']-t.mean())<1e-9 and abs(z['total_seconds']-t.sum()/1000)<1e-9
        for k,v in counts[i].items():assert z[k]==v
    return states


def main():
    seed_all(803);begin=time.perf_counter();root=Path('artifacts/reply-diagnostics');diag=json.loads((root/'diagnostic.json').read_text());cfg=json.loads((root/'config.json').read_text())
    groups=defaultdict(list);diagnostic_hashes=set();replay_states=set();proofs=[];node_replays=[];ev=ValueEvaluator(BASE,'cuda',.5,24)
    for d in diag['states']:
        groups[d['game']].append(d['kind']);s=State();hist=set()
        for move in d['game_moves']:
            hist.add(state_hash(s));m=Move(*move) if move else None
            if m is None:assert not legal_moves(s)
            play(s,m,validate=m is not None)
        assert s.done;replay_states.update(hist);z=State(**d['state']);h=state_hash(z);assert h==d['state_hash'] and h in hist and h not in diagnostic_hashes;diagnostic_hashes.add(h)
        records=d['records'];best=max(min(r['reference']) for r in records)
        old=max(records,key=lambda r:min(r['shallow'][:3]));assert abs(d['metrics']['old_regret']-(best-min(old['reference'])))<1e-8
        metrics=dict(reply_omission_rate=float(np.mean([min(r['reference'][:3])-min(r['reference'])>1e-6 for r in records])),reply_omission_gap=float(np.mean([min(r['reference'][:3])-min(r['reference']) for r in records])),leaf_mae=float(np.mean([abs(a-b) for r in records for a,b in zip(r['reference'],r['shallow'])])),root6_loss=best-max(min(r['reference']) for r in records[:6]))
        for k,v in metrics.items():assert abs(v-d['metrics'][k])<1e-8
        if d['kind']=='end':
            val,opt,n=oracle(z);computed={Move(*r['move']) if r['move'] else None for r in records if min(r['reference'])==best}
            assert val==best and computed==opt;proofs.append(dict(game=d['game'],nodes=n,value=val))
        # One fixed-node configuration per state, rotating widths/depth. Full record equality.
        c=d['fixed_node'][d['game']%8];a=NodeSearch(ev,2048,6).roots(z,c['root_width'],c['reply_width'],c['depth']);b=NodeSearch(ev,2048,6).roots(z,c['root_width'],c['reply_width'],c['depth'])
        assert a==b and a['nodes']<=2048
        for k,v in a.items():assert v==c[k]
        node_replays.append(dict(game=d['game'],kind=d['kind'],nodes=a['nodes']))
    assert len(groups)==16 and all(sorted(v)==['end','mid'] for v in groups.values())
    valfixture=read_prefixes(root/'validation-prefixes.json');holdfixture=read_prefixes(root/'holdout-prefixes.json');validation=json.loads((root/'validation-results.json').read_text());hold=json.loads((root/'holdout-results.json').read_text());sel=json.loads((root/'selection.json').read_text())
    assert sel['reply_width']==max(validation,key=lambda r:r['summary']['mean_difference'])['width']==hold['width']
    vs=set()
    for r in validation:vs.update(audit_match(r,valfixture))
    hs=audit_match(hold,holdfixture);assert not vs&hs and not replay_states&(vs|hs)
    write(root/'audit.json',dict(status='PASS',seconds=time.perf_counter()-begin,independent_endgames=proofs,node_replays=node_replays,diagnostic_games=len(groups),diagnostic_states=len(diag['states']),replayed_diagnostic_states=len(replay_states),validation_games=sum(len(r['games']) for r in validation),holdout_games=len(hold['games']),state_overlap=0,selection='PASS',accounting='PASS',checkpoint_sha256=sha(BASE)))
    print('PASS',time.perf_counter()-begin,flush=True)
if __name__=='__main__':main()
