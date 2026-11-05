"""Replay terminal labels, completed-round arithmetic and cold paired games."""
import json,sys,time
from pathlib import Path
from collections import defaultdict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from blokus.engine import State,Move,play,legal_moves
from blokus.visited_data import state_hash
from blokus.research import sha,clone
from blokus.aligned_data import BASE
from blokus.rollout_search import candidates,terminal_rollout,allocate
from blokus.node_diagnostic import row,child,NodeSearch
from blokus.value_search import ValueEvaluator
from blokus.tournament import read_prefixes,prefix_state,summarize
from blokus.value_experiment import write
from tests.test_deep_search import oracle
ROOT=Path('artifacts/terminal-rollouts')

def replay(s,moves):
    for r in moves:
        m=Move(*r) if r else None
        if m is None:assert not legal_moves(s)
        play(s,m,validate=m is not None)
    return s

def replay_snapshot(snapshot,moves):
    return replay(clone(State(**snapshot)),moves)

def audit_diagnostic():
    states=json.loads((ROOT/'states.json').read_text());data=json.loads((ROOT/'diagnostic.json').read_text());assert len(states)==len(data)==24
    by={(d['game'],d['kind']):d for d in states};source_states=set();exact_rows=[];trace_count=0;sample_count=0;fixed_count=0;fixed_timings=[]
    for d in data:
        meta=by[(d['game'],d['kind'])];s=State();hs=set()
        for r in meta['game_moves']:
            hs.add(state_hash(s));replay(s,[r])
        assert s.done and meta['state_hash'] in hs;source_states.update(hs);s=clone(State(**meta['state']));assert state_hash(s)==meta['state_hash'];p=s.turn
        for ref in d['references']:
            m=Move(*ref['move']) if ref['move'] else None
            assert ref['mean']==float(np.mean([z['value'] for z in ref['labels']]))
            for i,z in enumerate(ref['labels']):
                assert z['value']==z['scores'][p]-z['scores'][1-p]
                if z['moves'] is not None:
                    assert z['moves'][0]==ref['move']
                    t=replay_snapshot(meta['state'],z['moves']);assert t.done and [t.score(0),t.score(1)]==z['scores'] and z['plies']==len(z['moves']);trace_count+=1
            expected=terminal_rollout(s,m,750000+d['game']*10000+63)
            assert expected==ref['labels'][63];sample_count+=1
        for z in d['experiments']:
            ms=candidates(s,8,z['diverse']);assert z['completed']==z['committed']==len(z['labels']) and z['completed']<=32 and not z['fallback'] and not z['interrupted']
            values=[[] for _ in ms]
            for lab in z['labels']:
                assert lab['sample']==len(values[lab['candidate']])
                values[lab['candidate']].append(lab['value']);assert lab['seed']==z['seed']+lab['sample']*1000003
                assert lab['value']==lab['scores'][p]-lab['scores'][1-p]
                if lab['moves'] is not None:
                    assert lab['moves'][0]==row(ms[lab['candidate']])
                    t=replay_snapshot(meta['state'],lab['moves']);assert t.done and [t.score(0),t.score(1)]==lab['scores'] and lab['plies']==len(lab['moves']);trace_count+=1
            assert values==z['values'] and z['move']==row(ms[z['candidate']]) and abs(z['mean_regret']-(max(z['candidate_reference'])-z['candidate_reference'][z['candidate']]))<1e-9
            if z['seed']==731000+d['game']*100:
                start=time.perf_counter();rr=allocate(s,ms,z['seed'],32,z['method'],trace=True)
                fixed_timings.append(dict(game=d['game'],kind=d['kind'],method=z['method'],diverse=z['diverse'],ms=(time.perf_counter()-start)*1000,completed=rr['completed'],trace=True))
                for k,v in rr.items():assert v==z[k]
                fixed_count+=1
        if d['kind']=='end' and d['exact'].get('complete',True):
            v,best,n=oracle(s);assert v==d['exact']['value'] and {tuple(row(m)) if m else None for m in best}=={tuple(m) if m else None for m in d['exact']['best_moves']}
            # All-legal solver per candidate, independently checked root optimum.
            vals={}
            for q in legal_moves(s):
                c=child(s,q.move)
                value=(c.score(s.turn)-c.score(1-s.turn)) if c.done else -oracle(c)[0]
                vals[tuple(row(q.move))]=value
            assert max(vals.values())==v
            chosen=[dict(method=z['method'],diverse=z['diverse'],seed=z['seed'],regret=v-vals[tuple(z['move']) if z['move'] else None]) for z in d['experiments']]
            exact_rows.append(dict(game=d['game'],value=v,nodes=n,experiments=chosen))
    write(ROOT/'exact-diagnostic.json',exact_rows)
    write(ROOT/'fixed-allocation-timing.json',fixed_timings)
    return dict(source_games=12,states=24,independently_solved_endgames=len(exact_rows),capped_endgames=12-len(exact_rows),replayed_terminal_traces=trace_count,regenerated_reference_labels=sample_count,fixed_allocation_repeats=fixed_count),source_states

def audit_match(r,fixture):
    assert r['prefix_sha256']==fixture['sha256'] and r['checkpoint_sha256']==sha(BASE)
    assert r['left']['spec']['budget_ms']==r['right']['spec']['budget_ms']==100
    prefixes={p['id']:p for p in fixture['prefixes']};indices={p['id']:i for i,p in enumerate(fixture['prefixes'])};pairs=defaultdict(list);allstates=set();times=[[],[]];counts=[defaultdict(int),defaultdict(int)]
    assert len({state_hash(prefix_state(p)) for p in fixture['prefixes']})==len(prefixes)
    for g in r['games']:
        pairs[g['prefix_id']].append(g['seat']);s=prefix_state(prefixes[g['prefix_id']]);hashes=[state_hash(s)]
        for j,side in enumerate(['left','right']):
            if r[side]['spec']['method']=='old':assert g['cold'][j]==dict(cost_estimate=.0008)
            else:assert g['cold'][j]==dict(turn=0,seed=760000+indices[g['prefix_id']]*2+(g['seat'] if j==0 else 1-g['seat']))
        assert len(g['moves'])==len(g['decisions'])
        for m,z in zip(g['moves'],g['decisions']):
            who=0 if s.turn==g['seat'] else 1;assert z['who']==who and z['remaining_pieces']==sum(v.bit_count() for v in s.remaining)
            replay(s,[m]);hashes.append(state_hash(s));times[who].append(z['ms'])
            for k in ['completed','committed','fallback','completed_roots']:
                if k in z:counts[who][k]+=z[k]
            if 'committed' in z:assert z['completed']-z['committed']==z['discarded_completed'] and z['completed']<=32 and (not z['fallback'] or z['committed']==0)
        assert s.done and hashes==g['state_hashes'] and g['scores']==[s.score(0),s.score(1)] and g['difference']==s.score(g['seat'])-s.score(1-g['seat']);allstates.update(hashes)
    assert set(pairs)==set(prefixes) and all(sorted(v)==[0,1] for v in pairs.values()) and summarize(r['games'])==r['summary']
    for i,side in enumerate(['left','right']):
        z=r[side]['stats'];a=np.asarray(times[i]);assert z['calls']==len(a) and z['overruns']==sum(a>100)
        assert abs(z.get('mean_ms',z.get('milliseconds_per_move'))-a.mean())<1e-8
        for k,v in counts[i].items():assert z['fallbacks' if k=='fallback' else k]==v
    return allstates

def main():
    start=time.perf_counter();info,source=audit_diagnostic()
    if (ROOT/'holdout-results.json').exists():
        vr=json.loads((ROOT/'validation-results.json').read_text());hr=json.loads((ROOT/'holdout-results.json').read_text());vf=read_prefixes(ROOT/'validation-prefixes.json');hf=read_prefixes(ROOT/'holdout-prefixes.json');sel=json.loads((ROOT/'selection.json').read_text());dev=json.loads((ROOT/'dev-selection.json').read_text())
        assert sel['spec']==max(vr,key=lambda r:r['summary']['mean_difference'])['left']['spec']==hr[0]['left']['spec']
        data=json.loads((ROOT/'diagnostic.json').read_text());crit={}
        for diverse in [False,True]:
            loss=[]
            for d in data:
                if d['kind']!='mid':continue
                best=max(x['mean'] for x in d['references']);by={tuple(x['move']) if x['move'] else None:x['mean'] for x in d['references']}
                loss.append(np.mean([best-by[tuple(z['move']) if z['move'] else None] for z in d['experiments'] if z['diverse']==diverse]))
            crit[str(diverse)]=float(np.mean(loss))
        assert crit==dev['values'] and dev['diverse']==min([False,True],key=lambda x:crit[str(x)])
        assert all(r['left']['spec']['diverse']==dev['diverse'] for r in vr)
        vs=set();hs=set()
        for r in vr:vs.update(audit_match(r,vf))
        for r in hr:hs.update(audit_match(r,hf))
        assert not vs&hs and not source&(vs|hs)
        info.update(selection='PASS',state_overlap=0,games=sum(len(r['games']) for r in vr+hr))
    info.update(status='PASS',seconds=time.perf_counter()-start);write(ROOT/'audit.json',info);print(info,flush=True)
if __name__=='__main__':main()
