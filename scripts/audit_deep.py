"""Replay frozen search experiments, accounting, certificates and prefix separation."""
import argparse
import json
from pathlib import Path
import sys
from collections import defaultdict
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from blokus.engine import State,Move,play,legal_moves
from blokus.tournament import prefix_state,digest,summarize
from blokus.visited_data import state_hash
from blokus.research import sha
from blokus.value_experiment import write
from tests.test_deep_search import oracle


def audit(report_path,fixture_path):
    report=json.loads(Path(report_path).read_text());fixture=json.loads(Path(fixture_path).read_text())
    assert report['prefix_sha256']==fixture['sha256']==digest(fixture['prefixes']);prefixes={p['id']:p for p in fixture['prefixes']};states=set();certs=[]
    for r in report['results']:
        assert r['checkpoint_sha256']==sha(r['checkpoint']) and r['allowances_ms'][0]==r['allowances_ms'][1]
        seats=defaultdict(list);times=[[],[]];details=[[],[]];game_states={}
        for g in r['games']:
            s=prefix_state(prefixes[g['prefix_id']]);hashes=[state_hash(s)]
            for i,row in enumerate(g['moves']):
                game_states[(g['prefix_id'],g['seat'],i)]=state_hash(s)
                m=Move(*row) if row else None
                if m is None:assert not legal_moves(s)
                play(s,m,validate=m is not None);hashes.append(state_hash(s))
            assert s.done and hashes==g['state_hashes'] and g['scores']==[s.score(0),s.score(1)] and g['difference']==s.score(g['seat'])-s.score(1-g['seat'])
            assert len(g['moves'])==len(g['decision_ms'])==len(g['details'])
            seats[g['prefix_id']].append(g['seat']);states.update(hashes)
            for (who,ms),detail in zip(g['decision_ms'],g['details']):
                times[who].append(ms)
                if detail:assert abs(detail['ms']-ms)<1e-10;details[who].append(detail)
        assert set(seats)==set(prefixes) and all(sorted(x)==[0,1] for x in seats.values()) and r['summary']==summarize(r['games'])
        for i,side in enumerate(['left','right']):
            t=np.array(times[i]);z=r[side]['stats'];assert z['calls']==len(t) and abs(z['milliseconds_per_move']-t.mean())<1e-9 and abs(z['total_seconds']-t.sum()/1000)<1e-9
            assert z['overruns']==int((t>r['allowances_ms'][i]).sum())
            if details[i]:
                ds=details[i];active=[d for d in ds if d['phase'] not in ['old-opening','forced-pass']]
                assert z['exact_solved']==sum(d['exact'] for d in ds) and z['exact_attempts']==sum(d['exact_attempt'] for d in ds) and z['nodes']==sum(d['nodes'] for d in ds)
                assert z['selective_calls']==len(active)
                for depth,n in z['depth_histogram'].items():assert n==sum(d['completed_depth']==int(depth) for d in active)
        for c in r['exact_certificates']:
            assert state_hash(State(**c['state']))==c['state_hash']==game_states[(c['prefix_id'],c['seat'],c['ply'])]
            certs.append(c)
        assert len(r['exact_certificates'])==sum(r[side]['stats'].get('exact_solved',0) for side in ['left','right'])
    return dict(matches=len(report['results']),games=sum(len(r['games']) for r in report['results']),prefixes=len(prefixes),certificates=len(certs)),states,certs


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='artifacts/deep-search');a=p.parse_args();root=Path(a.root);result={};all_certs=[];states={}
    for phase in ['dev','six','validation','holdout']:
        if not (root/f'{phase}-results.json').exists():continue
        info,hs,cs=audit(root/f'{phase}-results.json',root/f'{phase}-prefixes.json');result[phase]=info;states[phase]=hs;all_certs.extend(cs)
    for a in states:
        for b in states:
            if a<b:assert not states[a]&states[b],(a,b)
    selection=json.loads((root/'selection.json').read_text());validation=json.loads((root/'validation-results.json').read_text())['results']
    best=max(validation,key=lambda r:r['summary']['mean_difference']);assert selection['spec']==best['left']['spec']
    hold=json.loads((root/'holdout-results.json').read_text())['results'];assert len(hold)==1 and hold[0]['left']['spec']==selection['spec']
    unique={c['state_hash']:c for c in all_certs};chosen=list(unique.values());proofs=[]
    for i in np.linspace(0,len(chosen)-1,min(12,len(chosen)),dtype=int):
        c=chosen[i];s=State(**c['state']);v,moves,nodes=oracle(s);m=Move(*c['move']) if c['move'] else None
        assert v==c['value'] and m in moves
        proofs.append(dict(state_hash=c['state_hash'],value=v,independent_nodes=nodes))
    result.update(status='PASS',selection='PASS',cross_phase_exact_state_overlap=0,independent_certificate_replay=proofs,unique_certificates=len(unique))
    write(root/'audit.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
