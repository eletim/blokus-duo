"""Replay data, frozen teacher depths, candidates, selection and paired arithmetic."""
import sys,json,gzip,time
from pathlib import Path
from collections import defaultdict
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from blokus.engine import State,Move,play,legal_moves,is_legal
from blokus.research import clone,sha
from blokus.visited_data import state_hash
from blokus.ai import features,heuristic,seed_all
from blokus.root_policy import inputs
from blokus.root_data import ROOT
from blokus.value_search import ValueEvaluator
from blokus.refinement_search import LimitedTree
from blokus.aligned_data import BASE
from blokus.node_diagnostic import child
from blokus.tournament import read_prefixes,prefix_state,summarize
from blokus.value_experiment import write

def move(r):return Move(*r) if r else None
def replay(s,r):
    m=move(r)
    if m is None:assert not legal_moves(s)
    play(s,m,validate=m is not None)

def audit_data():
    seed_all(901);sources=json.load(gzip.open(ROOT/'source-games.json.gz','rt'));records=json.load(gzip.open(ROOT/'teacher-traces.json.gz','rt'));data=np.load(ROOT/'data.npz');states={};all_hashes=set()
    for g in sources:
        s=State()
        for ply,m in enumerate(g['moves']):states[g['game'],ply]=state_hash(s);all_hashes.add(state_hash(s));replay(s,m)
        assert s.done and g['scores']==[s.score(0),s.score(1)]
    row=0;split_hashes=[set(),set(),set()];all_split=[set(),set(),set()];label_counts=[];replayed=0;tree=LimitedTree(ValueEvaluator(BASE,'cuda',.5,24))
    for j,r in enumerate(records):
        s=clone(State(**r['state']));assert states[r['game'],r['ply']]==r['state_hash']==state_hash(s);all_split[r['split']].add(r['state_hash'])
        qs=legal_moves(s);x,st=features(s,qs);order=np.argsort(-heuristic(x),kind='stable')[:128];ms=[qs[int(i)].move for i in order]
        assert [move(m) for m in r['moves']]==ms and all(is_legal(s,m) for m in ms)
        import hashlib
        assert hashlib.sha256(json.dumps(r['moves']).encode()).hexdigest()==r['candidate_hash']
        assert len(r['depths'])==len(ms) and set(r['depths'])<={0,2,4}
        assert sum(d>0 for d in r['depths'])==r['decision']['completed_roots'] and sum(d==4 for d in r['depths'])==r['decision']['refined']
        assert r['delta']==[None if a is None or b is None else b-a for a,b in zip(r['initial'],r['final'])]
        known=[v for v in r['final'] if v is not None]
        if known:assert r['final'][r['target']]>=max(known)-4e-6
        if r['trainable']:
            assert data['game'][row]==r['game'] and data['split'][row]==r['split'] and data['target'][row]==r['target']
            xx,ss=inputs(s,[qs[int(i)] for i in order],x[order],st);assert np.array_equal(data['x'][row,:len(ms)],xx) and np.array_equal(data['state'][row],ss)
            expected=np.array(r['depths'])==r['depths'][r['target']];assert np.array_equal(expected,data['mask'][row,:len(ms)]) and expected.sum()>=2
            assert not data['mask'][row,len(ms):].any() and data['legal_mask'][row,:len(ms)].all() and not data['legal_mask'][row,len(ms):].any()
            split_hashes[r['split']].add(r['state_hash']);label_counts.append(int(expected.sum()));row+=1
        if j%50==0:
            for i,m in enumerate(ms):
                if r['depths'][i]==0:continue
                c=child(s,m);v=-tree.search(c,1);assert abs(v-r['initial'][i])<1e-5
                final=-tree.search(c,3) if r['depths'][i]==4 else v;assert abs(final-r['final'][i])<1e-5
            replayed+=1
    assert row==len(data['x'])
    overlaps={f'{i}/{j}':len(split_hashes[i]&split_hashes[j]) for i in range(3) for j in range(i)};assert not any(overlaps.values())
    return dict(source_games=len(sources),records=len(records),training_rows=row,teacher_replayed_states=replayed,same_depth_comparison_mean=float(np.mean(label_counts)),eligible_overlap=overlaps,unused_overlap={f'{i}/{j}':len(all_split[i]&all_split[j]) for i in range(3) for j in range(i)},data_sha256=sha(ROOT/'data.npz')),all_hashes

def audit_matches():
    reports=[];all_hashes=set();count=0;phase_hashes={}
    for name in ['teacher','validation','holdout']:
        path=ROOT/f'{name}-results.json'
        if not path.exists():continue
        phase_hashes[name]=set()
        fixture=read_prefixes(ROOT/f'{name}-prefixes.json');by={p['id']:p for p in fixture['prefixes']};raw=json.loads(path.read_text());matches=[raw] if isinstance(raw,dict) else raw
        for r in matches:
            assert r['prefix_sha256']==fixture['sha256'];assert r.get('value_sha256',r.get('checkpoint_sha256'))==sha(BASE)
            for side in ['left','right']:
                if r[side]['spec'].get('policy'):assert r[side]['policy_sha256']==sha(r[side]['spec']['policy'])
            times=[[],[]];roots=[0,0];cluster=defaultdict(list)
            for g in r['games']:
                s=prefix_state(by[g['prefix_id']]);assert g['cold'] and state_hash(s)==g['state_hashes'][0]
                for i,(m,d) in enumerate(zip(g['moves'],g['decisions'])):
                    who=0 if s.turn==g['seat'] else 1;assert d['who']==who;times[who].append(d['ms']);roots[who]+=d['completed_roots'];all_hashes.add(state_hash(s));phase_hashes[name].add(state_hash(s));replay(s,m);assert state_hash(s)==g['state_hashes'][i+1]
                    if 'policy_ms' in d:assert 0<=d['policy_ms']<=d['ms'] and d['completed_roots']<=d['candidates']
                assert s.done and g['scores']==[s.score(0),s.score(1)] and g['difference']==s.score(g['seat'])-s.score(1-g['seat']);cluster[g['prefix_id']].append((g['seat'],g['difference']))
            assert all(sorted(s for s,d in x)==[0,1] for x in cluster.values()) and len(cluster)==len(by)
            ds=[g['difference'] for g in r['games']];summary=r['summary'];assert summary['wins']==sum(x>0 for x in ds) and summary['losses']==sum(x<0 for x in ds) and summary['draws']==sum(x==0 for x in ds)
            assert abs(sum(ds)/len(ds)-summary['mean_difference'])<1e-12 and summarize(r['games'])==summary
            means=np.array([sum(d for s,d in x)/2 for x in cluster.values()]);idx=np.random.default_rng(9817).integers(len(means),size=(4000,len(means)));assert np.allclose(np.quantile(means[idx].mean(1),[.025,.975]),summary['difference_ci95'])
            for i,side in enumerate(['left','right']):
                st=r[side]['stats'];assert len(times[i])==st['calls'] and abs(sum(times[i])/1000-st['total_seconds'])<1e-8 and roots[i]==st['completed_roots'];assert st['overruns']==sum(t>st['allowance_ms'] for t in times[i])
            reports.append(dict(phase=name,left=r['left']['spec'],right=r['right']['spec'],summary=summary));count+=len(ds)
    return dict(games=count,matches=reports,phase_overlap={a+'/'+b:len(phase_hashes[a]&phase_hashes[b]) for a in phase_hashes for b in phase_hashes if a<b}),all_hashes

def main():
    frozen=json.loads((ROOT/'teacher-frozen.json').read_text())
    actual=json.loads((ROOT/'teacher-results.json').read_text());assert actual['left']['spec']==frozen['spec']
    for path,digest in frozen['source_hashes'].items():
        if Path(path).name in ['refinement_search.py','value_search.py','engine.py','deep_search.py']:
            assert sha(path)==digest
    start=time.perf_counter();d,dh=audit_data();m,mh=audit_matches();selection=json.loads((ROOT/'selection.json').read_text()) if (ROOT/'selection.json').exists() else None
    if selection:
        assert max(selection['criteria'],key=selection['criteria'].get)==f"{selection['budget_ms']}/{selection['step']}"
        for seed,spec in selection['policies'].items():assert sha(spec['path'])==spec['sha256']
    pooled_checked=False
    if (ROOT/'holdout-pooled.json').exists():
        hold=json.loads((ROOT/'holdout-results.json').read_text());reported=json.loads((ROOT/'holdout-pooled.json').read_text())
        for name in ['old','untrained']:
            rows=[r for r in hold if r['right']['spec']['method']==name];assert {r['left']['spec']['seed'] for r in rows}=={901,902,903}
            by=defaultdict(list)
            for r in rows:
                spec=r['left']['spec'];assert spec['step']==selection['step'] and spec['budget_ms']==selection['budget_ms']
                assert r['left']['policy_sha256']==selection['policies'][str(spec['seed'])]['sha256']
                for g in r['games']:by[g['prefix_id']].append(g['difference'])
            assert len(by)==96 and all(len(v)==6 for v in by.values())
            means=np.array([sum(v)/6 for v in by.values()]);idx=np.random.default_rng(9817).integers(96,size=(4000,96))
            assert abs(float(means.mean())-reported[name]['mean_difference'])<1e-12
            assert np.allclose(np.quantile(means[idx].mean(1),[.025,.975]),reported[name]['difference_ci95'])
            wins=np.array([sum(x>0 for x in v)/6 for v in by.values()]);points=np.array([sum((x>0)+.5*(x==0) for x in v)/6 for v in by.values()])
            assert np.allclose(np.quantile(wins[idx].mean(1),[.025,.975]),reported[name]['win_ci95'])
            assert np.allclose(np.quantile(points[idx].mean(1),[.025,.975]),reported[name]['point_ci95'])
        pooled_checked=True
    result=dict(pass_=True,pooled_checked=pooled_checked,data=d,evaluation=m,source_game_vs_evaluation_hash_overlap=len(dh&mh),selection_checked=bool(selection),seconds=time.perf_counter()-start)
    write(ROOT/'audit.json',result);print('AUDIT PASS',d,m['games'],result['seconds'],flush=True)
if __name__=='__main__':main()
