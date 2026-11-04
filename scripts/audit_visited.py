"""Replay evaluation and labels; audit semantic state leakage beyond openings."""
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from blokus.engine import State,Move,legal_moves,play
from blokus.tournament import prefix_state,digest,summarize
from blokus.research import sha,clone
from blokus.visited_data import state_hash,label_one
from blokus.aligned_data import initialize
from blokus.visited_experiment import weight_hash
from blokus.value_experiment import write


def evaluate_audit(path,fixture_path):
    report=json.loads(Path(path).read_text());fixture=json.loads(Path(fixture_path).read_text());assert report['prefix_sha256']==fixture['sha256']==digest(fixture['prefixes'])
    prefixes={p['id']:p for p in fixture['prefixes']};all_states=set();middle=set()
    assert len({state_hash(prefix_state(p)) for p in prefixes.values()})==len(prefixes)
    for r in report['results']:
        seats=defaultdict(list);times=[[],[]]
        for g in r['games']:
            s=prefix_state(prefixes[g['prefix_id']]);hashes=[state_hash(s)]
            for row in g['moves']:play(s,Move(*row) if row else None);hashes.append(state_hash(s))
            assert s.done and hashes==g['state_hashes'] and g['scores']==[s.score(0),s.score(1)]
            assert g['difference']==s.score(g['seat'])-s.score(1-g['seat']);seats[g['prefix_id']].append(g['seat'])
            all_states.update(hashes);middle.update(hashes[6:])
            for who,ms in g['decision_ms']:times[who].append(ms)
        assert set(seats)==set(prefixes) and all(sorted(s)==[0,1] for s in seats.values())
        diffs=[g['difference'] for g in r['games']];z=r['summary'];assert z['wins']==sum(d>0 for d in diffs) and z['draws']==sum(d==0 for d in diffs) and z['losses']==sum(d<0 for d in diffs)
        assert abs(z['mean_difference']-np.mean(diffs))<1e-10
        assert z==summarize(r['games'])
        for i,side in enumerate(['left','right']):
            t=np.array(times[i]);st=r[side]['stats'];assert len(t)==st['calls'] and abs(t.mean()-st['milliseconds_per_move'])<1e-9
            assert abs(t.sum()/1000-st['total_seconds'])<1e-9 and st['overruns']==int((t>r['allowances_ms'][i]).sum())
            spec=r[side]['spec']
            if spec.get('checkpoint'):assert sha(spec['checkpoint'])==r[side]['checkpoint_sha256']
    return dict(status='PASS',matches=len(report['results']),prefixes=len(prefixes)),all_states,middle


def source_audit(path):
    source=json.loads(Path(path).read_text());states=set();middle=set()
    game_states={}
    for g in source['games']:
        s=State();hs=[state_hash(s)]
        for row in g['moves']:play(s,Move(*row) if row else None);hs.append(state_hash(s))
        assert hs==g['state_hashes'] and s.done;states.update(hs);middle.update(hs[6:]);game_states[g['game']]=set(hs)
    for row in source['records']:
        assert row['root_hash']==state_hash(State(**row['root'])) and row['root_hash'] in game_states[row['game']]
        assert row['completed_candidates']>=4
        assert row['leaf_hashes']==[[state_hash(State(**leaf)) for leaf in branch] for branch in row['leaves']]
    return states,middle


def two_ply_replay(root,target):
    s=clone(root)
    for _ in range(2):
        p=s.turn;desired=target.boards[p];qs=legal_moves(s)
        if desired==s.boards[p]:play(s,None)
        else:
            match=[q for q in qs if (s.boards[p]|q.bits)==desired]
            assert len(match)==1;play(s,match[0].move)
    assert state_hash(s)==state_hash(target)


def dataset_audit(path,replay=True):
    path=Path(path);d=np.load(path);rows=json.loads(path.with_suffix('.records.json').read_text());meta=json.loads(path.with_suffix('.json').read_text());assert sha(path)==meta['sha256']
    states={r['root_hash'] for r in rows}
    for row in rows:states.update(h for branch in row['leaf_hashes'] for h in branch)
    if replay:
        for i in np.linspace(0,len(rows)-1,4,dtype=int):
            row=rows[i];root=State(**row['root'])
            for branch in row['leaves']:
                for leaf in branch:two_ply_replay(root,State(**leaf))
            result=label_one((row,min(4,meta['repeats'])))
            assert np.array_equal(result[0],d['board'][i]) and np.array_equal(result[1],d['state'][i])
            assert np.array_equal(result[2],d['returns'][i,...,:min(4,meta['repeats'])]) and np.array_equal(result[4],d['terminal'][i]) and np.array_equal(result[5],d['score'][i])
    return states


def main():
    root=Path('artifacts/visited-diversity');initialize();result={};train_states=set();source_middle=set();validation_states=set()
    _,dev_middle=source_audit(root/'dev-visited.json');dev_states=dataset_audit(root/'dev.npz')
    for stage in ['pilot','full']:
        folder=root/stage
        if not (folder/'summary.json').exists():continue
        _,mid=source_audit(folder/'visited.json');source_middle.update(mid)
        wide=dataset_audit(folder/'wide.npz');dense=dataset_audit(folder/'dense.npz');assert dense<=wide and not wide&dev_states;train_states.update(wide)
        wr=json.loads((folder/'wide.records.json').read_text());dr=json.loads((folder/'dense.records.json').read_text());wd=np.load(folder/'wide.npz');dd=np.load(folder/'dense.npz');wi={r['root_hash']:i for i,r in enumerate(wr)}
        for j,r in enumerate(dr):assert np.array_equal(dd['returns'][j,...,:4],wd['returns'][wi[r['root_hash']]])
        selection=json.loads((folder/'selection.json').read_text());records=json.loads((folder/'validation-results.json').read_text())
        for choice in selection['choices']:
            vals=[r['summary']['mean_difference'] for r in records if r['condition']==choice['condition'] and r['selector']==choice['selector']]
            assert len(vals)==6 and abs(np.mean(vals)-choice['criterion'])<1e-12
        assert selection['chosen']['criterion']==max(c['criterion'] for c in selection['choices'])
        # Validate every saved selector against its own per-update criterion.
        for condition in ['wide','dense']:
            for seed in [601,602,603]:
                out=folder/condition/f'seed-{seed}';log=[json.loads(s) for s in (out/'train.jsonl').read_text().splitlines()]
                for selector in ['raw','live']:
                    ck=torch.load(out/f'{selector}.pt',weights_only=False);best=min(log,key=lambda z:(z[selector]['decision_regret_points'],z[selector]['pair_mse']))
                    assert ck['step']==best['step']
        # Validation is a list, adapt once for the common replay auditor.
        temp=folder/'validation-audit-input.json';write(temp,dict(prefix_sha256=records[0]['prefix_sha256'],results=records))
        report,states,mid=evaluate_audit(temp,folder/'validation-prefixes.json');temp.unlink()
        assert not states&train_states and not states&dev_states and not mid&source_middle
        validation_states.update(states)
        for r in records:assert weight_hash(r['left']['spec']['checkpoint'])==r['weight_sha256']
        result[stage]=dict(selection='PASS',validation=report,dense_first_four_labels_equal_wide=True)
    assert not source_middle&dev_middle
    assert not validation_states&train_states and not validation_states&dev_states and not validation_states&source_middle and not validation_states&dev_middle
    final,states,mid=evaluate_audit(root/'holdout-results.json',root/'holdout-prefixes.json');assert not states&train_states and not states&dev_states and not mid&source_middle and not mid&dev_middle and not states&validation_states and not states&source_middle and not states&dev_middle
    curve,curve_states,curve_mid=evaluate_audit(root/'curve-results.json',root/'curve-prefixes.json');assert not states&curve_states
    report=json.loads((root/'holdout-results.json').read_text());fixture=json.loads((root/'holdout-prefixes.json').read_text());estimates={};rng=np.random.default_rng(554433)
    selected=json.loads((root/'final-selection.json').read_text());choice=selected['chosen']
    assert choice==json.loads((root/selected['stage']/'selection.json').read_text())['chosen']
    assert [r['names'] for r in report['results']]==[[f'visited-{s}',o] for s in [601,602,603] for o in ['static','prior']]+[['prior','static']]
    for r in report['results']:
        if r['names'][0].startswith('visited-'):
            seed=r['names'][0].split('-')[-1];expected=root/selected['stage']/choice['condition']/f'seed-{seed}'/f"{choice['selector']}.pt"
            assert sha(expected)==r['left']['checkpoint_sha256']
    assert all(r['allowances_ms']==[6,6] for r in report['results'])
    for opponent in ['static','prior']:
        rows=[]
        for r in report['results']:
            if r['names'][0].startswith('visited-') and r['names'][1]==opponent:rows.append([np.mean([g['difference'] for g in r['games'] if g['prefix_id']==p['id']]) for p in fixture['prefixes']])
        a=np.array(rows);assert a.shape==(3,96);si=rng.integers(3,size=(10000,3));pi=rng.integers(96,size=(10000,96));means=a[si[:,:,None],pi[:,None,:]].mean((1,2))
        estimates[opponent]=dict(mean=float(a.mean()),ci95=np.quantile(means,[.025,.975]).tolist(),method='paired-seat average, crossed seed/prefix bootstrap')
    result.update(holdout=final,curve=curve,leakage='PASS: semantic exact root/leaf and source-game middle/end hashes disjoint from dev/evaluation; full label-continuation interiors not audited',train_root_leaf_states=len(train_states),dev_root_leaf_states=len(dev_states),source_middle_states=len(source_middle),crossed_bootstrap=estimates)
    write(root/'audit.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
