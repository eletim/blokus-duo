"""Post-measurement independent replay, paired throughput and split audit."""
import gzip,json
from pathlib import Path
import numpy as np
import gpu_generation as g
from audit_placement import replay
from blokus.engine import State,Move,play
from blokus.ppo_runtime import state_hash

def main():
    bank=json.loads((g.R/'bank.json').read_text());assert len({x['rng_seed'] for x in bank['games']})==24
    ex=set(json.loads((g.R/'excluded.json').read_text())['states']);idx=np.load(g.e.INDEX,mmap_mode='r')
    assert len({x['prefix']['state_hash'] for x in bank['games']})==24
    out={};moves_equal=0;total_pairs=0;root_error=0.;prefix_overlap=0
    for mode,repeat in [('cpu',1),('gpu',1),('gpu',2),('cpu',2)]:
        rows=[];seen=set();new=[]
        for task in bank['games']:
            p=g.R/f'{mode}-{repeat}'/f"{task['seed']}-{task['id']}.json.gz"
            d=json.load(gzip.open(p,'rt'));assert d['task']==task
            assert replay(d['moves'])==d['scores'];assert len(d['stats'])==len(d['moves'])-8
            s=State()
            for i,m in enumerate(d['moves']):
                if i>=8:seen.add(state_hash(s))
                play(s,Move(*m) if m else None,validate=True)
            assert s.done
            assert all(r['forced_pass'] or r['depth']==6 for r in d['stats'])
            rows.extend(d['stats'])
            if mode=='gpu':
                c=json.load(gzip.open(g.R/f'cpu-{repeat}'/p.name,'rt'))
                total_pairs+=1;moves_equal+=d['moves']==c['moves']
                # Root scores comparable only while the actual state is identical.
                for j,(a,b) in enumerate(zip(d['stats'],c['stats'])):
                    if d['moves'][:j+8]!=c['moves'][:j+8]:break
                    av=a.get('root_scores',[]);bv=b.get('root_scores',[])
                    assert len(av)==len(bv)
                    root_error=max(root_error,max([abs(x-y) for x,y in zip(av,bv)] or [0]))
        timing=json.loads((g.R/f'{mode}-{repeat}.json').read_text());n=sum(w['games'] for w in timing['workers']);assert n==24
        nonpass=[s for s in rows if not s['forced_pass']]
        out[f'{mode}-{repeat}']=dict(games=n,seconds=timing['seconds'],games_per_second=n/timing['seconds'],
            moves=len(rows),nodes=sum(s['nodes'] for s in rows),nodes_per_second=sum(s['nodes'] for s in rows)/timing['seconds'],
            nn_policy_calls=sum(s['nn_policy_calls'] for s in rows),nn_value_calls=sum(s['nn_value_calls'] for s in rows),
            completed_depth6=sum(s['depth']==6 for s in nonpass),nonpass=len(nonpass),
            visited=len(seen),prior_saved_overlap=len(seen&ex),prior_training_overlap=sum(g.e.contains(idx,h) for h in seen))
    ratios=[out[f'cpu-{i}']['seconds']/out[f'gpu-{i}']['seconds'] for i in (1,2)]
    numeric=json.loads((g.R/'numeric-check.json').read_text())
    decision=dict(passed=numeric['passed'] and root_error<=2e-5 and all(r>=1.15 for r in ratios) and all(v['prior_saved_overlap']==v['prior_training_overlap']==0 for v in out.values()),speed_ratios=ratios,
                  exact_game_trajectories=moves_equal,paired_games=total_pairs,max_same_state_root_error=root_error,
                  numeric_passed=numeric['passed'],results=out,bank_sha=g.e.sha(g.R/'bank.json'),
                  additional_training=False,reason='Stage2 requires both throughput repeats >=1.15x and numerical quality.')
    g.write(g.R/'audit.json',decision);print(json.dumps(decision,indent=2))
if __name__=='__main__':main()
