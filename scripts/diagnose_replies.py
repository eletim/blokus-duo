"""Finite independent-game diagnostic; GPU value weights remain fixed."""
import sys,json,random,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from blokus.node_diagnostic import *
from blokus.engine import State
from blokus.aligned_data import BASE
from blokus.ai import seed_all
from blokus.value_search import ValueEvaluator
from blokus.visited_data import state_dict,state_hash
from blokus.value_experiment import write
from blokus.research import sha,source_hashes

def main():
    root=Path('artifacts/reply-diagnostics');root.mkdir(parents=True,exist_ok=True)
    if (root/'diagnostic.json').exists():raise ValueError('existing diagnostic')
    seed_all(801);ev=ValueEvaluator(BASE,'cuda',.5,24);begin=time.perf_counter();data=[]
    write(root/'config.json',dict(checkpoint=BASE,sha256=sha(BASE),source_hashes=source_hashes(),games=16,seed=620000,node_limit=2048,reference_limit=50000,reference_root_width=24,reference_reply_width=12,reference_inner_width=6,reference_depth=4,validation_seed=621000,holdout_seed=622000,training_updates=0))
    for gid in range(16):
        rng=random.Random(620000+gid);s=State();history=[];moves=[]
        while not s.done:
            history.append(clone(s));ms=ordered(s);m=rng.choice(ms[:8]);moves.append(row(m));play(s,m,validate=False)
        late=next((z for z in reversed(history) if 2<=len(legal_moves(z))<=8),history[-2])
        for kind,s in [('mid',history[min(18+(gid%2)*6,len(history)//2)]),('end',late)]:
            # Depth4 reference is selective in middlegame; terminal solve is full.
            full=kind=='end';roots=ordered(s) if full else ordered(s)[:24];tree=NodeSearch(ev,50000,6);records=[];complete=True
            for m in roots:
                c=child(s,m);replies=ordered(c);replies=replies if full else replies[:12];values=[];shallow=[]
                try:
                    for r in replies:
                        leaf=child(c,r);values.append(tree.value(leaf,2,full));shallow.append(float(ev.batch([leaf])[0])*SCALE)
                except NodeLimit:complete=False;break
                records.append(dict(move=row(m),reply_moves=[row(r) for r in replies],reference=values,shallow=shallow))
            if not complete:raise RuntimeError(f'reference cap {gid}/{kind}; do not label incomplete reference')
            best=max(min(r['reference']) for r in records)
            old=max(records,key=lambda r:min(r['shallow'][:3]));oldloss=best-min(old['reference'])
            omit=[min(r['reference'][:3])-min(r['reference']) for r in records]
            leaferr=[abs(a-b) for r in records for a,b in zip(r['reference'],r['shallow'])]
            configs=[]
            for rw in [6,24]:
                for ow in [3,12]:
                    for depth in [2,4]:
                        t=NodeSearch(ev,2048,6);v=t.roots(s,rw,ow,depth);found=next(r for r in records if r['move']==v['chosen'])
                        v.update(root_width=rw,reply_width=ow,depth=depth,regret=best-min(found['reference']))
                        configs.append(v)
            data.append(dict(game=gid,kind=kind,state=state_dict(s),state_hash=state_hash(s),game_moves=moves,reference_exact=full,reference_nodes=tree.nodes,records=records,metrics=dict(old_regret=oldloss,reply_omission_rate=float(np.mean(np.array(omit)>1e-6)),reply_omission_gap=float(np.mean(omit)),leaf_mae=float(np.mean(leaferr)),root6_loss=best-max(min(r['reference']) for r in records[:6])),fixed_node=configs))
            print(gid,kind,data[-1]['metrics'],flush=True)
        write(root/'diagnostic.json',dict(states=data,seconds=time.perf_counter()-begin))
    summary={}
    for kind in ['mid','end']:
        ds=[d for d in data if d['kind']==kind];summary[kind]={k:float(np.mean([d['metrics'][k] for d in ds])) for k in ds[0]['metrics']}
        summary[kind]['configs']=[dict(root_width=rw,reply_width=ow,depth=dep,regret=float(np.mean([c['regret'] for d in ds for c in d['fixed_node'] if (c['root_width'],c['reply_width'],c['depth'])==(rw,ow,dep)]))) for rw in [6,24] for ow in [3,12] for dep in [2,4]]
    write(root/'diagnostic-summary.json',summary);print('COMPLETE',summary,flush=True)
if __name__=='__main__':main()
