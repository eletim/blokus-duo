"""Recorded-game statistics. Positive komi adds points to the second player."""
import hashlib
import json
from collections import defaultdict
import numpy as np
from .console import Sources, read_json
from .engine import State, Move, play, SIZES


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def records(artifacts):
    source=Sources(artifacts,'/nonexistent')
    for stage,path in source.shards():
        data=read_json(path);phase=path.name.split('-')[0]
        bank=read_json(path.parent/f'{phase}-bank.json')['prefixes']
        for index,g in enumerate(data['games']):
            prefix=bank[g['prefix']]['moves'];moves=prefix+g['moves'];s=State()
            for m in moves: play(s,Move(*m) if m is not None else None,validate=False)
            assert s.done and [s.score(p) for p in (0,1)]==g['scores']
            remaining=[sum(SIZES[i] for i in range(21) if s.remaining[p]>>i&1) for p in (0,1)]
            yield dict(stage=stage,phase=phase,seed=data['seed'],left=source.model(data['left'],stage),
                right=source.model(data['right'],stage),seat=g['seat'],scores=g['scores'],remaining=remaining,
                margin=g['scores'][0]-g['scores'][1],unplaced_advantage=remaining[1]-remaining[0],
                budget_ms=data['budget_ms'],max_depth=data['max_depth'],width=4,
                prefix_plies=len(prefix),cluster=digest(prefix[:8]),prefix=prefix,moves=moves,
                source=str(path.relative_to(source.artifacts)),index=index)


def summary(rows, values=None):
    if not rows:return None
    x=np.array([r['margin'] for r in rows] if values is None else values,dtype=float)
    clusters=defaultdict(list)
    for r,v in zip(rows,x):clusters[r['cluster']].append(v)
    keys=sorted(clusters);rng=np.random.default_rng(20261127)
    totals=np.array([[sum(np.array(clusters[k])>0),sum(np.array(clusters[k])==0),sum(clusters[k]),len(clusters[k])] for k in keys])
    b=totals[rng.integers(len(keys),size=(2000,len(keys)))].sum(1)
    ci=lambda v:np.quantile(v,[.025,.975]).tolist() if len(keys)>1 else None
    win=x[x>0]
    return dict(n=len(x),clusters=len(keys),wins=int(sum(x>0)),draws=int(sum(x==0)),losses=int(sum(x<0)),
        win_rate=float(np.mean(x>0)),point_rate=float(np.mean((x>0)+.5*(x==0))),
        mean=float(x.mean()),median=float(np.median(x)),quantiles=np.quantile(x,[0,.1,.25,.5,.75,.9,1]).tolist(),
        distribution=[dict(margin=float(v),count=int(sum(x==v))) for v in sorted(set(x))],
        winning_margin_mean=float(win.mean()) if len(win) else None,
        winning_margin_median=float(np.median(win)) if len(win) else None,
        mean_ci95=ci(b[:,2]/b[:,3]),win_rate_ci95=ci(b[:,0]/b[:,3]),point_rate_ci95=ci((b[:,0]+.5*b[:,1])/b[:,3]),
        uncertainty='prefix単位cluster bootstrap 2000回。seed・先後ペアは同じcluster。母集団の無作為標本とは限らない。')


def analyze(rows):
    groups=defaultdict(list)
    for r in rows:
        key=(r['stage'],r['phase'],r['left'],r['right'],r['budget_ms'],r['max_depth'],r['prefix_plies'])
        groups[key].append(r)
    out=[]
    for key,rs in groups.items():
        left_seats={}
        for seat in (0,1):
            sub=[r for r in rs if r['seat']==seat]
            left_seats[str(seat)]=summary(sub,[r['margin']*(1 if seat==0 else -1) for r in sub])
        out.append(dict(stage=key[0],phase=key[1],left=key[2],right=key[3],budget_ms=key[4],max_depth=key[5],prefix_plies=key[6],
            first=summary(rs),second=summary(rs,[-r['margin'] for r in rs]),
            unplaced_first=summary(rs,[r['unplaced_advantage'] for r in rs]),left_by_seat=left_seats,
            komi=[dict(k=k,**summary(rs,[r['margin']-k for r in rs])) for k in (0,1,2,3,4,5,6,8,10,12)],
            by_seed={str(seed):summary([r for r in rs if r['seed']==seed]) for seed in sorted({r['seed'] for r in rs})},
            rows=[{k:r[k] for k in ('margin','unplaced_advantage','seat','cluster','seed')} for r in rs]))
    return dict(groups=out,total_games=len(rows),bonus_difference_games=sum(r['margin']!=r['unplaced_advantage'] for r in rows),
        notes='全件は8/10/12手の共通prefix開始・異種AI対戦。正規初期盤面の同一AIコミ根拠ではない。勝率=勝/全局、得点率=(勝+0.5分)/全局。コミkは後手に加点。候補は再採点のみ。')
