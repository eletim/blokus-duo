"""Trace the frozen teacher; replay completed depths instead of wall-clock decisions."""
import gzip,json,random,time
from pathlib import Path
import numpy as np
from .engine import State,play,legal_moves
from .ai import features,heuristic,seed_all
from .research import clone,sha
from .aligned_data import BASE
from .value_search import BudgetAgent
from .reply_search import reset_game
from .visited_data import state_dict,state_hash
from .node_diagnostic import row,child
from .deep_search import key
from .root_policy import inputs,XDIM
from .value_experiment import write
from . import refinement_search as teacher
ROOT=Path('artifacts/root-policy-distillation')

class RecordingTree(teacher.LimitedTree):
    instances=[]
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.completed=[];self.instances.append(self)
    def search(self,s,depth,*args,**kwargs):
        v=super().search(s,depth,*args,**kwargs)
        if not args and not kwargs:self.completed.append((depth,key(s),v))
        return v

def trace(bot,s,rng):
    original=teacher.LimitedTree;RecordingTree.instances=[];teacher.LimitedTree=RecordingTree
    try:m=bot.act(s,rng)
    finally:teacher.LimitedTree=original
    qs=legal_moves(s);x,st=features(s,qs);order=np.argsort(-heuristic(x),kind='stable')[:128]
    tree=RecordingTree.instances[-1];depths={k:depth+1 for depth,k,v in tree.completed}
    ds=[depths.get(key(child(s,qs[int(i)].move)),0) for i in order]
    selected=next((j for j,i in enumerate(order) if qs[int(i)].move==m),None)
    initial=[bot.initial_values.get(int(i)) for i in order];final=[bot.last_values.get(int(i)) for i in order]
    return dict(state=state_dict(s),state_hash=state_hash(s),moves=[row(qs[int(i)].move) for i in order],depths=ds,initial=initial,final=final,
                delta=[None if a is None or b is None else b-a for a,b in zip(initial,final)],target=selected,decision=bot.decisions[-1],
                candidate_hash=__import__('hashlib').sha256(json.dumps([row(qs[int(i)].move) for i in order]).encode()).hexdigest())

def collect(games=300):
    seed_all(900);start=time.perf_counter();records=[];sources=[];bot=teacher.RefinementAgent(BASE,100,'priority');old=BudgetAgent('value',BASE,.5,24,6,'cuda')
    xs=[];ss=[];masks=[];targets=[];gids=[];splits=[];priors=[]
    for g in range(games):
        rng=random.Random(930000+g);s=State();history=[];moves=[];reset_game(old);bot.reset_game()
        while not s.done:
            history.append(clone(s));qs=legal_moves(s)
            if not qs:m=None
            elif rng.random()<.15:m=rng.choice(qs).move
            elif s.turn==g%2:m=old.act(s,rng)
            else:
                xx,_=features(s,qs);m=qs[int(np.argmax(heuristic(xx)))].move
            play(s,m,validate=False);moves.append(row(m))
        split=0 if g<games*.8 else 1 if g<games*.9 else 2
        sources.append(dict(game=g,seed=930000+g,split=split,moves=moves,scores=[s.score(0),s.score(1)]))
        for ply in sorted(set(int((len(history)-1)*f) for f in [.30,.45,.60,.75])):
            s=history[ply]
            if not legal_moves(s):continue
            r=trace(bot,s,random.Random(940000+g*100+ply));r.update(game=g,ply=ply,split=split)
            r['trainable']=bool(4 in r['depths'] and r['target'] is not None and sum(d>0 for d in r['depths'])>=2)
            records.append(r)
            if not r['trainable']:continue
            qs=legal_moves(s);xx,st=features(s,qs);order=np.argsort(-heuristic(xx),kind='stable')[:128];x,state=inputs(s,[qs[int(i)] for i in order],xx[order],st)
            pad=np.zeros((128,XDIM),np.float32);pad[:len(x)]=x;mask=np.zeros(128,bool);mask[:len(x)]=np.array(r['depths'])>0
            prior=np.zeros(128,np.float32);prior[:len(x)]=heuristic(xx[order]);xs.append(pad);ss.append(state);masks.append(mask);targets.append(r['target']);gids.append(g);splits.append(split);priors.append(prior)
        if (g+1)%25==0:print('games',g+1,'records',len(records),'eligible',len(xs),flush=True)
    with gzip.open(ROOT/'teacher-traces.json.gz','wt') as f:json.dump(records,f,separators=(',',':'))
    with gzip.open(ROOT/'source-games.json.gz','wt') as f:json.dump(sources,f,separators=(',',':'))
    np.savez_compressed(ROOT/'data.npz',x=xs,state=ss,mask=masks,target=targets,game=gids,split=splits,prior=priors)
    write(ROOT/'data-summary.json',dict(games=games,records=len(records),eligible=len(xs),split_rows={str(i):sum(s==i for s in splits) for i in [0,1,2]},seconds=time.perf_counter()-start,sha256=sha(ROOT/'data.npz'),teacher_checkpoint_sha256=sha(BASE),label='teacher action classification over evaluated candidates, NOT mixed-depth value regression',trace='same frozen teacher, logging completed top-level searches adds small measured overhead',shared='all update seeds share game split, labels, features, initial model'))

def prepare_labels():
    """Only same-completed-depth negatives; full candidate mask for evaluation."""
    with gzip.open(ROOT/'teacher-traces.json.gz','rt') as f:records=json.load(f)
    raw=np.load(ROOT/'data.npz');rows=[];masks=[];legals=[];index=0
    for r in records:
        if not r['trainable']:continue
        ds=np.array(r['depths']);target=r['target'];same=ds==ds[target]
        r['teacher_has_extra']=True;r['trainable']=bool(same.sum()>=2)
        if r['trainable']:
            rows.append(index);mask=np.zeros(128,bool);mask[:len(ds)]=same;masks.append(mask)
            legal=np.zeros(128,bool);legal[:len(ds)]=True;legals.append(legal)
        index+=1
    assert index==len(raw['x'])
    data={k:raw[k][rows] for k in raw.files};data['mask']=np.array(masks);data['legal_mask']=np.array(legals)
    np.savez_compressed(ROOT/'data.npz',**data)
    with gzip.open(ROOT/'teacher-traces.json.gz','wt') as f:json.dump(records,f,separators=(',',':'))
    summary=json.loads((ROOT/'data-summary.json').read_text());summary['eligible_before_same_depth']=summary['eligible'];summary['eligible']=len(rows)
    summary['split_rows']={str(i):int((data['split']==i).sum()) for i in [0,1,2]};summary['sha256']=sha(ROOT/'data.npz')
    summary['loss_mask']='Only candidates of the same completed requested depth as the teacher choice; unknown and other-depth candidates excluded. Full candidate mask used for agreement.'
    write(ROOT/'data-summary.json',summary);print('prepared',summary,flush=True)
