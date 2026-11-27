"""Finite, resumable komi-4 experiment. No production model or database writes.
Stages: prepare, bootstrap, generate, adapt, evaluate, summarize.
Run from repository root with PYTHONPATH=. and the training virtual environment.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,copy,hashlib,json,random,time
from pathlib import Path
import numpy as np
import torch
from blokus.engine import State,Move,play,legal_moves,SIZES
from blokus.gen4 import load,search
from blokus.margin_value import MarginValue
from blokus.margin_search import MarginSearch,key
from blokus.margin_network import MarginNet,NetworkEvaluator,inputs,terminal_margin

R=Path('artifacts/komi4-campaign'); OLD=Path('artifacts/margin-pilot'); R.mkdir(exist_ok=True)
torch.set_num_threads(1)
SEEDS=[1751401,1751402,1751403]

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    p=Path(p); t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2));t.replace(p)
def read(p):return json.loads(Path(p).read_text())
def event(stage,**kw):
    row=dict(at=time.time(),stage=stage,komi=4,**kw)
    with (R/'operations.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    write(R/'status.json',row);print(json.dumps(row),flush=True)
def save_torch(p,x):
    p=Path(p);tmp=p.with_suffix('.tmp');torch.save(x,tmp);tmp.replace(p)
def move(m):return None if m is None else [m.piece,m.orientation,m.row,m.col]
def replay(moves):
    s=State()
    for m in moves:play(s,Move(*m) if m is not None else None,validate=True)
    return s

def prepare():
    if (R/'plan.json').exists():
        assert read(R/'plan.json')['source_sha256']==digest(OLD/'split-records.json')
        event('prepared-existing');return
    tasks=[];rng=random.Random(42320261127);seen=set()
    # Entire random prefix belongs to exactly one split. No results used for selection.
    for split,n in [('train',48),('validation',12),('test',12),('comparison',8)]:
        for i in range(n):
            while True:
                s=State();prefix=[]
                for _ in range(8):
                    qs=legal_moves(s);m=rng.choice(qs).move if qs else None
                    play(s,m,validate=False);prefix.append(move(m))
                k=key(s)
                if k not in seen:seen.add(k);break
            group=f'{split}-{i:03}'
            if split=='comparison':
                for opp in ['ridge','gen4']:
                    for seat in [0,1]:tasks.append(dict(id=f'{group}-{opp}-{seat}',split=split,group=group,prefix=prefix,seed=SEEDS[i%3],seat=seat,opponent=opp,rng_seed=4230000+i))
            else:
                opp=['self','self','gen4','ridge'][i%4]
                for seat in [0,1]:tasks.append(dict(id=f'{group}-{seat}',split=split,group=group,prefix=prefix,seed=SEEDS[i%3],seat=seat,opponent=opp,rng_seed=4231000+len(tasks)))
    for seed in SEEDS:
        for opp in ['ridge','gen4']:
            for seat in [0,1]:tasks.append(dict(id=f'standard-{seed}-{opp}-{seat}',split='comparison',group=f'standard-{seed}',prefix=[],seed=seed,seat=seat,opponent=opp,rng_seed=4232000+seed))
    write(R/'tasks.json',tasks)
    plan=dict(version=1,created_at=time.time(),komi=4,rule='score0-score1-4; opposite for player1; bonuses included',source_sha256=digest(OLD/'split-records.json'),ridge_sha256=digest(OLD/'model.json'),tasks_sha256=digest(R/'tasks.json'),seeds=SEEDS,max_games=188,max_generations=1,epochs_per_stage=8,compute_cap_seconds=3600,gpu_cap_seconds=1200,selection='minimum fresh validation per-game MAE among gen0 and gen1 epochs, separately per seed; final seed chosen by validation only',split='all games with a prefix and seat pair stay together; conflicts with earlier split cause entire later group exclusion from prediction training/testing; final comparison conditions never train',bootstrap='old train only; old validation used for gen0; old test seen previously and never presented as unknown',generation=dict(budget_ms=100,max_depth=2,width=4),comparison=dict(budget_ms=500,max_depth=6,width=4),randomization='8 uniformly random legal placements, then deterministic top policy candidates; standard games have no opening randomization',stop='fixed 1 update, 188 games, 8 epochs/stage/seed, nonfinite or compute cap; no test-driven tuning',improvement='report paired MAE vs ridge and gen0 separately from WDL/margin vs ridge and gen4; no strength claim from MAE alone')
    write(R/'plan.json',plan);event('prepared',games_planned=len(tasks))

def dataset(rows,name,forbidden=None):
    path=R/(name+'.npz');meta=R/(name+'-meta.json')
    if path.exists():return np.load(path),read(meta)
    forbidden=forbidden or set();group_states={};games=[];traces=set();duplicates=0
    for idx,r in enumerate(rows):
        trace=json.dumps(r['moves'])
        if name.startswith('fresh-') and trace in traces:duplicates+=1;continue
        traces.add(trace)
        s=State();samples=[];group=r.get('group',r.get('component'))
        for ply,m in enumerate(r['moves']):
            if ply>=len(r['prefix']) and not s.done:
                k=key(s);group_states.setdefault(group,set()).add(k)
                # Even plies only for bootstrap; fresh games use every ply.
                if name!='bootstrap' or ply%2==0:samples.append((copy.deepcopy(s),ply))
            play(s,Move(*m) if m is not None else None,validate=False)
        games.append((r,samples,group))
    bad={g for g,ks in group_states.items() if ks&forbidden}
    b=[];st=[];ex=[];y=[];w=[];gs=[];stage=[];ridge=[];states=set()
    rv=MarginValue(OLD/'model.json',4)
    for r,samples,g in games:
        if g in bad:continue
        states.update(group_states[g])
        for s,ply in samples:
            a,c,e=inputs(s);b.append(a);st.append(c);ex.append(e)
            d=r['scores'][0]-r['scores'][1]-4;y.append(d if s.turn==0 else -d)
            w.append(1/len(samples));gs.append(g);stage.append('early' if ply<16 else 'middle' if ply<30 else 'late');ridge.append(rv(s))
    if not y:raise RuntimeError('empty dataset '+name)
    tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,b=np.array(b,dtype=np.uint8),st=np.array(st,np.float32),ex=np.array(ex,np.float32),y=np.array(y,np.float32),w=np.array(w,np.float32),group=np.array(gs),stage=np.array(stage),ridge=np.array(ridge));tmp.replace(path)
    sm=dict(duplicate_traces_removed=duplicates,positions=len(y),games=sum(g not in bad for r,s,g in games),groups=len(group_states)-len(bad),excluded_groups=sorted(bad),states=[repr(k) for k in states]);write(meta,sm)
    event('encoded',dataset=name,positions=len(y),games=sm['games'],excluded=len(bad))
    return np.load(path),sm

def keys_meta(meta):
    import ast
    return {ast.literal_eval(k) for k in meta['states']}

def metrics(y,p,w):
    e=p-y;return dict(mae=float(np.average(abs(e),weights=w)),bias=float(np.average(e,weights=w)),rmse=float(np.sqrt(np.average(e*e,weights=w))))
def predict(model,data,device):
    out=[];model.eval()
    with torch.inference_mode():
        for i in range(0,len(data['y']),512):out.append(model(*[torch.as_tensor(data[k][i:i+512],device=device) for k in ['b','st','ex']]).cpu().numpy())
    return np.concatenate(out)

def train_one(seed,stage,tr,va,base=None):
    result=R/f'{stage}-{seed}-result.json'
    if result.exists():
        saved=read(result);assert digest(saved['checkpoint'])==saved['sha256'];return saved
    device='cuda';torch.manual_seed(seed);np.random.seed(seed)
    model=MarginNet(load(seed,'native')[0].model).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=.001 if stage=='gen0' else .0003,weight_decay=.001)
    last=R/f'{stage}-{seed}-last.pt';best=R/f'{stage}-{seed}.pt';history=[];epoch0=0
    rng=np.random.default_rng(seed)
    if base:
        model.load_state_dict(torch.load(base,map_location=device,weights_only=False)['model'])
    if last.exists():
        ck=torch.load(last,map_location=device,weights_only=False);model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);epoch0=ck['epoch']+1;history=ck['history'];rng.bit_generator.state=ck['numpy_rng'];torch.set_rng_state(ck['torch_rng'].cpu());torch.cuda.set_rng_state(ck['cuda_rng'].cpu())
    best_mae=min([h['validation']['mae'] for h in history],default=float('inf'))
    if base and not best.exists():
        m=metrics(va['y'],predict(model,va,device),va['w']);best_mae=m['mae'];history.append(dict(epoch=-1,validation=m));save_torch(best,dict(model={k:v.cpu() for k,v in model.state_dict().items()},komi=4,seed=seed,generation=stage,epoch=-1))
    start=time.monotonic()
    # Weighted sampling gives each game equal influence. For gen1, old/new mass is 50/50.
    probability=tr['w'].astype(float);probability/=sum(probability)
    for ep in range(epoch0,8):
        enforce_cap();model.train();losses=[]
        ix=rng.choice(len(tr['y']),size=len(tr['y']),replace=True,p=probability)
        for i in range(0,len(ix),256):
            j=ix[i:i+256];args=[torch.as_tensor(tr[k][j],device=device) for k in ['b','st','ex']];y=torch.as_tensor(tr['y'][j],device=device)
            opt.zero_grad();p=model(*args);loss=torch.nn.functional.smooth_l1_loss(p/32,y/32)
            if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();losses.append(loss.item())
        m=metrics(va['y'],predict(model,va,device),va['w']);history.append(dict(epoch=ep,loss=float(np.mean(losses)),validation=m))
        ck=dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},komi=4,seed=seed,generation=stage,epoch=ep,optimizer=opt.state_dict(),history=history,numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state())
        save_torch(last,ck)
        if m['mae']<best_mae:best_mae=m['mae'];save_torch(best,{k:ck[k] for k in ['model','komi','seed','generation','epoch']})
        event('training',generation=stage,seed=seed,epoch=ep+1,epochs=8,validation=m,seconds=time.monotonic()-start)
    result_data=dict(seed=seed,generation=stage,history=history,selected_mae=best_mae,checkpoint=str(best),sha256=digest(best),seconds=time.monotonic()-start)
    write(result,result_data);return result_data

def bootstrap():
    rows=read(OLD/'split-records.json')
    tr,tm=dataset([r for r in rows if r['split']=='train'],'bootstrap')
    va,vm=dataset([r for r in rows if r['split']=='validation'],'old-validation',keys_meta(tm))
    for seed in SEEDS:train_one(seed,'gen0',tr,va)
    event('bootstrap-complete',generation=0)

def enforce_cap():
    events=[json.loads(x) for x in (R/'operations.jsonl').read_text().splitlines()]
    starts=[e['at'] for e in events if e['stage']=='compute-start']
    if starts and time.time()-starts[0]>3600:raise RuntimeError('compute wall-time cap reached')
    seconds=sum(read(p)['seconds'] for p in R.glob('gen*-result.json'))
    if seconds>1200:raise RuntimeError('training time cap reached')

class Gen4Rule:
    """Frozen gen4 heuristic, exact terminal WDL under research komi4."""
    def __init__(self,value):self.value=value
    def __call__(self,s):
        if s.done:
            d=terminal_margin(s);return float((d>0)-(d<0))
        return self.value(s)

def generate(final=False):
    out=R/'games';out.mkdir(exist_ok=True);pairs={};values={}
    frozen=read(R/'selection.json') if final else None
    for t in read(R/'tasks.json'):
        if (t['split']=='comparison')!=final:continue
        enforce_cap();path=out/(t['id']+'.json')
        model=R/f"{'gen1' if final else 'gen0'}-{t['seed']}.pt"
        mh=digest(model)
        if final:assert mh==frozen['models'][str(t['seed'])]['sha256']
        if path.exists():
            old=read(path);assert old['task']==t and old['model_sha256']==mh;continue
        if t['seed'] not in pairs:pairs[t['seed']]=load(t['seed'],'native')
        pair=pairs[t['seed']]
        if str(model) not in values:values[str(model)]=NetworkEvaluator(model)
        value=values[str(model)];rv=MarginValue(OLD/'model.json',4)
        budget,depth=(500,6) if final else (100,2)
        candidate=MarginSearch(pair[0],budget,4,depth,value_evaluator=value)
        other_value=value if t['opponent']=='self' else rv if t['opponent']=='ridge' else Gen4Rule(pair[1])
        other=MarginSearch(pair[0],budget,4,depth,value_evaluator=other_value)
        s=replay(t['prefix']);moves=list(t['prefix']);stats=[];predictions=[];rngs=[random.Random(t['rng_seed']+p) for p in [0,1]]
        value(s);other_value(s)
        while not s.done:
            p=s.turn;agent=candidate if p==t['seat'] else other
            predictions.append(dict(ply=len(moves),player=p,adjusted=value(s),ridge=rv(s),generation=1 if final else 0,komi=4))
            m=agent.act(s,rngs[p if t['seat']==0 else 1-p]);stats.append(dict(player=p,agent='nonlinear' if agent is candidate else t['opponent'],**copy.deepcopy(agent.stats)))
            play(s,m,validate=True);moves.append(move(m))
        scores=[s.score(p) for p in [0,1]];remaining=[sum(SIZES[i] for i in range(21) if s.remaining[p]>>i&1) for p in [0,1]]
        result=dict(task=t,model_sha256=mh,moves=moves,stats=stats,predictions=predictions,scores=scores,remaining=remaining,adjusted_first_margin=scores[0]-scores[1]-4,trace_sha256=hashlib.sha256(json.dumps(moves).encode()).hexdigest())
        write(path,result);event('comparison' if final else 'generation',completed=len(list(out.glob('*.json'))),planned=188,last_game=t['id'],scores=scores,adjusted_first_margin=result['adjusted_first_margin'])

def fresh_rows(split):
    rows=[]
    for p in sorted((R/'games').glob('*.json')):
        g=read(p);t=g['task']
        if t['split']==split:rows.append(dict(moves=g['moves'],scores=g['scores'],prefix=t['prefix'],group=t['group']))
    return rows

def adapt():
    old=np.load(R/'bootstrap.npz');forbidden=keys_meta(read(R/'bootstrap-meta.json'))|keys_meta(read(R/'old-validation-meta.json'))
    tr,tm=dataset(fresh_rows('train'),'fresh-train',forbidden)
    va,vm=dataset(fresh_rows('validation'),'fresh-validation',forbidden|keys_meta(tm))
    # Group exclusions are determined solely by position identity, before labels are used.
    te,tem=dataset(fresh_rows('test'),'fresh-test',forbidden|keys_meta(tm)|keys_meta(vm))
    merged={k:np.concatenate([old[k],tr[k]]) for k in ['b','st','ex','y']}
    merged['w']=np.concatenate([old['w']/sum(old['w'])*.5,tr['w']/sum(tr['w'])*.5])
    models={}
    for seed in SEEDS:models[str(seed)]=train_one(seed,'gen1',merged,va,R/f'gen0-{seed}.pt')
    selection=dict(models=models,selected_seed=int(min(models,key=lambda s:models[s]['selected_mae'])),criterion='fresh validation per-game MAE only',frozen_at=time.time(),test_inspected=False)
    if (R/'selection.json').exists():assert read(R/'selection.json')['models']==models
    else:write(R/'selection.json',selection)
    event('selection-frozen',selected_seed=selection['selected_seed'],excluded_test_groups=tem['excluded_groups'])

def evaluate():
    if (R/'prediction-test.json').exists():return
    sel=read(R/'selection.json');data=np.load(R/'fresh-test.npz');out={}
    for generation in ['gen0','gen1']:
        for seed in SEEDS:
            model=NetworkEvaluator(R/f'{generation}-{seed}.pt').model
            p=predict(model,data,'cpu');out[f'{generation}-{seed}']=evaluation_details(data,p)
    out['ridge']=evaluation_details(data,data['ridge'])
    write(R/'prediction-test.json',dict(models=out,selected_seed=sel['selected_seed'],test_kind='new fixed random-opening groups, gen0 policy outcomes; not optimal-game ground truth',komi=4));event('prediction-test-complete')

def evaluation_details(data,p):
    y,w=data['y'],data['w'];result=metrics(y,p,w)
    result['stages']={str(stage):metrics(y[data['stage']==stage],p[data['stage']==stage],w[data['stage']==stage]) for stage in np.unique(data['stage'])}
    result['groups']=len(np.unique(data['group']));result['positions']=len(y)
    a=[]
    for g in np.unique(data['group']):
        j=data['group']==g;a.append([sum(abs(p[j]-y[j])*w[j]),sum((p[j]-y[j])*w[j]),sum(w[j])])
    a=np.array(a);rng=np.random.default_rng(423);v=a[rng.integers(len(a),size=(2000,len(a)))].sum(1)
    result['mae_ci95']=np.quantile(v[:,0]/v[:,2],[.025,.975]).tolist();result['bias_ci95']=np.quantile(v[:,1]/v[:,2],[.025,.975]).tolist()
    diff=[]
    for g in np.unique(data['group']):
        j=data['group']==g;diff.append(np.average(abs(p[j]-y[j])-abs(data['ridge'][j]-y[j]),weights=w[j]))
    d=np.array(diff);b=d[rng.integers(len(d),size=(2000,len(d)))].mean(1);result['mae_minus_ridge']=float(d.mean());result['paired_mae_delta_ci95']=np.quantile(b,[.025,.975]).tolist()
    return result

def summarize():
    games=[read(p) for p in sorted((R/'games').glob('*.json'))];final=[g for g in games if g['task']['split']=='comparison']
    def summary(rows):
        d=np.array([g['adjusted_first_margin']*(1 if g['task']['seat']==0 else -1) for g in rows]);points=(d>0)+.5*(d==0)
        groups=sorted({g['task']['group'] for g in rows});a=[]
        for group in groups:
            ix=np.array([g['task']['group']==group for g in rows]);a.append([sum(points[ix]),sum(d[ix]),sum(ix)])
        a=np.array(a);rng=np.random.default_rng(423);z=a[rng.integers(len(a),size=(2000,len(a)))].sum(1)
        return dict(games=len(rows),groups=len(groups),wins=int(sum(d>0)),draws=int(sum(d==0)),losses=int(sum(d<0)),win_rate=float(np.mean(d>0)),point_rate=float(points.mean()),mean_margin=float(d.mean()),median_margin=float(np.median(d)),point_rate_ci95=np.quantile(z[:,0]/z[:,2],[.025,.975]).tolist(),mean_margin_ci95=np.quantile(z[:,1]/z[:,2],[.025,.975]).tolist(),unique_traces=len({g['trace_sha256'] for g in rows}))
    results={}
    for opp in ['ridge','gen4']:
        rows=[g for g in final if g['task']['opponent']==opp]
        results[opp]=dict(all=summary(rows),random_opening=summary([g for g in rows if g['task']['prefix']]),standard=summary([g for g in rows if not g['task']['prefix']]),seats={str(p):summary([g for g in rows if g['task']['seat']==p]) for p in [0,1]},seeds={str(s):summary([g for g in rows if g['task']['seed']==s]) for s in SEEDS})
    timing={}
    for who in ['nonlinear','ridge','gen4']:
        stats=[s for g in final for s in g['stats'] if s['agent']==who];times=np.array([s['elapsed_ms'] for s in stats]);depths=[s['depth'] for s in stats]
        timing[who]=dict(moves=len(stats),mean_ms=float(times.mean()),p95_ms=float(np.quantile(times,.95)),max_ms=float(max(times)),depth_counts={str(d):depths.count(d) for d in sorted(set(depths))},fallback=sum(s['fallback'] and not s['forced_pass'] for s in stats))
    # Prospective root predictions on final candidate-policy games, separate from gen0-policy test.
    prospective={}
    for kind in ['nonlinear','ridge']:
        y=[];pred=[];weights=[];groups=[];stages=[];ridge=[]
        for g in final:
            n=len(g['predictions'])
            for e in g['predictions']:
                y.append(g['adjusted_first_margin']*(1 if e['player']==0 else -1));pred.append(e['adjusted'] if kind=='nonlinear' else e['ridge']);weights.append(1/n);groups.append(g['task']['group']);ridge.append(e['ridge']);stages.append('early' if e['ply']<16 else 'middle' if e['ply']<30 else 'late')
        data=dict(y=np.array(y),w=np.array(weights),group=np.array(groups),stage=np.array(stages),ridge=np.array(ridge))
        prospective[kind]=evaluation_details(data,np.array(pred))
    write(R/'results.json',dict(komi=4,completed_games=len(games),generation=1,prediction=read(R/'prediction-test.json'),comparison=results,timing=timing,prospective_predictions=prospective,selection=read(R/'selection.json'),limits='small randomized-opening sample; only three standard trajectories per opponent/seat; gen4 nonterminal heuristic remains komi0-trained WDL, terminal WDL corrected to komi4',plan_sha256=digest(R/'plan.json')))
    event('complete',completed=len(games),planned=188,generation=1)

if __name__=='__main__':
    import fcntl
    campaign_lock=(R/'.run.lock').open('w')
    fcntl.flock(campaign_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','bootstrap','generate','adapt','evaluate','compare','summarize']);a=p.parse_args()
    if a.stage=='prepare':prepare()
    else:
        assert digest(R/'tasks.json')==read(R/'plan.json')['tasks_sha256']
        if not any(json.loads(x)['stage']=='compute-start' for x in (R/'operations.jsonl').read_text().splitlines()):event('compute-start')
        {'bootstrap':bootstrap,'generate':generate,'adapt':adapt,'evaluate':evaluate,'compare':lambda:generate(True),'summarize':summarize}[a.stage]()
