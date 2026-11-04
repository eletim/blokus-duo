import argparse, json, random, time, os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
from .ai import *

def train(a,model):
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    import sys, hashlib
    config=dict(arguments=vars(a),torch=torch.__version__,cuda=torch.version.cuda,python=sys.version,
                gpu=torch.cuda.get_device_name() if a.device.startswith('cuda') else None,
                source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')})
    (out/'config.json').write_text(json.dumps(config,indent=2))
    with (out/'config-history.jsonl').open('a') as f:f.write(json.dumps(config)+'\n')
    opt=torch.optim.Adam(model.parameters(),lr=a.lr)
    episode=0
    if a.resume:episode=load(a.resume,model,opt,True)['episode']
    else:save(out/'untrained.pt',model,opt,0,'untrained')
    begin=time.perf_counter(); start_episode=episode
    with open(out/'train.jsonl','a') as log:
        while episode-start_episode<a.games and time.perf_counter()-begin<a.seconds:
            s=State(); trajectory=[]; phase='imitation' if episode<a.imitation_games else 'selfplay'
            for group in opt.param_groups:group['lr']=a.lr if phase=='imitation' else getattr(a,'selfplay_lr',.0001)
            while not s.done:
                qs=candidates(s)
                if not qs:play(s,None,validate=False);continue
                x,st=features(s,qs); xt,stt=tensors(x,st,a.device)
                inference_start=time.perf_counter()
                logits,v=model(xt,stt)
                if phase=='imitation':
                    target=int(np.argmax(heuristic(x))); action=target if random.random()<0.9 else random.randrange(len(qs))
                else:
                    action=int(torch.distributions.Categorical(logits=logits.detach()).sample());target=action
                if a.device.startswith('cuda'):torch.cuda.synchronize()
                TIMES['network']+=time.perf_counter()-inference_start
                trajectory.append((x,st,action,target,s.turn))
                play(s,qs[action].move)
            scores=[s.score(p) for p in (0,1)]; losses=[]; opt.zero_grad()
            t=time.perf_counter()
            for x,st,action,target,p in trajectory:
                logits,v=model(*tensors(x,st,a.device)); reward=(scores[p]-scores[1-p])/109
                dist=torch.distributions.Categorical(logits=logits)
                if phase=='imitation':
                    teacher=torch.as_tensor(heuristic(x),device=a.device)
                    pl=-(teacher.softmax(0)*logits.log_softmax(0)).sum()
                    pl=pl+0.1*((logits-logits.mean())-(teacher-teacher.mean())).square().mean()
                else:pl=-dist.log_prob(torch.tensor(action,device=a.device))*(reward-v.detach())-0.01*dist.entropy()
                loss=pl+0.5*(v-reward).square()
                (loss/len(trajectory)).backward();losses.append(float(loss.detach()))
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
            if a.device.startswith('cuda'):torch.cuda.synchronize()
            update=time.perf_counter()-t;episode+=1
            record=dict(episode=episode,phase=phase,scores=scores,loss=float(np.mean(losses)),elapsed=time.perf_counter()-begin,update_seconds=update,timing=TIMES.copy())
            log.write(json.dumps(record)+'\n');log.flush()
            if episode%10==0:print(json.dumps(record),flush=True)
            save(out/'latest.pt',model,opt,episode,phase)
            if episode==a.imitation_games:save(out/'imitation.pt',model,opt,episode,phase)
        print(json.dumps(dict(completed=episode-start_episode,total_episode=episode,seconds=time.perf_counter()-begin,timing=TIMES)),flush=True)

def evaluate(a,model):
    results=[]
    for opponent in ('random','heuristic'):
        wins=losses=draws=0; diffs=[]; games=[]
        for i in range(a.games):
            rng=random.Random(a.seed+i//2);s=State();seat=i%2
            while not s.done:
                kind=a.agent if s.turn==seat else opponent
                m=select(s,kind,rng,model,a.device);play(s,m,validate=m is not None)
            d=s.score(seat)-s.score(1-seat);wins+=d>0;losses+=d<0;draws+=d==0;diffs.append(d)
            games.append(dict(seed=a.seed+i//2,seat=seat,scores=[s.score(0),s.score(1)],difference=d))
        results.append(dict(opponent=opponent,wins=wins,losses=losses,draws=draws,mean_difference=float(np.mean(diffs)),games=games))
    report=dict(agent=a.agent,checkpoint=a.checkpoint,seed=a.seed,results=results,timing=TIMES)
    print(json.dumps(report,indent=2))
    destination=Path(a.output);destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(report,indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['train','evaluate','play','gpu-check'])
    p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--checkpoint');p.add_argument('--resume');p.add_argument('--output')
    p.add_argument('--games',type=int,default=None);p.add_argument('--seconds',type=float,default=300)
    p.add_argument('--imitation-games',type=int,default=60);p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--selfplay-lr',type=float,default=.0001)
    p.add_argument('--agent',choices=['neural','random','heuristic'],default='neural')
    p.add_argument('--players',nargs=2,default=['human','neural'],choices=['human','neural','random','heuristic'])
    a=p.parse_args()
    if a.games is None:a.games=200 if a.command=='train' else 20
    if a.output is None:a.output='artifacts/evaluation.json' if a.command=='evaluate' else 'artifacts/run'
    seed_all(a.seed);model=Net().to(a.device)
    if a.checkpoint:load(a.checkpoint,model)
    if a.command=='train':train(a,model)
    elif a.command=='evaluate':
        if a.games<=0 or a.games%2:p.error('--games must be positive and even for alternating seats')
        evaluate(a,model)
    elif a.command=='gpu-check':
        x=torch.randn(4096,DIM,device=a.device);s=torch.randn(STATE_DIM,device=a.device)
        l,v=model(x,s);(l.square().mean()+v.square()).backward();torch.cuda.synchronize()
        print(json.dumps(dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),capability=torch.cuda.get_device_capability(),arch=torch.cuda.get_arch_list(),finite_grad=all(bool(torch.isfinite(z.grad).all()) for z in model.parameters()),allocated=torch.cuda.max_memory_allocated(),python=__import__('sys').version),indent=2))
    else:
        s=State();rng=random.Random(a.seed)
        while not s.done:
            print(s.render());kind=a.players[s.turn];print('turn',s.turn,kind)
            if kind=='human':
                qs=legal_moves(s)
                if not qs:m=None;print('Forced pass')
                else:
                    for i,q in enumerate(qs):print(i,q.move,'cells',q.move.cells)
                    try:m=qs[int(input('move index> '))].move
                    except (ValueError,IndexError):print('Invalid index');continue
            else:m=select(s,kind,rng,model,a.device)
            play(s,m,validate=m is not None)
        print(s.render());print('scores',s.score(0),s.score(1))
if __name__=='__main__':main()
