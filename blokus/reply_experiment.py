"""Cold-per-game 25ms reply-width ablation; selection precedes fresh holdout."""
import argparse,json,random,time
from pathlib import Path
from .ai import seed_all
from .engine import State,legal_moves,play
from .aligned_data import BASE
from .value_search import BudgetAgent
from .reply_search import ReplyAgent,reset_game
from .research import sha,source_hashes
from .tournament import make_prefixes,prefix_state,summarize,read_prefixes
from .visited_data import state_hash
from .value_experiment import write

def agent(width,device='cuda'):
    if width==3:return BudgetAgent('value',BASE,.5,24,25,device)
    return ReplyAgent('value',BASE,.5,24,25,device,reply_width=width)

def match(width,fixture):
    agents=[agent(width),agent(3)];games=[];start=time.perf_counter()
    for i,p in enumerate(fixture['prefixes']):
        for seat in [0,1]:
            for a in agents:reset_game(a)
            s=prefix_state(p);rngs=[random.Random(8899+i*2+j) for j in [0,1]];moves=[];hashes=[state_hash(s)];decisions=[]
            while not s.done:
                who=0 if s.turn==seat else 1;a=agents[who];nr=a.roots;nn=a.nodes;nf=a.fallbacks
                m=a.act(s,rngs[s.turn]);decisions.append(dict(who=who,ms=a.times[-1]*1000,completed_roots=a.roots-nr,nodes=a.nodes-nn,fallbacks=a.fallbacks-nf))
                if m is None:assert not legal_moves(s)
                play(s,m,validate=m is not None);moves.append([m.piece,m.orientation,m.row,m.col] if m else None);hashes.append(state_hash(s))
                assert len(moves)<=86
            games.append(dict(prefix_id=p['id'],seat=seat,scores=[s.score(0),s.score(1)],difference=s.score(seat)-s.score(1-seat),moves=moves,state_hashes=hashes,decisions=decisions))
    return dict(width=width,budget_ms=25,cache='cold every game; no TT; adaptive cost estimates reset both sides',resets_per_agent=len(games),checkpoint=BASE,checkpoint_sha256=sha(BASE),prefix_sha256=fixture['sha256'],games=games,summary=summarize(games),left=agents[0].stats(),right=agents[1].stats(),seconds=time.perf_counter()-start)

def run(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    if (root/'validation-results.json').exists():raise ValueError('existing results')
    seed_all(802);start=time.perf_counter();validation=make_prefixes(621000,32);write(root/'validation-prefixes.json',validation);results=[]
    for width in [6,12]:
        r=match(width,validation);results.append(r);write(root/'validation-results.json',results);print('VALIDATION',width,r['summary'],flush=True)
    best=max(results,key=lambda r:r['summary']['mean_difference']);write(root/'selection.json',dict(reply_width=best['width'],budget_ms=25,criterion='newvalidation mean point difference',validation_prefix_sha256=validation['sha256'],checkpoint_sha256=sha(BASE)))
    with open('PROGRESS.md','a') as f:f.write(f"\nReply width fixed to {best['width']} after newvalidation621000; fresh622000/96 begins.\n")
    hold=make_prefixes(622000,96);write(root/'holdout-prefixes.json',hold);r=match(best['width'],hold);write(root/'holdout-results.json',r)
    write(root/'run-summary.json',dict(seconds=time.perf_counter()-start,training_updates=0,source_hashes=source_hashes(),games=sum(len(x['games']) for x in results)+len(r['games'])));print('HOLDOUT',r['summary'],flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['run','play','tournament']);p.add_argument('--root',default='artifacts/reply-diagnostics');p.add_argument('--prefix-file');p.add_argument('--output',default='artifacts/reply-replay.json');args=p.parse_args()
    if args.command=='run':return run(args.root)
    width=json.loads((Path(args.root)/'selection.json').read_text())['reply_width']
    if args.command=='tournament':return write(args.output,match(width,read_prefixes(args.prefix_file)))
    a=agent(width);reset_game(a);s=State();rng=random.Random(0)
    while not s.done:
        print(s.render())
        if s.turn==0:
            qs=legal_moves(s);m=None
            if qs:
                for i,q in enumerate(qs):print(i,q.move)
                try:m=qs[int(input('index> '))].move
                except (ValueError,IndexError):continue
        else:m=a.act(s,rng)
        play(s,m,validate=m is not None)
    print('scores',s.score(0),s.score(1));print(a.stats())
if __name__=='__main__':main()
