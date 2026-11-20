"""CPU human/AI play with the fixed gen4 policy, value and search settings."""
import argparse,json,random
import torch
from .engine import State,play
from .ppo_play import human_move,HeuristicAgent
from .gen4 import load,search,verify

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-seed',type=int,default=1751401)
    p.add_argument('--backend',choices=['auto','native','python'],default='auto')
    p.add_argument('--human-seat',type=int,choices=[0,1])
    p.add_argument('--verify',action='store_true')
    a=p.parse_args()
    if a.verify: print(json.dumps(verify(),indent=2));return
    torch.set_num_threads(1);bot=search(load(a.model_seed,a.backend));other=HeuristicAgent();s=State();rng=random.Random(1);moves=[]
    try:
        while not s.done:
            if s.turn==a.human_seat: print(s.render());m=human_move(s)
            elif a.human_seat is not None or s.turn==0: m=bot.act(s,rng)
            else: m=other.act(s,rng)
            play(s,m,validate=True);moves.append(None if m is None else [m.piece,m.orientation,m.row,m.col])
    except (EOFError,KeyboardInterrupt): print('対局を終了しました');return
    print(json.dumps(dict(generation='gen4',model_seed=a.model_seed,epochs=5,moves=moves,scores=[s.score(0),s.score(1)])))
if __name__=='__main__':main()
