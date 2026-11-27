"""Fixed gen4-search selfplay: no greedy teacher substitution."""
from future_common import *
from concurrent.futures import ProcessPoolExecutor
import multiprocessing,argparse
AGENTS={}
def game(job):
 phase,gid,seed=job;torch.set_num_threads(1)
 path=R/'games'/phase/f'{gid:05d}.json'
 if path.exists():return json.loads(path.read_text())
 if seed not in AGENTS:
  p,v=load(seed,'native');AGENTS[seed]=DeepSearch(p,500,width=4,max_depth=6,value_evaluator=v)
 agent=AGENTS[seed];base={'pilot':924000000,'train':924100000,'dev':924200000,'test':924300000}[phase]
 rng=random.Random(base+gid);pr=prefix(rng,8+2*(gid%3));s=state(pr);ms=list(pr);stats=[];start=time.perf_counter()
 while not s.done:
  m=agent.act(s,rng);stats.append(dict(agent.stats,player=s.turn));play(s,m,validate=True);ms.append(raw(m))
 d=dict(phase=phase,id=gid,teacher_seed=seed,rng_seed=base+gid,prefix_length=len(pr),moves=ms,stats=stats,boards=s.boards,scores=[s.score(0),s.score(1)],seconds=time.perf_counter()-start,teacher='Frozen gen4 search CPU500ms width4 max6')
 save(path,d);return d

def generate(phase,n):
 start=time.perf_counter();jobs=[(phase,i,SEEDS[i%3]) for i in range(n)]
 with ProcessPoolExecutor(4,mp_context=multiprocessing.get_context('spawn')) as pool:
  for i,g in enumerate(pool.map(game,jobs)):
   if i%16==0:print('GENERATE',phase,i+1,n,round(time.perf_counter()-start,1),flush=True)
 save(R/f'{phase}-generation.json',dict(games=n,seconds=time.perf_counter()-start,workers=4,settings=dict(budget_ms=500,width=4,max_depth=6)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase');p.add_argument('n',type=int);a=p.parse_args();generate(a.phase,a.n)
