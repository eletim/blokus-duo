"""Separate CPU profile, never concurrent with the paired timing runs."""
import cProfile,pstats,json,gzip,random
import torch
import gpu_generation as g
from blokus.ppo_policy import PPOAgent
from blokus.ppo_search_deeper import DeepSearch

def main():
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 p=PPOAgent(g.e.SEEDS[0],'cpu',backend='native');a=DeepSearch(p,60000)
 game=json.load(gzip.open(next((g.R/'cpu-1').glob(f'{g.e.SEEDS[0]}-*.gz')),'rt'))
 profiler=cProfile.Profile();states=[g.e.state(game['moves'][:n]) for n in (8,16,24)]
 profiler.enable()
 for s in states:a.act(s,random.Random(51));assert a.stats['depth']==6
 profiler.disable();profiler.dump_stats(str(g.R/'cpu-profile.pstats'))
 with (g.R/'cpu-profile.txt').open('w') as out:pstats.Stats(profiler,stream=out).strip_dirs().sort_stats('cumulative').print_stats(55)
if __name__=='__main__':main()
