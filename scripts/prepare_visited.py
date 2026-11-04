import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from blokus.ai import seed_all
from blokus.aligned_data import BASE
from blokus.visited_data import gather,label,stratified_subset
from blokus.visited_eval import tournament
from blokus.tournament import make_prefixes
from blokus.value_experiment import write


def main():
    root=Path('artifacts/visited-diversity');root.mkdir(parents=True,exist_ok=True)
    if (root/'prepare-config.json').exists():raise ValueError('existing preparation')
    begin=time.perf_counter();seed_all(601)
    write(root/'prepare-config.json',dict(curve_seed=400000,curve_prefixes=8,budgets=[6,25,100],wide_roots=512,wide_repeats=4,dense_roots=128,dense_repeats=16,dev_roots=256,dev_repeats=8,source_seeds=[420000,430000]))
    curve=make_prefixes(400000,8);write(root/'curve-prefixes.json',curve)
    specs={str(ms):dict(kind='value',checkpoint=BASE,alpha=.5,gate=24,budget_ms=ms) for ms in [6,25,100]}
    tournament(specs,[('25','6'),('100','6'),('100','25')],curve,root/'curve-results.json')
    wide=gather(root/'pilot/visited.json',512,420000);dev=gather(root/'dev-visited.json',256,430000)
    metas={}
    metas['dev']=label(root/'dev.npz',dev['records'],8)
    metas['wide']=label(root/'pilot/wide.npz',wide['records'],4)
    metas['dense']=label(root/'pilot/dense.npz',stratified_subset(wide['records'],128),16)
    write(root/'prepare-summary.json',dict(seconds=time.perf_counter()-begin,labels=metas));print('PREPARED',flush=True)
if __name__=='__main__':main()
