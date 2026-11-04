"""CLI for reproducible three-method experiments. No unattended infinite runs."""
import argparse
import itertools
import json
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import random
import shutil
import time
from pathlib import Path
from .ai import seed_all
from .research import Agent, sha, source_hashes
from .research_train import train
from .tournament import make_prefixes, read_prefixes, match, tournament
from .engine import State, play, legal_moves


def write_json(path, data):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(data, indent=2))


def run(a):
    root = Path(a.output)
    if root.exists() and any(root.iterdir()): raise ValueError('run output must be new/empty; use train/tournament to continue individual stages')
    root.mkdir(parents=True, exist_ok=True); begin = time.perf_counter()
    shutil.copy2(a.baseline, root/'v1.pt')
    validation = make_prefixes(41000, a.validation_prefixes); holdout = make_prefixes(51000, a.holdout_prefixes)
    assert not {p['id'] for p in validation['prefixes']} & {p['id'] for p in holdout['prefixes']}
    write_json(root/'validation-prefixes.json', validation); write_json(root/'holdout-prefixes.json', holdout)
    write_json(root/'baseline-config.json', dict(v1_source=a.baseline, v1_sha256=sha(root/'v1.pt'),
               heuristic='blokus.ai.heuristic fixed weights', random='uniform legal actions', source_hashes=source_hashes()))
    write_json(root/'run-config.json', dict(arguments=vars(a), source_hashes=source_hashes(), prefix_validation_sha256=validation['sha256'],
               prefix_holdout_sha256=holdout['sha256'], selection='mean paired score difference against heuristic and v1; midpoint/latest; tie favors midpoint',
               equal_thinking_time=False, training_limit_per_method_seconds=a.seconds))
    selected = {}; validation_results = []; training = {}
    for method, seed in [('A',101),('B',102),('C',103)]:
        print('TRAIN', method, flush=True)
        training[method] = train(method, root/method, root/'v1.pt', seconds=a.seconds, seed=seed, device=a.device)
        choices = []
        for stage in ('midpoint','latest'):
            scores = []
            for opponent in ('heuristic',str(root/'v1.pt')):
                result = match(str(root/method/(stage+'.pt')), opponent, validation, a.device)
                result.update(method=method, stage=stage); validation_results.append(result)
                scores.append(result['summary']['mean_difference'])
                write_json(root/'validation-results.json', validation_results)
            choices.append(dict(stage=stage, criterion=sum(scores)/len(scores)))
        best = max(choices, key=lambda c:c['criterion'])
        destination = root/method/'selected.pt'; shutil.copy2(root/method/(best['stage']+'.pt'), destination)
        selected[method] = dict(**best, choices=choices, checkpoint=str(destination), sha256=sha(destination))
        print('SELECT', method, selected[method], flush=True)
    # Immutable selection written before the first holdout match is played.
    selection = dict(validation_sha256=validation['sha256'], holdout_sha256=holdout['sha256'], selected=selected, source_hashes=source_hashes())
    write_json(root/'selection.json', selection)
    specs = {k:v['checkpoint'] for k,v in selected.items()}
    specs.update(heuristic='heuristic', random='random', v1=str(root/'v1.pt'), C_search=specs['C']+':search')
    for method in ('A','B','C'): specs[method+'_untrained'] = str(root/method/'untrained.pt')
    pairs = [(m,o) for m in ('A','B','C') for o in ('heuristic','v1','random')]
    pairs += list(itertools.combinations(('A','B','C'),2))
    pairs += [(m,m+'_untrained') for m in ('A','B','C')]
    pairs += [('C_search','C'),('C_search','B'),('C_search','heuristic'),('C_search','v1')]
    pairs += [('heuristic','random'),('v1','heuristic')]
    report = tournament(specs, pairs, holdout, root/'holdout-results.json', a.device)
    write_json(root/'run-summary.json', dict(wall_seconds=time.perf_counter()-begin, training=training, selection_sha256=sha(root/'selection.json'),
               holdout_sha256=sha(root/'holdout-results.json'), matches=len(report['results']), completed=True))
    print('COMPLETE', root, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['run','train','tournament','play','prefixes'])
    p.add_argument('--output', default='artifacts/research-round1')
    p.add_argument('--device', default='cuda')
    p.add_argument('--method', choices=['A','B','C'], default='A')
    p.add_argument('--baseline', default='artifacts/best-trained.pt')
    p.add_argument('--seconds', type=float, default=180)
    p.add_argument('--games', type=int, default=100000)
    p.add_argument('--seed', type=int, default=101)
    p.add_argument('--resume')
    p.add_argument('--lr', type=float, default=.0005)
    p.add_argument('--validation-prefixes', type=int, default=12)
    p.add_argument('--holdout-prefixes', type=int, default=24)
    p.add_argument('--prefix-file')
    p.add_argument('--agents', nargs='+', default=['heuristic','random'])
    a = p.parse_args(); seed_all(a.seed)
    if a.seconds <= 0 or a.games <= 0: p.error('positive seconds/games required')
    if a.validation_prefixes <= 0 or a.holdout_prefixes <= 0: p.error('positive prefix counts required')
    if a.command == 'run': run(a)
    elif a.command == 'train': train(a.method, a.output, a.baseline, a.seconds, a.games, a.seed, a.device, a.resume, a.lr)
    elif a.command == 'prefixes': write_json(a.output, make_prefixes(a.seed, a.holdout_prefixes))
    elif a.command == 'tournament':
        if not a.prefix_file: p.error('--prefix-file required')
        specs = {str(i):s for i,s in enumerate(a.agents)}
        tournament(specs, list(itertools.combinations(specs,2)), read_prefixes(a.prefix_file), a.output, a.device)
    else:
        if len(a.agents) != 2: p.error('play needs exactly two --agents')
        agents = [Agent(s,a.device) for s in a.agents]; s = State(); rng = random.Random(a.seed)
        while not s.done:
            print(s.render()); print('turn',s.turn,a.agents[s.turn])
            m = agents[s.turn].act(s,rng)
            if m is None:
                if legal_moves(s): raise AssertionError('illegal pass')
                print('forced pass'); play(s,None,validate=False)
            else: play(s,m)
        print(s.render()); print('scores',s.score(0),s.score(1))
if __name__ == '__main__': main()
