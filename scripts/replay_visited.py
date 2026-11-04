"""Replay the frozen seven holdout contrasts; do not use for reselection."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from blokus.tournament import read_prefixes
from blokus.visited_eval import tournament
from blokus.ai import seed_all


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='artifacts/visited-diversity');p.add_argument('--output',required=True);a=p.parse_args()
    if Path(a.output).exists():raise ValueError('refusing to overwrite results')
    root=Path(a.root);specs=json.loads((root/'play-specs.json').read_text());fixture=read_prefixes(root/'holdout-prefixes.json');seed_all(601)
    tournament(specs,[(f'visited-{s}',o) for s in [601,602,603] for o in ['static','prior']]+[('prior','static')],fixture,a.output)
if __name__=='__main__':main()
