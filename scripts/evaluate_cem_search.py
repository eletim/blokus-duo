"""Predeclared A2-search ablation on the fixed new holdout; no reselection."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from blokus.ai import seed_all
from blokus.research import sha
from blokus.tournament import read_prefixes,tournament


def main():
    root=Path(sys.argv[1]);previous=Path('artifacts/research-round1')
    selection=json.loads((root/'selection.json').read_text());original=json.loads((previous/'selection.json').read_text())
    assert sha(selection['checkpoint'])==selection['sha256']
    prefixes=read_prefixes(root/'holdout-prefixes.json')
    assert prefixes['sha256']==selection['holdout_sha256']
    specs=dict(A2_search=selection['checkpoint']+':search',A2=selection['checkpoint'],
               C_search=original['selected']['C']['checkpoint']+':search',heuristic='heuristic',v1=str(previous/'v1.pt'))
    pairs=[('A2_search',opponent) for opponent in ('A2','C_search','heuristic','v1')]
    seed_all(101)
    tournament(specs,pairs,prefixes,root/'search-results.json',device='cuda')
    assert sha(selection['checkpoint'])==selection['sha256']
if __name__=='__main__':main()
