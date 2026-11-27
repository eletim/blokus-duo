"""Audit every saved evaluation stat without inference or source mutations."""
from collections import Counter
import json
from blokus.console import Sources, read_json, recorded_evaluation
s=Sources('/home/eletim/blokus-value-placement/artifacts','/home/eletim/.local/share/blokus-web/games.sqlite3')
c=Counter()
for stage,path in s.shards():
    data=read_json(path)
    bank=read_json(path.parent/(path.name.split('-')[0]+'-bank.json'))['prefixes']
    for game in data['games']:
        prefix=len(bank[game['prefix']]['moves']);c['games']+=1;c['prefix_positions_without_evaluation']+=prefix
        assert len(game['stats'])==len(game['moves'])
        for i,(st,move) in enumerate(zip(game['stats'],game['moves'])):
            player=(prefix+i)%2
            assert st['player']==player
            assert st['method']==(data['left'] if player==game['seat'] else data['right'])
            e=recorded_evaluation(st,player,s.model(st['method'],stage),move,trusted=True)
            c['recorded_stats']+=1
            if e['value'] is not None:
                assert e['value']==max(st['root_scores'])
                assert st['stages'][-1]==e['depth']
                c['selected_move_evaluations']+=1
            else:
                assert st['forced_pass'] and move is None
                c['forced_pass_without_evaluation']+=1
print(json.dumps(dict(c),indent=2))
