"""Additional held-out comparison audit (no regeneration or inference)."""
from scripts.komi4_scale import R,read,write,state_hash
from blokus.engine import State,Move,play

def states(d):
    s=State();keys=set()
    for ply,m in enumerate(d['moves']):
        if ply>=len(d['prefix']):keys.add(state_hash(s))
        play(s,Move(*m) if m is not None else None,validate=False)
    return keys

def run():
    selection=set()
    for p in (R/'games').glob('selection-*.json.gz'):selection.update(states(read(p)))
    heldout=set(read(R/'features/test.json')['keys'])
    hits={}
    for p in (R/'games').glob('*.json.gz'):
        if not p.name.startswith(('comparison-','standard-')):continue
        ks=states(read(p));a=len(ks&selection);b=len(ks&heldout)
        if a or b:hits[p.name]=dict(selection=a,test=b)
    write(R/'holdout-audit.json',dict(final_comparison_overlap=hits,selection_states=len(selection),test_states=len(heldout),note='Checks post-prefix nonterminal positions; intentional sharing within a paired final group is not cross-split leakage.'))
    assert not hits,hits
    print('Final comparison overlaps with selection and test: 0')
if __name__=='__main__':run()
