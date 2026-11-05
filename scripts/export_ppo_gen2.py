"""Export the frozen selection, never choose a seed/checkpoint from confirmation."""
import sys,json
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from blokus.research import sha
out=Path('artifacts/ppo-gen2');out.mkdir(exist_ok=True);manifest={}
selection=json.loads(Path('artifacts/ppo-generation2/lock.json').read_text())['models']
for seed,c in selection.items():
 path=Path(c['path']);assert sha(path)==c['sha256'];ck=torch.load(path,map_location='cpu',weights_only=False);target=out/f'seed-{seed}.pt'
 data=dict(format='terminal-ppo-inference-v1',model=ck['model'],model_seed=int(seed),iteration=ck['iteration'],generation='gen2',source_checkpoint_sha256=sha(path),source_commit='fd527ca',architecture='TerminalNet all-legal exact-cell policy',reward='terminal win/draw/loss; no shaping')
 torch.save(data,target);manifest[seed]=dict(path=str(target),sha256=sha(target),source_sha256=sha(path),iteration=ck['iteration'],bytes=target.stat().st_size)
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
