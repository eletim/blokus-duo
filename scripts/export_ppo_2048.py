"""Export only weights and provenance; optimizer/RNG/replays deliberately excluded."""
import sys,json
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from blokus.research import sha
out=Path('artifacts/ppo-2048');out.mkdir(exist_ok=True);manifest={}
for seed in (1401,1402,1403):
 path=Path(f'artifacts/ppo-learning-curve/seed-{seed}/iteration-2048.pt');c=torch.load(path,map_location='cpu',weights_only=False);target=out/f'seed-{seed}.pt'
 data=dict(format='terminal-ppo-inference-v1',model=c['model'],model_seed=seed,iteration=c['iteration'],source_checkpoint_sha256=sha(path),source_commit='df7bdf8',architecture='TerminalNet all-legal exact-cell policy',reward='terminal win/draw/loss; no shaping')
 torch.save(data,target);manifest[str(seed)]=dict(path=str(target),sha256=sha(target),source_sha256=sha(path),bytes=target.stat().st_size)
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
