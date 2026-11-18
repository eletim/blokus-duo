"""Reconstruct verified gen3 training-state index from immutable Git artifacts.
Requires full repository history (including research/ppo-ragged-update).
Run from repository root. Output is an ephemeral ~881 MB numpy index.
"""
import gzip,hashlib,json,posixpath,subprocess,argparse,time
import numpy as np
from blokus.engine import State,Move,play
from blokus.ppo_runtime import state_hash
REF='828e8ea3883da8463dde4a55d7d853607337a84e'
def read(ref,path):return subprocess.check_output(['git','show',ref+':'+path])
def historical(ref,path,seen,parts):
 if path in seen:return
 seen.add(path);d=json.loads(read(ref,path));directory=posixpath.dirname(path)
 for entry in d.get('parts',[]):
  raw=read(ref,posixpath.normpath(posixpath.join(directory,entry['path'])));assert hashlib.sha256(raw).hexdigest()==entry['sha256'];parts.append(np.frombuffer(gzip.decompress(raw),dtype='S32'))
 for entry in d.get('bases',[]):
  p=posixpath.normpath(posixpath.join(directory,entry['path']));assert hashlib.sha256(read(ref,p)).hexdigest()==entry['sha256'];historical(ref,p,seen,parts)
def restore(output):
 start=time.time();root='artifacts/ppo-four-gpu';hashes=set();games=0
 paths=subprocess.check_output(['git','ls-tree','-r','--name-only',REF,root],text=True).splitlines();paths=[p for p in paths if '/seed-' in p and '/history-' in p and p.endswith('.jsonl.gz')]
 sources=[('pilot',read('69e4c7b','artifacts/ppo-four-channel-runtime/pilot/games.jsonl')),(root+'/optimization-pilot-history.jsonl.gz',None)]+[(p,None) for p in paths]
 for name,data in sources:
  if data is None:data=gzip.decompress(read(REF,name))
  for line in data.decode().splitlines():
   for g in json.loads(line)['games']:
    s=State();hashes.add(bytes.fromhex(state_hash(s)))
    for m in g['prefix']+g['moves']:
     play(s,Move(*m) if m else None,validate=False);hashes.add(bytes.fromhex(state_hash(s)))
    assert s.done and g['scores']==[s.score(0),s.score(1)];games+=1
  print('replayed',name,games,flush=True)
 parts=[np.array(sorted(hashes),dtype='S32')];del hashes
 historical('a011c78','artifacts/ppo-short-search/training-states/index.json',set(),parts);u=np.unique(np.concatenate(parts));del parts
 assert hashlib.sha256(u.tobytes()).hexdigest()==json.loads(read(REF,root+'/training-audit.json'))['union_sha256']
 for name in ('ppo-minibatch','ppo-lr64'):
  p='artifacts/'+name;d=json.loads(read(REF,p+'/training-audit.json'));chunks=[]
  for entry in d['delta_parts']:
   raw=read(REF,p+'/'+entry['path']);assert hashlib.sha256(raw).hexdigest()==entry['sha256'];chunks.append(raw)
  raw=b''.join(chunks);assert hashlib.sha256(raw).hexdigest()==d['delta_sha256'];u=np.unique(np.concatenate([u,np.frombuffer(gzip.decompress(raw),dtype='S32')]))
  assert len(u)==d['union_states'] and hashlib.sha256(u.tobytes()).hexdigest()==d['union_sha256']
 assert len(u)==27533566 and hashlib.sha256(u.tobytes()).hexdigest()=='cdd1f0f1ac7a0e722a670fb541ce904187949c6982cd03fda1fa7acad30f7b4c'
 np.save(output,u);print('RESTORED',len(u),time.time()-start,flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--output',default='/tmp/ppo-lr64-training.npy');restore(ap.parse_args().output)
