"""Restore bounded raw archive parts and verify all member content hashes."""
from pathlib import Path
import hashlib,json,tarfile,tempfile,shutil
R=Path('artifacts/future-occupancy')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 m=json.loads((R/'archive-manifest.json').read_text())
 with tempfile.TemporaryDirectory() as tmp:
  p=Path(tmp)/'raw.tar.gz'
  with p.open('wb') as out:
   for part in m['parts']:
    q=R/part['name'];assert sha(q)==part['sha256'];
    with q.open('rb') as f:shutil.copyfileobj(f,out)
  assert sha(p)==m['archive_sha256']
  with tarfile.open(p,'r:gz') as t:t.extractall(R,filter='data')
 for name,record in m['members'].items():assert sha(R/name)==record['sha256'],name
 print('RESTORE PASSED',len(m['members']))
if __name__=='__main__':main()
