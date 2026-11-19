import gzip,hashlib,json,subprocess,sys
from pathlib import Path
SCRIPT=Path(__file__).resolve().parents[1]/'scripts/materialize_async_bank.py'

def test_compressed_json_restores_canonical_sha(tmp_path):
 obj={'games':[{'rng':[1,2,3]}],'models':{}};raw=(json.dumps(obj,indent=2)+'\n').encode();archive=tmp_path/'bank.json.gz'
 with gzip.open(archive,'wt') as h:json.dump(obj,h)
 sha=hashlib.sha256(raw).hexdigest();cmd=[sys.executable,str(SCRIPT),str(archive),'--expected-sha',sha]
 subprocess.run(cmd,check=True,capture_output=True);assert archive.with_suffix('').read_bytes()==raw
 subprocess.run(cmd,check=True,capture_output=True)

def test_materialization_rejects_wrong_hash_and_existing_conflict(tmp_path):
 archive=tmp_path/'bank.json.gz'
 with gzip.open(archive,'wt') as h:json.dump({'games':[]},h)
 cmd=[sys.executable,str(SCRIPT),str(archive)]
 assert subprocess.run(cmd+['--expected-sha','wrong'],capture_output=True).returncode!=0
 out=archive.with_suffix('');assert not out.exists();out.write_text('keep me')
 assert subprocess.run(cmd,capture_output=True).returncode!=0
 assert out.read_text()=='keep me'


def test_original_compact_archive_hash_is_preserved(tmp_path):
 raw=b'{"games":[]}'
 archive=tmp_path/'old-bank.json.gz';archive.write_bytes(gzip.compress(raw))
 subprocess.run([sys.executable,str(SCRIPT),str(archive),'--expected-sha',hashlib.sha256(raw).hexdigest()],check=True,capture_output=True)
 assert archive.with_suffix('').read_bytes()==raw
