"""Build a minimal inference candidate only after the predeclared L/F success."""
import hashlib,json,subprocess,sys
from pathlib import Path
from scripts import linear_L_confirm as ev

def main():
 d=ev.read(ev.R/'decision.json');assert d['complete'] and d['old_nonNN_superiority'];assert ev.read(ev.R/'audit.json')['pass_']
 root=Path('/tmp/blokus-linear-L-integration');branch='integration/linear-L-candidate'
 subprocess.run(['git','fetch','origin','main'],check=True)
 subprocess.run(['git','worktree','add',str(root),'-b',branch,'origin/main'],check=True)
 for rel in ['blokus/linear_features.py','blokus/linear_value.py','blokus/linear_tree.py','blokus/non_neural_play.py','models/linear-L','tests/test_linear_tree.py','tests/test_linear_L_export.py','LINEAR-L.md']:assert not (root/rel).exists(),f'preserve existing main file: {rel}'
 # Existing numerical inference and search stay intact; no trainers/history imports.
 rl=Path('blokus/linear_rl.py').read_text().split('class LinearAgent:')[0]
 (root/'blokus/linear_features.py').write_text(rl.replace('NumPy-only policy math: phase-specific linear REINFORCE, no neural model.','Non-neural candidate features used by fixed linear inference.').rstrip()+'\n')
 (root/'blokus/linear_value.py').write_text(Path('blokus/linear_value.py').read_text().replace('from .linear_rl import','from .linear_features import'))
 (root/'blokus/linear_tree.py').write_bytes(Path('blokus/linear_tree.py').read_bytes())
 folder=root/'models/linear-L';folder.mkdir(parents=True)
 manifest=[]
 for s in ev.SEEDS:
  source=ev.source('L',s);dest=folder/f'seed-{s}.json';dest.write_bytes(source.read_bytes());assert ev.base.sha(dest)==ev.base.sha(source);manifest.append(dict(seed=s,path=f'models/linear-L/seed-{s}.json',sha256=ev.base.sha(dest),source=str(source),research_base='b32b89caf483e98d75ba553731dfaca8cd302915'))
 (folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 (root/'blokus/non_neural_play.py').write_text(CLI)
 test=Path('tests/test_linear_tree.py').read_text();start=test.index('def test_saved_position_deepcopy_regression():');end=test.index('def test_depth_four_matches_separate_legacy_fixed_tree():',start);test=test[:start]+test[end:]
 (root/'tests/test_linear_tree.py').write_text(test)
 subprocess.run([sys.executable,'-m','scripts.linear_L_fixture',str(root)],check=True)
 (root/'LINEAR-L.md').write_text(DOC)
 with (root/'README.md').open('a') as f:f.write('\n### 非NN線形価値Lの追加候補\n\n公開gen3/default/Webは変更しません。CPUで遊べる固定線形方策・価値＋24×8/2ply探索の追加候補は[LINEAR-L.md](LINEAR-L.md)を参照。研究履歴は推論に不要です。\n')
 (root/'models/linear-L/confirmation.json').write_text(json.dumps(dict(results=ev.read(ev.R/'results.json'),decision=d,prefix_sha256=ev.read(ev.R/'prefixes.json')['sha256']),indent=2)+'\n')
 ev.write('integration-prepared.json',dict(path=str(root),branch=branch,base_main=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),model_exports=manifest))

CLI='''"""CPU-only non-neural linear policy/value candidate; no training history required."""
import argparse,json,random
from pathlib import Path
from .engine import State,play,legal_moves
from .linear_value import Agent
from .linear_tree import CompletedSearch

def main():
 p=argparse.ArgumentParser();p.add_argument('--model-seed',type=int,choices=[3101,3102,3103],default=3101);p.add_argument('--checkpoint');p.add_argument('--backend',choices=['python','native'],default='python');p.add_argument('--human-seat',type=int,choices=[0,1]);p.add_argument('--random-seed',type=int,default=1);a=p.parse_args()
 path=Path(a.checkpoint) if a.checkpoint else Path(__file__).resolve().parents[1]/'models'/'linear-L'/f'seed-{a.model_seed}.json'
 bot=CompletedSearch(Agent.load(path,a.backend),ms=100,width=24,replies=8,depth=2);other=Agent('old',backend=a.backend);s=State();rng=random.Random(a.random_seed);moves=[]
 try:
  while not s.done:
   if s.turn==a.human_seat:
    qs=legal_moves(s)
    if not qs:m=None
    else:
     print(s.render())
     for i,q in enumerate(qs):print(i,q.move)
     while True:
      try:
       i=int(input('合法手番号: '))
       if 0<=i<len(qs):m=qs[i].move;break
      except ValueError:pass
      print('表示された合法手番号を入力してください')
   else:m=(bot if a.human_seat is not None or s.turn==0 else other).act(s,rng)
   play(s,m);moves.append(None if m is None else [m.piece,m.orientation,m.row,m.col])
 except (EOFError,KeyboardInterrupt):print('対局を終了しました');return
 print('\\n'+json.dumps(dict(scores=[s.score(0),s.score(1)],moves=moves)))
if __name__=='__main__':main()
'''
DOC='''# 固定線形L：CPUで遊べる非ニューラル候補

既存の公開gen3・グローバルdefault・稼働Webを変更しない追加候補です。NumPyの段階別線形方策30係数と局面価値24係数を使用し、ニューラルネットを方策/価値に使いません。既存基盤の依存としてTorch CPUパッケージは必要です。推論に研究module・訓練履歴・巨大indexは不要です。

方策は以前の終局REINFORCE学習済み旧方策を固定。今回の価値は元の学習済み24係数を初期値に、固定された広幅非NN探索の自己対戦1,024ゲーム/seed・8epochの実終局勝敗から学び直しました（ゼロからの学習ではありません）。gen3の値や着手を教師にしていません。全3seed3101/2/3を提供し、例の既定3101は成績で選ばず先頭固定です。

```bash
python -m blokus.non_neural_play --model-seed 3101 --backend python --human-seat 0
# 後手: --human-seat 1、AI対戦: --human-seatを省く
# nativeを使う場合:
python -m blokus.ppo_native build
python -m blokus.non_neural_play --model-seed 3102 --backend native --human-seat 1
```

盤面開始点(4,4)/(9,9)、合法性、pass、全置き15/最後monomino20は既存ルールのまま。探索は完成6×3/2plyを確保し、24×8/2plyが全root完成した場合だけ置換、100ms soft deadline。未完了値を採用せず完成段階/greedyへfallbackします。完全探索ではありません。Python fallbackは遅く、同じdeadlineで完成範囲/着手が変わり得ます。cold初回の依存import費用とwarm対戦費用を区別してください。

評価数値はmodels/linear-L/confirmation.jsonと研究branch research/linear-L-confirmのREPORTを参照。新256prefix×先後×3seed、各1,536局、L/FとL/gen3の2主比較の補正CIで判定。旧32prefixを混ぜず、固定3seed/この評価分布の結論です。gen3単体より実思考が長く、同wall優越・全seed一般化・一般的最強の証明ではありません。旧局所表Tのgateを変更したものではなく、別の事前計画確認です。
'''
if __name__=='__main__':main()
