"""Write concise experiment report and compress replay logs after verification."""
import json,gzip,hashlib,time
from pathlib import Path
import numpy as np
ROOT=Path('artifacts/terminal-ppo')
def write(path,obj):path.write_text(json.dumps(obj,indent=2))

def main():
 ext=ROOT/'extended';d=json.loads((ext/'decision.json').read_text());sel=json.loads((ext/'selection.json').read_text());compute=json.loads((ext/'compute.json').read_text());timing=[];table='|seed / iteration|対学習前 W-L-D / 得点差|対heuristic W-L-D / 得点差|対旧value/2ply6ms W-L-D / 得点差|\n|---|---|---|---|\n'
 for seed,c in sel.items():
  cells=[]
  for name,path in [('untrained',ext/f"seed-{seed}/validation-{c['step']}-untrained.json"),('heuristic',ext/f"seed-{seed}/validation-{c['step']}-heuristic.json"),('oldvalue',ext/f'seed-{seed}/validation-selected-oldvalue.json')]:
   r=json.loads(path.read_text());s=r['summary'];cells.append(f"{s['wins']}-{s['losses']}-{s['draws']} / {s['mean_difference']:+.3f}");timing.append(dict(seed=seed,opponent=name,timing=r['timing']))
  table+=f"|{seed} / {c['step']}|"+'|'.join(cells)+'|\n'
 status='通過' if d['advance'] else '未達'
 hold=ext/'holdout-audit.json';holdtext='gate未達のためfresh対戦holdoutは作成していない。'
 if hold.exists():
  h=json.loads(hold.read_text());holdtext='選択固定後fresh96prefix先後交換を実施。集計（3seedはprefix内平均、CIはprefix単位）:\n\n'+json.dumps(h['aggregate'],ensure_ascii=False,indent=2)
  holdtext+='\n\n対heuristicのseed別（各192局）:\n\n|seed|勝-敗-分|得点差|得点差CI95%|\n|---|---|---|---|\n'
  for seed in sel:
   r=json.loads((ext/f'seed-{seed}/holdout-heuristic.json').read_text())['summary'];holdtext+=f"|{seed}|{r['wins']}-{r['losses']}-{r['draws']}|{r['mean_difference']:+.5f}|{r['difference_ci95']}|\n"
  holdtext+='\n各seedの平均差/CIは上表の通り。CIはこの3seedと共有prefixに条件付けたもので、全ての初期化や相手への安定した優越を保証しない。1401はvalidationにより128回（2048games）を選択、他2seedは512回（8192games）を選択し、holdoutで再選択していない。旧value/2ply6msには全seedで負けている。'

 details=[]
 for seed in [1401,1402,1403]:
  logs=json.loads((ext/f'seed-{seed}/curve.json').read_text());rows={}
  for name,ls in [('first16',logs[:16]),('last16',logs[-16:])]:rows[name]={k:float(np.mean([l['update'][k] for l in ls])) for k in ['entropy','value_mse','approx_kl','clip_fraction']}
  details.append(dict(seed=seed,**rows))
 (ext/'learning-summary.json').write_text(json.dumps(details,indent=2))
 text=f'''# 終局勝敗による自己対戦PPO（2026-11-05）

新系統を実装・実学習し、3seedの初回pilotと経験量だけを増やした1回の再pilotを完了。再pilotの固定gateは**{status}**。mainの既存低予算value/2ply・100ms教師は変更しない。以下はネット単体の結果で、探索教師の成績を流用しない。

設計・v1との差・報酬・制約は[METHOD.md](METHOD.md)、事前条件は[PLAN.md](PLAN.md)、再pilotの固定条件は[REPAIR-PLAN.md](REPAIR-PLAN.md)。全合法手、6ch盤面・49状態・正確な候補セルを使用。初期heuristic logit+ゼロ残差、valueゼロ。終局勝敗+1/0/-1、gamma1、shapingなし。得点差は評価指標のみ。PPO clip .2、2epochs、minibatch16、収集16games、固定pool現在50%/過去25%/heuristic25%。3seedは初期重みを共有、生成ゲーム・行動sample・更新順は異なる。

初回64iterations=1024games/seedでは、選択後の対heuristic点率.52344、CI[.46354,.58333]、平均得点差+.89583でgate未達。KL小・方策崩壊なし・経験量が少ないことから、許可された1回の修正として学習量だけを512iterations=8192games/seedへ増やした。初回で選ばれたcheckpointではなく各seedの最新64から継続し、評価prefixを1430000から1470000へ変更。他のモデル・損失・学習率・相手割合・gateは不変。これ以上の修正はしない。

## holdout分離の修正

当初seed1480000の96prefixを評価した後、厳密な分離監査で1prefixが訓練に一致と判明。seed分離のみでは短いprefixの重複を防げなかった。旧成績・監査失敗logをextended/holdout-1480000-overlapへ保全し、最終の独立holdout結果として採用しない。

checkpointを変更せず、seed1500000から訓練prefix hash/訓練state hashおよび旧validation/holdout/診断prefixとの一致を事前除外し、最初の96prefixを固定した。実際は先頭96件が全て条件を満たした。新結果を見てモデルやprefixを選び直していない。以下の最終holdoutはこの再評価。holdout-quality-failure/disjoint-planに経緯と固定hashを保存。

## 新validation（32prefix、各組64局）

{table}
選択にはheuristic/学習前を同じ比率で使用し、旧valueは除外。学習前はheuristic事前値を持つため両対照はほぼ同じ方策であり、独立した2種類の改善証拠と数えない。旧value/2plyは6msの追加探索を使うため、同一思考時間の比較ではない。validationはcheckpoint選択に使った結果であり、独立最終成績と扱わない。

固定gateの集計:

```json
{json.dumps(d['checks'],ensure_ascii=False,indent=2)}
```

{holdtext}

## 計算量・検証

初回を含む3seed累積の生成+更新wall **{compute['wall_seconds']:.2f}秒**、**{compute['games']}対局**、**{compute['ppo_updates']} minibatch更新**、current-policy記録**{compute['records']}件**。収集{compute['generation_seconds']:.2f}秒（CPU合法手生成/pack {compute['cpu_generation_pack_seconds']:.2f}秒、batch推論collation/転送/同期 {compute['inference_collation_transfer_sync_seconds']:.2f}秒）、更新{compute['update_seconds']:.2f}秒。初回112秒はこの累積に含み、二重加算しない。上限1800秒内。GPU更新だけで予算を数えていない。初期化・最終export・評価は別計測。主な負荷はCPU手生成/入力で、GPUだけを増強しても解決しない。

実対局の全推論費用・p95・全手msは各validation JSONのtiming、CUDA eventとbatchメモリはextended/policy-diagnostic.json。思考時間に合法手生成・特徴・転送・同期を含む。GPU phase時間と純device時間は区別。phase間に生成/監査を重ねていない。

合法手/mask、配置の区別と勾配、実方策更新、両者視点、pass、15/20点bonus、終局returnを検証。CUDA別process再開は過去policy交換をまたぐ6iterations連続 vs4保存+追加2、model/past/optimizer/対局履歴/Python/NumPy/Torch/CUDA RNGをbit比較。全テスト結果はtests.log。全訓練/評価対局再生、初期log_prob再計算、終局target、相手割合、選択最大値、prefix対の算術をaudit.jsonに保存。これは実装側監査であり秘書の独立実装監査とは呼ばない。

## 原因分析と次段階の判断材料

曲線はextended/curves.png/svgとextended/learning-summary.json。終局returnとpolicyは実際に変化したが、PPOの学習損失低下や自己対戦勝率を固定基準への強化成功と扱わない。heuristic事前分布の強さ、確率的訓練対greedy実戦の差、終局勝敗の高分散、短い過去policy pool、少数seed・prefixに限られる。詳細な方策変化/初期とのKLはextended/policy-diagnostic.json。

この依頼ではroot補正へ戻らず、自己対戦系統の再現可能な土台を残した。改善が確認できない場合は、追加の無根拠な長時間更新ではなく、固定poolに対する方策改善の再現性と確率的/greedy評価の差を先に診断する。次の学習条件変更や推奨変更は秘書の判断へ返す。

## 再現コマンド

```bash
cd /home/eletim/blokus-duo-ai
# 初回（他seedは1402,1403）
.venv/bin/python -m blokus.terminal_ppo --seed 1401 --iterations 64 --output /tmp/ppo-first
# 経験量だけ増加。--iterationsは累積値、最新64から開始
.venv/bin/python -m blokus.terminal_ppo --seed 1401 --iterations 512 --resume /tmp/ppo-first/iteration-64.pt --save-steps 128 256 512 --output /tmp/ppo-extended
# 評価は学習/生成と同時実行しない
.venv/bin/python -m blokus.terminal_eval validation --root artifacts/terminal-ppo/extended --prefix-seed 1470000 --steps 128 256 512
.venv/bin/python -m blokus.terminal_eval holdout --root artifacts/terminal-ppo/extended --fixture artifacts/terminal-ppo/extended/holdout-disjoint-prefixes.json
.venv/bin/python scripts/audit_terminal_holdout.py
.venv/bin/python scripts/check_terminal_resume.py
.venv/bin/python scripts/audit_terminal.py --root artifacts/terminal-ppo/extended
.venv/bin/python -m pytest -q
# ネット単体CLI（選択checkpointはextended/selection.json）
.venv/bin/python -m blokus.terminal_play --model {sel['1401']['path']} --opponent heuristic
# 同コマンドに --human-seat 1 で人間対AI
```

結果JSON・source hash・設定・seed・checkpoint metadataを保存。過去bank/holdout/探索教師ラベルを本学習には使わない。必要な初期/選択/継続用checkpointを保持し、重複latestと未選択中間checkpointは検証後にpruned-checkpoints.jsonへhashを記録して削減する。学習対局ログはgzip圧縮で保存し、auditは両形式に対応する。
'''
 (ROOT/'REPORT.md').write_text(text)
 # Save lightweight aggregate timing without duplicating arrays.
 write(ROOT/'timing-summary.json',[dict(seed=x['seed'],opponent=x['opponent'],timing=[{k:v for k,v in t.items() if k!='all_ms'} for t in x['timing']]) for x in timing])
 for root in [ROOT,ext]:
  selection=json.loads((root/'selection.json').read_text());pruned=json.loads((root/'pruned-checkpoints.json').read_text()) if (root/'pruned-checkpoints.json').exists() else []
  for seed,c in selection.items():
   d=root/f'seed-{seed}';last=64 if root==ROOT else 512
   keep={Path(c['path']).name,f'iteration-{last}.pt','untrained.pt'}
   for path in d.glob('*.pt'):
    if path.name not in keep:pruned.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),reason='unselected intermediate or duplicate latest; selected and final resume checkpoints retained'));path.unlink()
   g=d/'games.json'
   if g.exists():
    with gzip.open(d/'games.json.gz','wt') as f:json.dump(json.loads(g.read_text()),f,separators=(',',':'))
    g.unlink()
  write(root/'pruned-checkpoints.json',pruned)
if __name__=='__main__':main()
