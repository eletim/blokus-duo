"""Render final factual summary only after all computation/audits/checks exist."""
import json
from pathlib import Path
import numpy as np
R=Path('artifacts/value-placement-24576')
def read(name):return json.loads((R/name).read_text())
def main():
 for name in ('audit.json','export-check.json','checks-post-training.json','clean-check.json'):assert read(name)['pass_']
 data=read('data-manifest.json');sel=read('checkpoint-selection.json');screen=read('selection.json');train=read('training-time.json');gen=read('games/timing.json');tele=read('generation-telemetry.json');phase='confirmation' if screen['passed'] else 'validation';out=read(phase+'-summary.json');passed=read('decision.json')['passed'] if screen['passed'] else False
 gpu=np.array([float(s['gpu'].split(',')[0]) for s in tele['samples']]);memory=np.array([float(s['gpu'].split(',')[1]) for s in tele['samples']]);steps=(data['old_labels']+data['new_labels']+63)//64
 text=f'''# GPU生成から24,576局の実学習・新対戦まで

**追加学習と必要な性能評価を実行完了。採用gate：{'通過' if passed else '未達'}。** main・Frozen gen3・公開Webは変更していません。

検証済みasync512を使って新18,432訓練局と別192開発局を生成し、既存6,144訓練局と合わせた24,576局でA/C各3seedを元gen3から実際にCUDA更新しました。旧モデルのload/exportだけではありません。勝点は勝ち1、引分0.5。{'新32prefixの独立確認まで実施しました。' if screen['passed'] else '新16prefixのvalidation全288局は完了しましたが、事前screen未達のため独立確認を実施せず、追加条件で救済していません。'}

## 固定条件と学習の事実

- [PLAN](PLAN.md)は生成・学習・新対戦結果より前のcommit `f858953`。train/dev bank固定は`38ad89f`。途中の担当AI capacity失敗時も外部生成は継続し、待機監督だけ引き継ぎました（handoff.json）。
- 教師：Frozen gen3の3seed、幅4・完成6ply。A教師への変更なし。各game独立RNG、既存速度bank/評価bank/人間棋譜は学習へ転用なし。
- 学習方策は別の元gen3として凍結。valueコピーのみ更新。Aは合法配置の252特徴を追加、Cは従来構造。同一データ・同順序・同更新予算。真の終局WDL MSE＋元valueからの逸脱抑制0.1、Adam1e-4、batch64、float32。
- 訓練局面数：**{data['old_labels']+data['new_labels']:,}**（旧{data['old_labels']:,}＋新{data['new_labels']:,}）、新開発局面{data['dev_labels']:,}。全6モデルを各16epoch、各**{16*steps:,} optimizer updates**実行。
- A/C×全3seed平均の新dev MSEだけで共通**epoch {sel['epoch']}**を選択。各公開研究exportは**{sel['epoch']*steps:,} updates**時点。対戦成績によるモデル/seed/epoch選別なし。最新16epochのAdam・CPU/CUDA/NumPy/Python RNGも別ファイルで保存。
- 既存6,144局はCPU500ms生成で完成深度が変わり得る一方、新規分はGPU完成6plyです。近い同点でGPU丸めによる着手差もあります。したがって純粋なデータ数だけの因果効果、CPU/GPU bit同値とは呼びません。

![実CUDA学習の曲線](learning-curves.png)
![同じ新devでの量別誤差](quantity-dev-curve.png)

量別図は旧3,072/6,144/新24,576局を固定16epochで同じ新devへ適用した記述的比較です。新devは選択用であり、棋力の独立証拠ではありません。

## 新しい対戦結果

全条件CPU native、torch1thread、core0、幅4/max6/500ms上限。学習・生成・重い監査と時間対戦を重ねていません。Q=旧成功3,072局A16、F=Frozen gen3+H、C=今回同量の従来構造。3seed固定、先後交換、prefixをクラスタとして20,000回bootstrap。通常95%区間と3主比較補正98.3333%区間を分け、旧試験やvalidationを確認標本へ合算していません。
'''
 for ph in ('validation','confirmation'):
  if not (R/f'{ph}-summary.json').exists():continue
  d=read(ph+'-summary.json');text+=f'\n### {ph}\n\n|比較|勝–分–敗|勝点率|通常95% CI|3比較補正CI|得点差（95% CI）|\n|---|---:|---:|---:|---:|---:|\n'
  for pair in ('AQ','AF','AC'):
   v=d[pair];ci=v['point_ci95'];adj=v['point_adjusted_ci'];sc=v['difference_ci95'];text+=f"|{pair}|{v['wins']}–{v['draws']}–{v['losses']}|{100*v['point']:.2f}%|{100*ci[0]:.2f}–{100*ci[1]:.2f}%|{100*adj[0]:.2f}–{100*adj[1]:.2f}%|{v['difference']:+.2f}（{sc[0]:+.2f}–{sc[1]:+.2f}）|\n"
 text+='\n![新対戦の勝点と補正区間](strength-points.png)\n\nseed別・先後別値、得点差補正CIは各summary JSONに保存しています。標本を途中追加せず、全seedを保持しました。人間への強さは未検証です。\n\n### 実時間と探索\n\n|比較/手法|平均ms|最大ms|p95ms|p99ms|6ply完成率|550ms超過率|fallback数|\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
 for pair in ('AQ','AF','AC'):
  for name,v in out[pair]['stats'].items():text+=f"|{pair}/{name}|{v['mean_ms']:.2f}|{v['max_ms']:.2f}|{v['p95_ms']:.2f}|{v['p99_ms']:.2f}|{100*v['depth6']:.2f}%|{100*v['over_550']:.3f}%|{v['fallback']}|\n"
 cpu=sum(g['server_cpu_seconds']+sum(w['cpu_seconds'] for w in g['workers']) for g in gen['groups']);text+=f'''
同じ500ms上限でも実消費時間は同じではありません。cold/warm準備費用は各phase-time.jsonに分離。完了段階だけ採用し、深く読めたことだけを棋力改善とはしていません。

## 費用

- 新18,624局生成（開発含む）：外側wall **{tele['outer_seconds']/60:.2f}分**、内部初期化{gen['initialization_seconds']:.2f}秒、内部定常＋fill/tail {gen['seconds']/60:.2f}分。再測定の速度比較ではなく本データ生成の費用です。
- 生成GPU利用率 平均{gpu.mean():.1f}%/中央値{np.median(gpu):.1f}%、外部観測VRAM最大{memory.max():.0f}MiB、Torch peak{gen['peak_memory']/2**20:.1f}MiB。外部GPU利用率にはデスクトップ等も含まれます。
- 生成worker/coordinator CPU累積{cpu:.1f}秒（初期化のCPU秒を含まない）。実batch/queue/transfer/capture/occupancyはgames/timing.json、毎秒telemetryに記録。
- 独立再生＋特徴生成：{data['seconds']/60:.2f}分。
- 6モデル実CUDA学習：**{train['seconds']/60:.2f}分**、Torch peak{train['cuda_peak_bytes']/2**20:.1f}MiB。epoch/update数はlearning-*.json、GPU観測はtraining-telemetry.json。
'''
 for ph in ('validation','confirmation'):
  if (R/f'{ph}-time.json').exists():text+=f"- {ph}時間対戦：{read(ph+'-time.json')['seconds']/60:.2f}分（warmup内訳別記）。\n"
 text+='''
## 検証・再現・限界

- 新18,624局は全て独立座標checkerで合法性・強制pass・終局+15/+20を再生し、実手番WDL・完成6ply・RNG/game識別・元Frozen SHAを監査。学習ラベルは8ply prefix後の局面。初期空盤面など共有されるprefix途中状態と採用評価開始/学習対象状態は区別。
- 新train/dev、過去評価/速度bank、元PPO訓練index、今回対戦の実訪問状態を照合。新対戦全棋譜・得点・勝点・CIを再計算（audit.json）。旧6,144局の過去監査と元履歴はSHA参照で保持。
- CPU/CUDA別processで連続3epochと1＋再開2epochの重み・Adam・全RNGがbit一致。最新の実学習checkpointからも検査。A/C初期予測一致と実重み変更、凍結policy保持を確認。
- 新exportのCPU/CUDA値は事前絶対許容差2e-5以内。合法手と完成木を確認。生成全軌跡のCPU/GPU bit一致を意味しません。
- クリーンGit checkoutでCUDAを隠し、全testsと全6exportのCPU対局、人間先後CLIを実行・独立再生。既存CUDA-capable torch wheelをCUDA非表示で利用しており、CPU専用wheelの新規導入試験とは区別。
- 元gen3コード/重みを上書きせず、推論exportとAdam/RNG再開checkpointを区別。mainへのmerge/deployなし。

[再現コマンドと履歴不要CPU CLI](REPRODUCE.md)。新棋譜はgame-archivesの小分けアーカイブ、同一byteのmanifest付き。旧巨大履歴は複製していません。

今回の未達は一般的な強さの上限や学習不可能の証明ではありません。結果が改善を示さない場合、固定幅4の元policyが作る継続分布、WDL回帰と探索時の選択損失の差、データ分布変化が残る仮説です。この試験内で教師・幅・loss・LRを救済変更することはしませんでした。
'''
 (R/'REPORT.md').write_text(text);print('REPORT WRITTEN',phase,passed)
if __name__=='__main__':main()
