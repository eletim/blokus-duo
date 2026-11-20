"""Factual extension report from completed files; no post-hoc selection."""
import json,collections
from pathlib import Path
import numpy as np
R=Path('artifacts/value-placement-49152')
def read(p):return json.loads((R/p).read_text())
def table(out):
 text='|比較|勝–分–敗|勝点|通常95%CI|family補正CI|得点差（通常95%CI）|\n|---|---:|---:|---:|---:|---:|\n'
 for p,v in out.items():
  ci=v['point_ci95'];a=v['point_adjusted_ci'];s=v['difference_ci95'];text+=f"|{p}|{v['wins']}–{v['draws']}–{v['losses']}|{100*v['point']:.2f}%|{100*ci[0]:.2f}–{100*ci[1]:.2f}%|{100*a[0]:.2f}–{100*a[1]:.2f}%|{v['difference']:+.2f}（{s[0]:+.2f}–{s[1]:+.2f}）|\n"
 return text

def main():
 audit=read('audit.json');assert audit['pass_'] and read('n49152/checks-post-training.json')['pass_'] and read('n49152/clean-check.json')['pass_']
 for n in (36864,49152):assert read(f'n{n}/export-check.json')['pass_']
 tele=read('generation-telemetry.json');data=read('data-audit.json');decision=read('n49152/decision.json');final=read('n49152/confirmation-summary.json');passed=decision['passed'];trend=[read(f'n{n}/validation-summary.json')['AP'] for n in (36864,49152)];flat=all(v['point_ci95'][1]<.53 for v in trend)
 text=f'''# 49,152局までの追加データ学習と独立確認

**GPU生成・2段階の実CUDA学習・新1,728局の対戦・監査を完了。採用gate：{'通過' if passed else '未達'}。** main/Frozen gen3/Webは不変。新規24,576訓練局と192dev局を生成し、旧24,576訓練局のみへ追加しました。旧評価/速度bankや旧devを訓練へ転用していません。勝点は勝1・引分0.5。

5時間枠の事前PLANは`cdd2210`、新bankは`3701319`。段階36,864/49,152、各A/C全3seed、各16epochを事前固定。最終候補は対戦成績を見ず49,152局版Aに固定しました。前回結果・gateの変更や救済スイープはありません。

## 実際に更新した学習

教師は固定Frozen gen3幅4完成6ply、生成はasync512/batch128/8workers+coordinator。方策は別の元gen3で固定、価値コピーだけを更新。Aは252合法配置特徴追加、Cは元構造。両者同じゲーム/局面/順序/更新予算、Adam1e-4/batch64/float32、実終局WDLのgame正規化MSE＋元value逸脱抑制0.1。各段階は元gen3から新規開始し、旧学習済valueからの継続ではありません。

|累積訓練ゲーム|局面数|各モデル最大updates|6モデル計updates|dev選択共通epoch|選択時updates|CUDA学習wall分|
|---:|---:|---:|---:|---:|---:|---:|
'''
 for n in (36864,49152):
  m=read(f'n{n}/data-manifest.json');sel=read(f'n{n}/checkpoint-selection.json');t=read(f'n{n}/training-time.json');ss=audit['stage_selections'][str(n)];text+=f"|{n:,}|{m['old_labels']+m['new_labels']:,}|{ss['updates_per_model']:,}|{6*ss['updates_per_model']:,}|{sel['epoch']}|{sel['epoch']*ss['steps_per_epoch']:,}|{t['seconds']/60:.2f}|\n"
 text+=f"\n全12モデル合計 **{audit['total_real_updates']:,} Adam更新**。各段階の6モデル平均新devMSE最小で共通epochを選択し、同点は早いepoch。新devは両段階で共用した選択データであり、独立棋力証拠ではありません。最大16epochのoptimizer/全RNGと全epoch推論exportを保持。\n\n![実学習曲線](learning-curves.png)\n"
 text+='\n## 新しい段階比較（探索的）\n\nCPU native1thread/core0、幅4/max6/500ms。各段階は別の新8prefix×先後×全3seed、2比較、96局。Pは直前量のA（36,864段階では24,576、49,152段階では36,864）、FはFrozen H。2比較補正97.5%CIと通常95%CI。ここで最終候補を差し替えていません。\n'
 for n in (36864,49152):text+=f'\n### {n:,}ゲーム段階\n\n'+table(read(f'n{n}/validation-summary.json'))
 text+=f"\n事前の実用差+3ppに対して、両段階の95%CI上限が53%未満という伸び鈍化条件は **{'成立' if flat else '不成立'}**。{'この固定方法・分布での鈍化の証拠であり、一般的な棋力上限ではありません。' if flat else '点推定の上下や単発の非有意だけから上限・同等性を結論しません。'}\n\n![段階別の傾向](stage-trends.png)\n"
 text+='\n## 独立確認（最終判定）\n\n事前固定の新64prefix×先後×3seed、各384局・計1,536局。Q=旧成功3,072局A16、R=前回24,576局A6、F=Frozen gen3+H、C=今回49,152局C。同一prefix内の先後/seed相関を保った20,000回bootstrap、4比較Bonferroni98.75%区間。段階試験や旧確認は合算していません。\n\n'+table(final)
 text+='\n![独立確認の勝点](confirmation-points.png)\n\n'
 for pair,label in [('AQ','旧3,072局A'),('AR','前回24,576局A'),('AF','Frozen H'),('AC','同量C')]:
  v=final[pair];text+=f"- {label}：補正下限{100*v['point_adjusted_ci'][0]:.2f}%。この評価分布での優越は{'確認' if v['point_adjusted_ci'][0]>.5 and v['difference']>0 else '未確認'}。\n"
 text+=f"\n**推奨：{'新49,152局Aを統合検討候補として返します（本番未適用）。' if passed else '旧成功候補を維持します。全比較の事前採用条件は未達で、新モデルの自動採用は推奨しません。'}** 未確認は同等性や劣化の証明ではありません。人間への強さ・固定3seed以外への一般化は未検証。\n"
 text+='\n### 実思考時間\n\n|比較/手法|平均ms|最大ms|p95ms|p99ms|完成6ply率|550ms超過率|greedy fallback|\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
 for pair,v in final.items():
  for name,s in v['stats'].items():text+=f"|{pair}/{name}|{s['mean_ms']:.2f}|{s['max_ms']:.2f}|{s['p95_ms']:.2f}|{s['p99_ms']:.2f}|{100*s['depth6']:.2f}%|{100*s['over_550']:.3f}%|{s['fallback']}|\n"
 text+='\n同じ500ms上限でも実消費wallは異なります。完成6ply未達時に完成4plyを保持した着手とgreedy fallbackを区別。warmup費用は各phase-time.json、時間対戦と生成/学習/重監査は別工程です。\n'
 gpu=np.array([float(s['gpu'].split(',')[0]) for s in tele['samples']]);mem=np.array([float(s['gpu'].split(',')[1]) for s in tele['samples']]);timing=read('games/timing.json');batches=collections.Counter()
 for g in timing['groups']:batches.update({int(k):v for k,v in g['batch_hist'].items()})
 text+=f"\n## 費用と予算\n\n新24,768局生成wall **{tele['outer_seconds']/60:.2f}分**、初期化{timing['initialization_seconds']:.2f}秒。GPU利用率平均{gpu.mean():.1f}%、中央値{np.median(gpu):.1f}%、外部観測VRAM最大{mem.max():.0f}MiB（デスクトップ等の利用を含みます）。512同時slotは実NN batchと同じ意味ではありません。raw batch/queue/転送/pack/CPU秒はgames/timing.jsonに保持。独立特徴化{data['seconds']/60:.2f}分。\n"
 for n in (36864,49152):
  tt=read(f'n{n}/training-telemetry.json');util=np.array([float(x['gpu'].split(',')[0]) for x in tt['samples']]);text+=f"- {n:,}学習：外側{tt['outer_seconds']/60:.2f}分、GPU利用率平均{util.mean():.1f}%。段階対戦{read(f'n{n}/validation-time.json')['seconds']/60:.2f}分。\n"
 text+=f"- 最終確認：{read('n49152/confirmation-time.json')['seconds']/60:.2f}分。\n"
 text+='''
追加第三段階・49,152局を超えるデータ・教師/構造変更・LR探索は実施していません。今回の有限5時間枠は2段階と独立確認・保存へ先に配分しました。旧6,144局のCPU500ms由来と追加GPU完成6plyの差、GPU近接tie丸め、旧A16とdev選択モデルの更新数/epoch差を保持しており、純粋な局数のみの因果効果やCPU/GPU bit同値とは呼びません。

## 正しさと再現

全新生成24,768局を独立座標checkerで合法手/forcedpass/終局+15/+20/得点再生、実手番WDL・game重み・完成6plyを監査。既存train/dev/評価/速度bankと元PPOindexへの分離、新train/dev分離、3組の新評価実訪問状態の相互分離を確認。全1,728新対戦を独立再生し勝点・得点・CIを再計算。source/bank/code/model SHAと実更新数・epoch選択も再検算（audit.json）。初期空盤面など共通prefix途中と学習対象/評価開始は区別。

変更していない更新式とAdam/RNG復元は前回のCPU/CUDA別process bit一致検証を参照し、新ドライバは旧式のAST一致と累積shard/対照識別をテスト。今回実学習checkpointからのCPU/CUDA別process再開も確認。全12exportのCPU/native/PythonとCUDA値は事前許容差2e-5以内、実重み変化・policy保持・合法完成木を確認。クリーンGit checkout、CUDA非表示で関連回帰・全6最終exportのCPU対局・人間先後CLI、全8棋譜の独立再生を確認。既存CUDA対応wheelを非表示で使用し、新CPU専用wheel導入とは区別。

途中の待機監督processがtool shell終了後に残らなかったため監督のみ復旧しました。GPU生成processは継続、二重生成/計算再実行なし（supervision.json）。AI担当runと外部計算障害を混同しません。

[再現コマンド](REPRODUCE.md)。旧巨大履歴を複製せずSHA参照、新棋譜だけbyte一致アーカイブ。推論exportは履歴/optimizer/index不要。全結果と失敗を研究branch/同PRへ保存、本番変更なし。
'''
 (R/'REPORT.md').write_text(text)
 if batches:pass
 e=dict(passed=passed,flattening_criterion_met=flat,total_real_updates=audit['total_real_updates'],candidate='fixed n49152 A, dev epoch only',comparison_family=4)
 (R/'final-outcome.json').write_text(json.dumps(e,indent=2));print('REPORT READY',e)
if __name__=='__main__':main()
