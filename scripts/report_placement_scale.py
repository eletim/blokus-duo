"""Render the recorded scaling and separate early-stopping studies without pooling them."""
from pathlib import Path
import json
import numpy as np
R=Path('artifacts/value-placement-scale')
def read(p):return json.loads((R/p).read_text())
def ci(x):return f'{100*x[0]:.2f}–{100*x[1]:.2f}%'
def table(results):
 lines=['| 比較 | 勝–分–敗 | 勝点率 | 補正区間 | 平均得点差 |','|---|---:|---:|---:|---:|']
 for pair,v in results.items():lines.append(f"| {pair[0]} / {pair[1]} | {v['wins']}–{v['draws']}–{v['losses']} | {100*v['point']:.2f}% | {ci(v['point_adjusted_ci'])} | {v['difference']:+.2f} |")
 return '\n'.join(lines)
def latency(results):
 lines=['| 比較・手法 | 非pass平均ms | 最大ms | 6ply完了率 | 550ms超過率 | 方策fallback |','|---|---:|---:|---:|---:|---:|']
 for pair,v in results.items():
  for method,st in v['stats'].items():lines.append(f"| {pair}・{method} | {st['mean_ms']:.1f} | {st['max_ms']:.1f} | {100*st['depth6']:.2f}% | {100*st['over_550']:.2f}% | {st['fallback']} |")
 return '\n'.join(lines)
fixed=read('fixed16-decision.json');stopped=read('earlystop/decision.json');audit=read('audit.json');eaudit=read('earlystop/audit.json');assert audit['pass_'] and eaudit['pass_']
seeds=(1751401,1751402,1751403);sizes=(384,1536,3072)
curves={(n,v,k):read(f'n{n}/learning-{v}-{k}.json') for n in sizes for v in 'ABC' for k in seeds}
base=np.mean([curves[3072,'A',k]['curve'][0]['dev_mse'] for k in seeds]);scale_rows=[]
for n in sizes:
 d=curves[n,'A',seeds[0]];vals=[np.mean([curves[n,v,k]['curve'][-1]['dev_mse'] for k in seeds]) for v in 'ABC'];scale_rows.append(f"| {n:,} | {d['train_labels']:,} | {d['updates']:,} | "+' | '.join(f'{v:.4f}' for v in vals)+' |')
choice=read('earlystop-epochs.json');traincheck=read('earlystop/training-check.json');assert all(x['train_difference']==0 and x['dev_difference']==0 for x in traincheck['checks'])
stop_rows=[]
for v in 'ABC':
 checks=[x for x in traincheck['checks'] if x['variant']==v];stop_rows.append(f"| {v} | {choice[v]['epoch']} | {checks[0]['updates']:,} | {np.mean([x['dev_mse'] for x in checks]):.4f} |")
main_time=sum(read(f'n{n}/curve-time.json')['seconds'] for n in (384,1536))+read('n3072/validation-time.json')['seconds']+read('n3072/confirmation-time.json')['seconds'];early_time=read('earlystop/confirmation-time.json')['seconds']
status=(f"停止点選択版は **{','.join(stopped['passed'])}** が事前条件を達成しました。" if stopped['passed'] else '停止点選択版は事前条件未達でした。')
text=f'''# 残り駒と盤面を結びつける評価NN：学習量拡大と独立確認

**学習用データを96局から3,072局へ増やし、固定16周版Aは独立確認で現行gen3＋深い先読みへの改善を確認しました。** {status} 新モデルのWeb公開やmainへのマージは行っていません。

このレポートは[前回の小規模実験](../value-placement-ab/REPORT.md)の追加検証です。元の不採用結果は保持しています。固定16周のデータ量比較と、後から開発誤差を根拠に追加した停止点選択比較は、別々の試験として報告します。

## 何を学習したか

- A：両者の残り駒それぞれの現在の合法配置有無・配置数・4領域の分布、計252特徴をvalueへ追加。
- B：持ち駒42次元で2つの畳み込み層をFiLM変調し、盤面圧縮前から持ち駒を反映。
- C：元のvalue構造を同じデータで再学習する対照。
- F：未変更のgen3＋H探索。候補幅4、最大6ply、500ms。

A/B/Cはvalue側の畳み込み・盤面圧縮・評価部を学習します。候補手を選ぶpolicyは別の元gen3のまま固定しています。A72,529、B57,777、C56,401パラメータで、容量を揃えた構造だけの因果比較ではありません。

新しい自己対戦3,264局を生成し、学習3,072局・開発192局に分けました。学習84,361局面（先手42,959、後手41,402）、開発5,253局面です。3つの元gen3を使い、ランダム8手の後は元のHで終局まで継続しています。人間棋譜は使っていません。学習目標は実終局の勝敗で、得点差や真のminimax値ではありません。

各モデルは元gen3から開始。Adam1e-4、batch64、ゲーム単位の重みを揃えたMSE＋元valueからの変化を抑える係数0.1。3seedとも同じ学習・開発データを使い、良いseedだけの選別はしていません。大きい段階は小さい段階のデータを含みます。異なるデータ量では更新回数も増えるため、データ量と更新回数を独立に分離した実験ではありません。

## データ量と予測誤差

すべて16周の終点。誤差は共通の開発192局に対するゲーム重み付きMSEの3seed平均です。元gen3の平均は **{base:.4f}** です。

| 学習対局数 | 学習局面数 | 各モデルの更新回数 | A誤差 | B誤差 | C誤差 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(scale_rows)}

![学習曲線](learning-curves.png)

Aはデータ量を増やすと開発誤差が低下しました。一方、最大段階でも16周まで回すと、途中の最小誤差から上昇しています。**学習データを増やすことと、同じデータを長く回すことは別**です。予測誤差の改善だけでは棋力改善とは扱いません。

固定16周版Aの診断では、現在置けない残駒が10マス以上ある開発局面の誤差は元モデル0.4446から0.3200へ低下しました。2,347局面は同じ192ゲーム内で相関しており、独立標本数ではありません。この診断は採用判定に使っていません。

## 固定16周版：候補選択と独立確認

[事前計画](PLAN.md)をデータ生成前に保存しました。384局・1,536局段階は共通の6開始局面で各108対局の参考比較。最大段階は新16開始局面×両席×3seed、各96局で候補を選択しました。次表の区間は参考値で、5比較全体の有意差判定には使いません。採用の根拠は後続の独立確認です。

{table(read('selection.json')['results'])}

Aが選択条件を通過したため、新32開始局面でA/F、A/Cを各192局、計384局の独立確認を実施しました。以下はこの確認だけの結果で、上の選択用対戦や過去実験とは合算していません。勝点率は勝ち1・引分0.5。開始局面ごとに両席・3seedをまとめて20,000回bootstrapし、2主要比較を補正した97.5%区間です。

{table(fixed['results'])}

A/Fの平均得点差の区間は+1.47～+3.20、A/Cは+0.73～+2.68でした。両比較の勝点率下限が50%を超え、正の得点差・遅延条件も満たしたため、**固定16周版Aは事前条件達成**です。

{latency(fixed['results'])}

同じ500ms上限ですが、Aは実際の計算時間を多く使っています。同じ実消費時間での優越を示す試験ではありません。6plyを完了できない場合は完成済み4plyへ戻し、未完成の探索値は使いません。

## 開発誤差で停止点を選んだ追加比較

16周の終点で過学習が見えたため、固定版のvalidation進行中・対戦成績を参照する前に、[別の追加計画](EARLYSTOP_PLAN.md)を保存しました。固定版の結果や確認は置き換えていません。

各構造で「3seed平均の開発MSEが最小のepoch、同点なら早い方」という共通ルールで、A6周・B5周・C7周を選択しました。モデルごとに有利なseedを選んだものではありません。元重みから同じ更新列を再実行し、記録済みepochの訓練・開発MSEは全9モデルで差0でした。

| 構造 | 停止epoch | 各モデルの更新回数 | 開発MSE・3seed平均 |
|---|---:|---:|---:|
{chr(10).join(stop_rows)}

停止点の選択に使った開発データは、以下の対戦開始局面と分離しています。過去・今回の訓練/開発・元の選択/確認対戦の訪問状態を除外した、さらに新しい32開始局面×両席×3seedです。各比較192局、計960局を実行しました。

主要比較はA/F、A/C、B/F、B/Cの4つ。開始局面単位のbootstrapで4比較を補正した98.75%区間です。C/Fは参考対照で、後付けの採用候補にはしていません。

{table(stopped['results'])}

{status} 判定は各候補のF/C両方で勝点率下限50%超、正の平均得点差、非passの550ms超過率1%以下です。追加の救済調整や標本追加はしていません。

固定16周版と停止点選択版は異なる独立bankを使っています。A16対A6の直接対戦ではないため、両表の数値差を停止時点だけの効果として断定できません。小さい段階の参考対戦も、開始局面が6つのみなので、学習量ごとの勝点差は記述的な結果です。

{latency(stopped['results'])}

## 検証と計算量

- 固定16周版の生成・評価4,344対局＋追加960対局、**合計{audit['games']+eaudit['games']:,}対局記録**を独立した座標集合実装で再生。合法性・強制pass・終局得点・算術を確認。
- 元gen3の訓練訪問状態27,533,566件との検査対象状態の重複0。学習/開発、試験群間、追加比較と過去・今回の訪問状態の重複検査もPASS。2つの小規模curve段階は計画どおり同じbankを共有しています。空盤面付近のランダムprefix過程は採用局面から除外しています。
- これは状態hashでの分離であり、対称形や戦略的に似た局面が一切ないという意味ではありません。
- 新clone CPUテスト110成功・5skip。固定版27export、追加9exportのCPU読み出し・合法着手・元policy不変を確認。実保存Adam/RNGのCPU/CUDA別プロセス再開を確認（137局面fixtureで6周継続と2周＋復元＋4周の一致）。追加9モデルは新cloneのCPU経路でも確認済み。
- 生成：{read('generation-time.json')['seconds']/60:.2f}分（8物理コアへ固定）。特徴準備：{read('feature-time.json')['seconds']:.2f}秒。固定版27モデル学習：{read('training-time.json')['seconds']/60:.2f}分。停止点選択版9モデルの再現：{traincheck['seconds']/60:.2f}分。
- 固定版の段階比較・validation・confirmation計1,080局：{main_time/60:.2f}分。停止点選択版960局：{early_time/60:.2f}分。対戦はCPU core0、torch1thread、native、逐次実行。学習・監査負荷を対戦中には重ねていません。思考時間はウォームアップ後の非pass着手を集計しています。

[元試験の全監査](audit.json)、[追加試験の監査](earlystop/audit.json)、学習曲線・全棋譜・保存モデル・optimizer/RNG・ファイルSHAはこのディレクトリにあります。[再現手順](README.md)。固定版モデル/bankを固定したcommitはd80fe52、追加計画はd81ef80、追加モデル/bank固定は71c0bf1です。

## 適用範囲

元の候補policy上位4手、最大6ply、同じ500ms上限、3つの元gen3モデル、新しい合法ランダムprefixからの対戦での結果です。空盤面からの人間対戦での優越や、あらゆる対戦相手への優越を示していません。

Aは現在の合法配置情報を明示できますが、将来その置き場を確保できることまでは保証しません。A/Bのパラメータ数は異なり、停止点選択版では更新回数も異なります。結果から「81次元の圧縮だけが弱さの唯一の原因だった」とは断定できません。

**main・公開設定・Frozen gen3の重みとタグは維持しています。新しい候補は研究用PRで保存した段階で、Webへの反映は未実施です。**
'''
(R/'REPORT.md').write_text(text);print(R/'REPORT.md')
