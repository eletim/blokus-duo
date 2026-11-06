"""Summarize preserved observations; does not select settings or rerun games."""
import json
from pathlib import Path
R=Path('artifacts/gen3-value-leaf')
def load(n):return json.loads((R/n).read_text())
def interval(x):return '–'.join(f'{v*100:.2f}' for v in x)
def table(summary):
 out=['|比較|point %|95% CI %|平均得点差|得点差95% CI|実ms/手（左/右）|','|---|---:|---|---:|---|---|']
 for name,x in summary.items():
  a,b=name.split('-');t=x['time'];out.append(f"|{name}|{x['point']*100:.2f}|{interval(x['point_ci95'])}|{x['difference']:+.3f}|{x['difference_ci95'][0]:+.3f}–{x['difference_ci95'][1]:+.3f}|{t[a]['mean_ms']:.2f}/{t[b]['mean_ms']:.2f}|")
 return '\n'.join(out)
def main():
 selection=load('selection.json');v=load('validation-summary.json');aux=load('aux2000-summary.json')
 text='''# 葉value専用経路Eと完成4ply後拡幅D

追加学習0。Frozen gen3全3seed（1751401/2/3）とmain/Webは変更していない。人間棋譜は未取得で、人間より強いとは主張しない。

## 固定設計と等価性

A=既存幅4/深さ2→4。E=同じ木で葉のpolicy候補生成/forwardを省略。D=Eの幅4/depth4完成後、内部幅4のままroot8→12へ拡幅。各段階は全root同深度完成後だけ採用する。途中打切りは最後の完成段階へ戻り、greedyを候補から落とさない。Dの完成値cacheは1着手内のみ、状態/手番/残駒/last/pass/深さ/内部候補幅を区別し未完成値を保存しない。

value入力は既存board4/6chとstate49（残駒・面積・frontier・両者得点・pass）。候補policy特徴への依存はない。事前atol1e-7/rtol0、1188値比較の最大差0、102完成木の着手一致。CPU/Python/native、4/6ch、序中終盤/pass/bonusを含む。同一完成木の結果と、deadlineにより完成深さが変わる実対戦着手は区別する。

## 処理費用pilot

8保存局面・2反復をA/E/D順と逆順で交互測定、初期化除外。2000msで同じ4ply完成木のPython A320.3→E97.9ms（約3.27倍）、native A92.1→E39.6ms（約2.33倍）。native自体の効果は別要因。Dは追加rootを評価するためEより実消費が増える。小pilot分布での測定であり一般速度保証/棋力改善ではない。全計測192件はpilot.json。

## 新validation（選択用）

新16prefix×先後×固定3seed、各比較96局、5比較480局。pointは勝1/分0.5。両席・3seedを同じprefixクラスタとして20,000回bootstrap。以下は探索的通常95%CIで、多数validationからの優越証明ではない。

'''+table(v)
 text+='\n\n## 2000ms補助（主条件の救済選択には使わない）\n\n新8prefix×先後×3seed=48局。\n\n'+table({'D-A':aux})
 text+=f"\n\n事前gate通過: **{selection['gate']}**。PLAN.mdの条件は変更していない。\n"
 if (R/'confirmation-summary.json').exists():
  c=load('confirmation-summary.json');text+='\n## 独立確認（500ms）\n\n新32prefix×先後×3seed、各比較192局。旧8prefix/validationは合算しない。\n\n'+table(c)+'\n\n'
  for n,x in c.items():text+=f"- {n}: 2主比較Bonferroni 97.5% point CI {interval(x['point_ci_bonferroni2'])}%\n"
  success=all(x['point_ci_bonferroni2'][0]>.5 and x['difference']>0 for x in c.values());text+=f'\n両比較での事前改善確認条件達成: **{success}**。\n'
 else:text+='\n確認試験は未実施。gate未達を条件調整で救済せず、未使用確認bankを保持した。\n'
 text+='''
## 計測と監査の読み方

各match JSONに全棋譜、全着手時間、完成depth/root幅、NN呼出数、cache hit、fallback、cold load/warmup、wallを保存。Aは旧実装の統計範囲（depth/nodes/fallback）であり記録されないNN回数を0回と解釈しない。forced passと時間切れfallbackを区別する。予算は同じ上限で実時間同一ではなく、単体gen3も同時間対照ではない。

モデルSHAはmodels.json、分離元はseparation-source.json。全訓練訪問状態indexと過去評価開始状態から開始bankを除外、評価後の訪問状態もaudit.jsonへ記録する。全棋譜を独立座標checkerで再生し合法配置・pass・終局得点を確認、集計とCIを再計算する。詳細はcoordinate-audit.json。前回B/C不採用の記録は保持。

## 再現

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python scripts/check_value_leaf.py
python scripts/pilot_value_leaf.py
# 元の分離index/過去bank参照が必要。既存bankは成績を見て再生成しない。
python scripts/run_value_leaf.py prepare
python scripts/run_value_leaf.py run
python scripts/run_value_leaf.py audit
python scripts/audit_value_leaf_records.py
python -m pytest -q
python -m blokus.ppo_value_leaf_play --method E --budget-ms 500 --human-seat 0
python -m blokus.ppo_value_leaf_play --method D --budget-ms 500 --human-seat 1
```

研究CLIのDは研究条件であり本番推薦とは別。保存済みmatchをrunは再利用するため、再測定時は既存結果を保全し別出力ディレクトリ/新計画を使う。
'''
 (R/'REPORT.md').write_text(text)
if __name__=='__main__':main()
