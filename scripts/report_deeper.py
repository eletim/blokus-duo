import json
from pathlib import Path
R=Path('artifacts/gen3-deeper')
def table(d):
 rows=['|比較|point|95% CI|得点差|得点差95% CI|平均ms 左/右|','|---|---:|---|---:|---|---|']
 for k,x in d.items():
  a,b=k.split('-');lo,hi=x['point_ci95'];dl,dh=x['difference_ci95'];rows.append(f"|{k}|{100*x['point']:.2f}%|{100*lo:.2f}–{100*hi:.2f}%|{x['difference']:+.3f}|{dl:+.3f}–{dh:+.3f}|{x['time'][a]['mean_ms']:.2f}/{x['time'][b]['mean_ms']:.2f}|")
 return '\n'.join(rows)
def main():
 sel=json.load(open(R/'selection.json'));v=json.load(open(R/'validation-summary.json'));diag=json.load(open(R/'diagnostic.json'))
 out='''# E公開統合と固定幅6ply研究

## 現在使える変更
Eはmain **8fb3e9d**へ通常FF+分離記録commitで統合し、http://100.95.30.96:8765 の既存Webへ反映。rollbackは4f13460、設定/起動コマンドはartifacts/value-leaf-deployment/DEPLOY.md。NN探索の葉でpolicy候補処理を省略する計算効率改善であり、棋力改善とは表示しない。単体gen3/非NNを含む人間先後smoke PASS、gen3全重みとFrozenタグを保持。

## 今回の研究条件
Hは幅4を全深度で固定し、深さ2→4→6の各全root完了時のみ採用。時間切れは最後の完成段階（通常4ply）へ復帰。alpha-betaとTTは固定木を変えない。TT keyは盤面/手番/残駒/last/pass/深さ/幅、EXACT/LOWER/UPPERを区別、未完了ノードは保存しない。boundは既存窓で確実なcutoff時のみ利用し、窓を狭めない保守的方式。act間再利用なし。モデル再学習0、候補拡幅Dは不採用のまま。

固定木の事前許容差はatol1e-7/rtol0。枝刈りなし幅4/depth6、幅2/depth4/6、従来E4との値/選択、小終盤独立oracle、pass/bonus、打切り4ply復帰をテストした。初回1件はテスト側が旧クラスに上限外20000msを指定した失敗で、5000msへ修正（元ログ保持）。TT窓/flag分類の潜在的不整合をvalidation前レビューで保守化し、pilotを再実施。修正前結果は別名で保持、validation中の変更なし。

## 費用pilotと固定計画
新9保存局面×2反復、E/Hを交互順に測定。500ms E69.21ms/H271.10ms、H6ply8/18。2000ms E67.82ms/H341.91ms、H6ply14/18。成績を見る前に500ms/maxdepth6を固定、2000ms対戦の追加探索はしない。合法手生成/NN推論/同期を含む実時間、同上限であり実消費同一ではない。

新validation16prefix×先後×固定3seed（1751401/2/3）、各比較96局。H/Eが主、H/単体policyが補助。pointは勝1/分0.5。両席/3seedを同prefixクラスタとしてbootstrap20,000回。以下は選択用通常95%CIで、確認標本と合算しない。

'''+table(v)
 out+=f"\n\n事前gate: **{sel['gate']}**、主比較でHの非pass6ply完成率{sel['depth6_fraction']*100:.2f}%。gate/標本数は変更していない。\n"
 if (R/'confirmation-summary.json').exists():
  c=json.load(open(R/'confirmation-summary.json'));out+='\n## 独立確認\n新32prefix×先後×3seed、各192局。\n\n'+table(c)+'\n'
  for k,x in c.items():out+=f"\n{k} 2比較補正97.5% point CI: {x['point_ci_bonferroni2'][0]*100:.2f}–{x['point_ci_bonferroni2'][1]*100:.2f}%。\n"
  out+=f"\n主H/Eの事前改善条件達成: {c['H-E']['point_ci_bonferroni2'][0]>.5 and c['H-E']['difference']>0}。\n"
 else:out+='\n事前gate未達のため独立確認は実施しない。未使用32prefix bankは保持し、救済スイープはしない。\n'
 out+='''
## 候補推定値と実継続WDLの診断
新6元ゲーム、各1局面（12/20/28ply各2）、seed1751401のみ。共通top8（greedy保持）候補の固定4ply幅4のNN backed-up値と、その候補を置いて終局まで実プレイしたWDLを比較。継続分布は固定greedyと固定E2幅4の完成探索。E2はもう一つの探索継続だがE4より強いとは仮定しない。共通乱数2繰返し、同ゲームの候補・繰返しは相関し独立標本ではない。gen3重みは変更しない。

6局面中、root4→8で推定値が上昇して選択が変わったのは1局面（推定+0.017程度）のみ。その局面で両継続WDLの増加は0。他5局面は推定選択不変。これだけで「value誤差が主因」とは断定できない。離散WDLの同値が多い、小標本、継続方策依存、ゲーム段階分布が限られる。推定値はminimax真値ではなく、継続結果も最善応手を保証しない。人間棋譜は未取得。

各候補の値/段階/終局得点/全棋譜、元ゲームID、予測MSE/符号精度/予測レンジはdiagnostic.json。棋譜追加保存の再実行は同じ固定データでsummary一致を確認し、条件変更には使っていない。

## 再現
```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m pytest -q tests/test_ppo_deeper.py
python scripts/pilot_deeper.py
python scripts/diagnose_deeper_value.py
# 過去bankと訓練index参照が必要。既存評価bankを成績で再生成しない。
python scripts/run_deeper.py prepare
python scripts/run_deeper.py run
python scripts/run_deeper.py audit
python scripts/audit_deeper.py
python -m blokus.ppo_deeper_play --method H --budget-ms 500 --human-seat 0
```

研究Hは本番未適用。mainには検証済みEのみ。独立棋力確認なしに深さ増を強化成功と呼ばない。研究の選択/モデルhash/全棋譜/seed別・席別成績/冷起動warmup/超過と完成深さは各JSONに保存。過去巨大訓練履歴はSHA参照のみ。
'''
 (R/'REPORT.md').write_text(out)
if __name__=='__main__':main()
