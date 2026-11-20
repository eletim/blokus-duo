"""Write a reviewable PR body from final audited artifacts; never publishes itself."""
import json
from pathlib import Path
R=Path('artifacts/value-placement-24576')
def read(n):return json.loads((R/n).read_text())
def main():
 assert read('audit.json')['pass_'] and read('clean-check.json')['pass_']
 phase='confirmation' if read('selection.json')['passed'] else 'validation';d=read(phase+'-summary.json');passed=read('decision.json')['passed'] if phase=='confirmation' else False
 text=f'''This draft research PR now completes the requested pipeline: **async512 GPU self-play generation → real CUDA learning on 24,576 games → new CPU timed matches → independent audits**. It changes neither main, the public Web, nor Frozen gen3. The final predeclared adoption gate is **{'passed' if passed else 'not met'}**; execution completion and strength adoption are separate.

New 18,432 training games plus 192 separate development games were generated with the fixed Frozen gen3 teacher, width 4 and completed 6 ply. They extend the existing 6,144 training games to 24,576 (673,764 states). The new generation took 55.77 minutes; real six-model CUDA training took 20.71 minutes. No repeat speed study or new teacher was substituted.

A adds 252 legal-placement mobility features to the value model; C retains the original value structure. Both use identical data/order/update budgets and three fixed seeds. The move policy remains separate, unchanged gen3. Targets are actual terminal WDL, with the previous 0.1 original-value penalty. Adam 1e-4, batch 64, float32, 16 epochs: **168,448 optimizer updates per model, 1,010,688 total**. All six selected exports use common epoch 6 / 63,168 updates, chosen only by mean new-development MSE. Latest optimizer/RNG checkpoints and all curves are retained. Later epochs reduce training loss but worsen development error.

## New strength results

{phase.capitalize()} only; validation and independent confirmation are not pooled. Each confirmation comparison has 32 new prefixes × two seats × three fixed seeds = 192 games. Win=1, draw=0.5. Intervals cluster all seats/seeds by prefix; the table shows Bonferroni three-comparison 98.3333% intervals. Ordinary 95% and seed/seat results are in the report.

| New A versus | W–D–L | Point | Adjusted CI | Mean game-score difference |
|---|---:|---:|---:|---:|
'''
 for pair,label in [('AQ','prior successful 3,072-game A'),('AF','Frozen gen3+H'),('AC','same-data original-structure C')]:
  v=d[pair];lo,hi=v['point_adjusted_ci'];text+=f"| {label} | {v['wins']}–{v['draws']}–{v['losses']} | {v['point']*100:.2f}% | {lo*100:.2f}–{hi*100:.2f}% | {v['difference']:+.2f} |\n"
 text+='''
All final matches use CPU native, one Torch thread/core 0, width 4 / max 6 / 500 ms, with generation/training/heavy audits excluded from timing. Actual thinking time, maximum delay, depth completion, fallback and cold/warm preparation are reported. Same time ceiling does not imply equal consumed time. No claim about strength against humans or all future seeds is made.

The old 6,144 games used CPU time-limited teachers; new GPU games require completed 6 ply and known near-tie rounding can change moves. The old A uses epoch16 whereas the new A uses dev-selected epoch6; their optimizer-update counts also differ. This is a documented data-expansion experiment, not an isolated causal proof that game count alone causes improvement or global CPU/GPU bit equivalence.

## Verification and reuse

- Every new generated game and every evaluation game independently replayed for coordinate legality, forced passes, terminal bonuses/scores and result arithmetic. Data/model/code hashes, real-turn targets, game weights, training/development/evaluation state separation and original training index checked.
- Representative A/C CPU/CUDA fresh-process continuation matches uninterrupted weights, Adam and RNG bitwise, including continuation from actual trained checkpoints. Exports show real weight changes and retain the original policy.
- All six exports pass CPU/CUDA value tolerance 2e-5. A clean history-free Git checkout with CUDA hidden passes 127 tests (6 skipped) and CPU CLI games, including scripted human seats and independent replay. This uses the installed CUDA-capable wheel with CUDA hidden, not a fresh CPU-only wheel installation.
- New game archives preserve source bytes and hashes; previous large histories are referenced, not copied. Inference needs only code, original gen3 and one new value export, not training histories/indexes. Resume checkpoints are separate.

Reports, curves and commands:
- [Final 24,576-game report](artifacts/value-placement-24576/REPORT.md)
- [Preregistered finite plan](artifacts/value-placement-24576/PLAN.md)
- [Generation/training/restart and CPU play commands](artifacts/value-placement-24576/REPRODUCE.md)
- [Real CUDA learning curves](artifacts/value-placement-24576/learning-curves.png)
- [Independent strength results](artifacts/value-placement-24576/strength-points.png)

## Preserved earlier work

The verified async512 generator previously achieved about 2.72× CPU8-native throughput on matched completed-six-ply work. This uses eight workers plus a coordinator and is not a GPU-kernel-only or equal-total-CPU claim. Earlier synchronous RPC and 64-cohort failures remain unchanged. The prior 3,072-game A remains separately documented; the 6,144-game follow-up did not establish improvement over it. Nothing here silently replaces those records.

- [Async512 speed, correctness and limits](artifacts/gpu-async512/REPORT.md)
- [64-frontier experiment](artifacts/gpu-frontier/REPORT.md)
- [Synchronous RPC experiment](artifacts/gpu-generation/REPORT.md)
- [Prior 3,072-game placement-value study](artifacts/value-placement-scale/REPORT.md)
- [6,144-game follow-up](artifacts/value-placement-6144/REPORT.md)

No merge, deployment, public-default change or Frozen gen3 overwrite is included. This PR remains draft for review.
'''
 (R/'PR-body.md').write_text(text)
if __name__=='__main__':main()
