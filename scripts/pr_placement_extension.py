"""Build final PR body from audited artifacts; caller publishes with --body-file."""
import json
from pathlib import Path
R=Path('artifacts/value-placement-49152')
def read(p):return json.loads((R/p).read_text())
def main():
 audit=read('audit.json');assert audit['pass_'] and read('n49152/clean-check.json')['pass_'];out=read('n49152/confirmation-summary.json');outcome=read('final-outcome.json');gen=read('generation-telemetry.json')
 text=f'''This draft research PR now includes the requested finite data-scale follow-up: **new GPU self-play generation → real CUDA learning at 36,864 and 49,152 training games → independent CPU timed evaluation → audits and history-free exports**. Main, the production Web, and Frozen gen3 remain unchanged. The final predeclared strength-adoption gate is **{'passed' if outcome['passed'] else 'not met'}**; completed execution is separate from adoption.

The fixed Frozen gen3 teacher generated 24,576 new training games plus 192 separate development games, using the verified async512 / batch128 / eight-worker+coordinator implementation, width4 and completed6ply. Generation took {gen['outer_seconds']/60:.2f} minutes. The old 24,576 training games are referenced and retained, not duplicated or replaced; prior development, speed banks, strength tests and human games are not training data.

A has 252 legal-placement mobility inputs for value evaluation; C uses the original value structure. Each stage trains A/C at all three fixed seeds, freshly initialized from gen3, on the same data/order/update budget. The separate move policy stays frozen. Actual terminal WDL MSE +0.1 original-value anchoring, Adam1e-4, batch64, float32, maximum16epochs are unchanged. Across the 12 models, **{audit['total_real_updates']:,} actual optimizer updates** were performed. Per-stage common checkpoint selection uses only mean new-development MSE across all six models, never playing-strength outcomes.

| Total training games | Common dev-selected epoch | Updates/model at epoch16 | CUDA training minutes |
|---:|---:|---:|---:|
'''
 for n in (36864,49152):
  s=audit['stage_selections'][str(n)];t=read(f'n{n}/training-time.json');text+=f"| {n:,} | {s['epoch']} | {s['updates_per_model']:,} | {t['seconds']/60:.2f} |\n"
 text+='''
Two distinct exploratory stage banks (8 prefixes each, both seats/all seeds) compare each new A with the immediately preceding data stage and Frozen H. The final candidate was always fixed to49,152-game A, not chosen by those match results. Before any new training or match outcomes, the confirmation allocation was explicitly amended from32 to64prefixes to use the five-hour budget for precision; the original plan remains in history. No outcome-dependent extension or rescue was used.

## Independent confirmation

64 new prefixes × two seats × three fixed seeds =384 games/comparison,1,536 total, separate from192 stage games. Prefix-cluster bootstrap20,000 draws; the table uses Bonferroni four-comparison98.75% point intervals. Win1, draw0.5. Ordinary95%, score intervals, all seeds/seats and timing distributions are retained.

| Fixed49,152 A versus | W–D–L | Point | Family-adjusted CI | Mean game-score difference |
|---|---:|---:|---:|---:|
'''
 for p,label in [('AQ','old successful3,072-game A'),('AR','previous24,576-game A'),('AF','Frozen gen3+H'),('AC','same49,152-game C')]:
  v=out[p];lo,hi=v['point_adjusted_ci'];text+=f"| {label} | {v['wins']}–{v['draws']}–{v['losses']} | {100*v['point']:.2f}% | {100*lo:.2f}–{100*hi:.2f}% | {v['difference']:+.2f} |\n"
 text+=f"\nThe predeclared adjacent-stage flattening criterion is **{'met' if outcome['flattening_criterion_met'] else 'not met'}**. Neither a single nonsignificant result nor development loss is a general playing-strength ceiling. {'The new A is a review candidate, not deployed.' if outcome['passed'] else 'Retain the prior successful candidate; the required all-comparison promotion gate was not established.'}\n"
 text+='''
All final matches remain CPU native / one Torch thread / core0 / width4 / max6 /500ms. Generation, learning and heavy audits do not overlap timed evaluation. Actual mean/p95/p99/max latency, completed-depth rate, fallback and warmup are reported; equal ceilings are not equal consumed wall time. Human strength and generalization beyond these fixed seeds/distributions remain unverified.

## Correctness and reproducibility

- Independently replayed all24,768 new generated games and1,728 new matches for legal moves, forced passes, terminal bonuses and scores. Checked actual-turn WDL, game-normalized weights, game/RNG/model/source hashes, training/dev/evaluation and historical/original-PPO state separation, and recomputed result/CI arithmetic.
- Old update equations and RNG/Adam restoration are unchanged; focused tests compare them with the verified trainer and test stage composition/opponent identity. Actual new checkpoints pass CPU/CUDA separate-process continuation checks. All12 exports pass CPU/native/Python vs CUDA value tolerance2e-5, actual weight-change and unchanged-policy checks.
- A clean history-free Git checkout with CUDA hidden runs relevant regression tests, CPU CLI games for all six final exports, and both scripted human seats; all eight CLI games are independently replayed. The installed CUDA-capable PyTorch wheel was hidden from CUDA; this is not a fresh CPU-only wheel installation.
- New trajectories are byte-verified small archives; old large histories are only SHA references. Latest optimizer/RNG checkpoints are distinct from inference exports. The CPU CLI needs originalgen3 plus one research value export, not historical datasets/indexes.
- HistoricalCPU500ms games and newGPU completed6ply games have documented provenance differences and near-tie floating-point move differences. Epoch/update counts and dev selection also differ from oldA16. No global bit-equivalence or pure game-count causal claim is made.

Reports and commands:
- [49,152-game finite plan](artifacts/value-placement-49152/PLAN.md)
- [Final report and all comparisons](artifacts/value-placement-49152/REPORT.md)
- [Reproduction, resume and CPU play](artifacts/value-placement-49152/REPRODUCE.md)
- [Actual CUDA learning curves](artifacts/value-placement-49152/learning-curves.png)
- [Exploratory data-stage trends](artifacts/value-placement-49152/stage-trends.png)
- [Independent confirmation and adjusted CI](artifacts/value-placement-49152/confirmation-points.png)

## Preserved previous results

The prior24,576-game experiment trained1,010,688updates, beat FrozenH and matched-data C, but did not establish improvement over old3,072A under its three-comparison gate. Its plan, samples, models and failed promotion remain unchanged. The earlier6,144 follow-up likewise did not establish an increment. The successful3,072 candidate remains available. The async512 speed result is retained separately from strength results, together with the failed synchronousRPC/64-cohort approaches.

- [Previous24,576-game report](artifacts/value-placement-24576/REPORT.md)
- [Verified async512 generation](artifacts/gpu-async512/REPORT.md)
- [Prior3,072-game value result](artifacts/value-placement-scale/REPORT.md)
- [6,144-game follow-up](artifacts/value-placement-6144/REPORT.md)
- [64-frontier and limitations](artifacts/gpu-frontier/REPORT.md)
- [SynchronousRPC failure](artifacts/gpu-generation/REPORT.md)

No merge, deployment, public-default change or Frozen gen3 overwrite. The PR remains draft for review.
'''
 (R/'PR-body.md').write_text(text)
if __name__=='__main__':main()
