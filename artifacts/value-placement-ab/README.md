# Remaining-piece-aware value research

This branch adds two experimental value architectures and a same-data control. It does not replace production models or change the default search behavior.

- **A (explicit legal mobility):** exact per-player/per-piece availability, placement count, and four-bin spatial distribution. These 252 features are projected into the value hidden layer.
- **B (early conditioning):** remaining-piece indicators condition both convolution layers through FiLM, before board compression.
- **C (retraining control):** original value architecture retrained on the same examples with the same optimizer schedule.
- **F (frozen reference):** unchanged gen3 plus existing width4 / maximum6ply search.

`PlacementValue` owns a separate copy of the value trunk. The `PPOAgent` used to rank candidate moves remains unchanged at every search node. `DeepSearch(..., value_evaluator=...)` is opt-in; omitted argument follows the original implementation. Completed-depth timeout behavior and actual terminal WDL are retained.

## Reproduce

From the repository root, with Python3.12, the project dependencies and a C compiler (or the existing ziglang build fallback):

```bash
python -m blokus.ppo_native build
python -m pytest -q tests/test_placement_value.py tests/test_ppo_deeper.py
PYTHONPATH=.:scripts python scripts/check_placement_resume.py
PYTHONPATH=.:scripts python scripts/audit_placement.py
```

The audit requires the original gen3 training-state index. It is deliberately not copied into Git:

```bash
git fetch origin research/ppo-ragged-update
PYTHONPATH=. python scripts/restore_placement_index.py
```

The reconstruction checks the immutable artifacts and final 27,533,566-state hash. It needs the repository's full referenced history, RAM for sorting state hashes and about881MB for the resulting numpy file.

Generation and evaluation refuse to overwrite existing result files. To rerun the full experiment, use a separate checkout and move its `artifacts/value-placement-ab` result directory aside, retaining `PLAN.md` for reference. The sequence is:

```bash
PYTHONPATH=.:scripts python scripts/placement_experiment.py generate
PYTHONPATH=.:scripts python scripts/placement_experiment.py train
PYTHONPATH=.:scripts python scripts/check_placement_resume.py
PYTHONPATH=.:scripts python scripts/placement_experiment.py validation
PYTHONPATH=.:scripts python scripts/placement_experiment.py fixed
PYTHONPATH=.:scripts python scripts/placement_experiment.py select
# Only if selection.json names A or B:
PYTHONPATH=.:scripts python scripts/placement_experiment.py confirmation
PYTHONPATH=.:scripts python scripts/audit_placement.py
```

Do not run training or audits concurrently with timed evaluation. Exact wall-clock-limited outcomes can vary with hardware/load; all completed games and inference exports from this run are retained.

To use one research evaluator from Python:

```python
from scripts.placement_experiment import agent
from blokus.ppo_policy import PPOAgent
policy = PPOAgent(1751401, 'cpu', backend='native')
search = agent('A', policy, budget=500, depth=6)  # A/B/C/F
# move = search.act(state, random.Random(123))
```

These exports are separate value models, not drop-in PPO policy files. The original gen3 files and Frozen tag remain the benchmark. See `PLAN.md` for fixed hypotheses, gates and limitations, and `REPORT.md` for measured results.
