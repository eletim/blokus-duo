# Follow-up requested after checkpoint selection

Preserve preregistered selection and 1920 matches. No retraining or extra matches.

After timed evaluation ends:
- Report all seed selections; C-1751401 selects epoch 0.
- Verify saved C epoch-0/export tensors against original gen4 shared/value tensors; verify map_value is zero. Occupancy heads are random initialized; zero map-to-value projection means no learned prediction contribution. Confirm outputs on saved current states. Do not claim this seed tests effective learned map use.
- Existing selection.json contains full dev value MSE and B/C empty-cell CE/Brier/class/stage curves, but training script did NOT log train metrics. State this explicitly. Reconstruct train metrics from saved epoch checkpoints and fixed train data only AFTER matches, with no optimizer updates or selection changes; label as post-hoc diagnostic reconstruction, not original training logs.
- Plot train/dev value MSE and empty-cell CE/Brier by epoch/seed. Distinguish observed early dev minima from hypotheses of overfitting or loss competition; neither cause is established by these curves alone.
- Preserve unchanged models/optimizer/RNG and fixed test/independent evaluation.
