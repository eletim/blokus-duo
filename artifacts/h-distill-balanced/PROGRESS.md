# Progress
Paired-seat control in progress. PLAN frozen before main labels/training/evaluation. Commit e6e29a0 records initial plan and pilot.
Focused tests:6PASS. Separate-process CUDA resume12 vs5+7PASS. Pilot12games,20.135seconds,62teacherlabels.
Main data generation:256train+96dev games perseed,3seeds. Equal A/B192labels/seed; stage×actualturn quota and game-split assertions precede learning. Nonpolicy weights and published service unchanged.
Timing/validation starts only after generation+training complete. No retrospective gate changes.

First seed train bank complete:256games; stage12/20/28 eligible pairs256/248/129, all quotas64 met. Remaining seeds in progress.
Second seed train bank complete:256games; eligible pairs256/249/123, all quotas64 met. Third seed generation underway.

Data/training complete. Each A/B:192labels/seed,96updates; unique selected games93/96/93. All split overlaps0. Export frozen tensor/value CPU/CUDA checksPASS; real-data separate-process20 vs7+13 across epoch boundary botharmsPASS;15focused/rule testsPASS. New validation12 and conditional confirmation32 banks/models fixed before timing.
