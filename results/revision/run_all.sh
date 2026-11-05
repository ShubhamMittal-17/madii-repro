#!/bin/sh
# Test-phase experiments with the validation-chosen settings (revision/routers.py TUNED).
set -e
for e in main trace lossy kappa grid; do
  .venv/bin/python -m revision.run --exp $e --workers 4 > results/revision/$e.log 2>&1
done
OMP_NUM_THREADS=1 .venv/bin/python -m revision.scaling > results/revision/scaling.log 2>&1
