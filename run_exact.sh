#!/bin/sh
# Option 3: (A) the paper's exact exploration schedule at v3 size, then (B) the paper's full model size.
# Each run is evaluated on the 30 held-out deployments, best and final checkpoints.
cd "$(dirname "$0")"
PY=.venv/bin/python
echo "A start $(date -u +%FT%TZ)" >> logs/exact_status.txt
$PY train.py --arch informer --tag madii_exact_eps --episodes 1000 --il-episodes 300 --experts mixed \
    --eps-max 1.0 --eps-min 0.1 --threads 4 > logs/madii_exact_eps.out 2>&1
echo "A trained $(date -u +%FT%TZ) exit $?" >> logs/exact_status.txt
$PY evaluate.py --ckpt checkpoints/madii_exact_eps_best.pt checkpoints/madii_exact_eps.pt --seeds 30 > logs/eval_exact_eps.txt 2>&1
cp results_eval.json logs/results_eval_exact_eps.json
echo "A evaluated $(date -u +%FT%TZ)" >> logs/exact_status.txt
echo "B start $(date -u +%FT%TZ)" >> logs/exact_status.txt
$PY train.py --arch informer --tag madii_papersize --episodes 1000 --il-episodes 300 --experts mixed \
    --d-model 1024 --d-ff 4096 --heads 8 --eps-max 0.1 --eps-min 0.01 --threads 4 > logs/madii_papersize.out 2>&1
echo "B trained $(date -u +%FT%TZ) exit $?" >> logs/exact_status.txt
$PY evaluate.py --ckpt checkpoints/madii_papersize_best.pt checkpoints/madii_papersize.pt --seeds 30 > logs/eval_papersize.txt 2>&1
cp results_eval.json logs/results_eval_papersize.json
git checkout -- results_eval.json 2>/dev/null
echo "B evaluated $(date -u +%FT%TZ)" >> logs/exact_status.txt
