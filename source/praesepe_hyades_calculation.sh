#!/bin/bash

ulimit -S -m $((2 * 1024 * 1024))

nohup ./calculate_e_Q_grid.py \
    --num-parallel-processes 20 \
    --initial-eccentricity 0.5 \
    --use-binary-stars 'praesepe/hyades' \
    --progress-pickle progress_praesepe_hyades_einit0.5.pickle \
    --fallback-initial-eccentricity 0.49 \
    --fallback-initial-eccentricity 0.51 \
    > \
    praesepe_hyades_einit0.5_with_fallback.outerr \
    2>&1
