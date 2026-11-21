#!/bin/bash

nohup ./final_initial_eccentricity_dependence.py \
    --primary-reference-dissipation 1e-7 31.41592653589793 0.0 '3.0' \
    --secondary-reference-dissipation 1e-7 31.41592653589793 0.0 '3.0' \
    --secondary-mass 1 \
    --disk-lock-frequency 1.0 \
    --disk-dissipation-age 0.02 \
    --final-age 5.0 \
    --precision 1e-5 \
    --logging-verbosity debug \
    > \
    final_initial_eccentricity_dependence_pwr-3_break_20d.out \
    2>&1 \
    &
