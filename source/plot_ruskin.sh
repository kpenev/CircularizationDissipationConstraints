#!/bin/bash

nohup \
    bayesian/plot_lgq_vs_period_constraints.py \
    --method spin \
    --samples-dir samples/spin/w19/converged/ \
    --download-from \
    --combined-constraint-period-range 1.0 50.0 \
    --lgQ-grid 5.0 10.0 300 \
    --constraint-validity-threshold 0.7 \
    --subplot-layout 10 5 \
    --overwrite-mrt \
    >\
    plot_ruskin.out \
    2>&1 &
