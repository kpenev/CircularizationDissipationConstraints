#!/bin/bash

rm burnin_log.txt

for f in NGC188_*_mcmc_samples.h5 NGC6819_*_mcmc_samples.h5; do
    SYS=${f%_mcmc_samples.h5}

    if [ "$SYS" == "NGC188_5601" ]; then
        BURNIN=65
    elif [ \
        "$SYS" == "NGC188_4618"\
        -o "$SYS" == "NGC188_5733"\
        -o "$SYS" == "NGC188_4904"\
    ]; then
        BURNIN=55
    elif [\
        "$SYS" == "NGC188_5463"\
        -o "$SYS" == "NGC188_4289"\
        -o "$SYS" == "NGC188_6292"\
    ]; then
        BURNIN=40
    elif [ "$SYS" == "NGC188_6171" ]; then
        BURNIN=30
    else
        BURNIN=0
    fi

    bayesian/visualize_emcee.py $f\
        --trace-plot-fname ${SYS}_traces.eps\
        --max-traces-per-plot 16

    bayesian/visualize_emcee.py $f\
        --corner-plot-fname ${SYS}_corner.eps\
        --burn-in $BURNIN

    echo "constQ: $SYS $BURNIN" >> burnin_log.txt

done

for f in NGC188_*_powerlawlgQ_samples.h5; do

    SYS=${f%_powerlawlgQ_samples.h5}

    BURNIN=0
    if [ "$SYS" == "NGC188_4618" ] ; then
        BURNIN=50
    elif [ "$SYS" == "NGC188_4904" ]; then
        BURNIN=30
    fi

    bayesian/visualize_emcee.py $f\
        --trace-plot-fname ${SYS}_powerlawlgQ_traces.eps\
        --max-traces-per-plot 16

    bayesian/visualize_emcee.py $f\
        --corner-plot-fname ${SYS}_powerlawlgQ_corner.eps\
        --burn-in $BURNIN

    echo "powerlawQ: $SYS $BURNIN" >> burnin_log.txt

done

bayesian/visualize_emcee.py\
    NGC188_4618_mcmc_powerlawlgQ_samples.h5\
    NGC188_4904_mcmc_powerlawlgQ_samples.h5\
    --frequency-dependence-plot NGC188_4618_4904_frequency_dependence.png\
    --burn-in 30\
    --plot-confidence 0.9544997361036416 0.6826894921370859\
    --frequency-dependence-plot-no-lines
