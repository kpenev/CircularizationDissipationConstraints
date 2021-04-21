#!/bin/bash

if [ "$1" == "all" ]; then
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

        bayesian/visualize_emcee.py ${f}\
            --trace-plot-fname ${SYS}_traces.eps\
            --max-traces-per-plot 16

        bayesian/visualize_emcee.py ${f}:${BURNIN}\
            --corner-plot-fname ${SYS}_corner.eps\

        echo "constQ: $SYS $BURNIN" >> burnin_log.txt

    done

    for f in NGC188_*_powerlawlgQ_samples.h5 NGC6819_59003; do

        SYS=${f%_mcmc_powerlawlgQ_samples.h5}

        if [ "$SYS" == "NGC188_4618" ] ; then
            BURNIN=50
        elif [ "$SYS" == "NGC188_4904" ]; then
            BURNIN=30
        else
            BURNIN=0
        fi

        bayesian/visualize_emcee.py $f\
            --trace-plot-fname ${SYS}_powerlawlgQ_traces.eps\
            --max-traces-per-plot 16

        bayesian/visualize_emcee.py $f:${BURNIN}\
            --corner-plot-fname ${SYS}_powerlawlgQ_corner.eps\

        echo "powerlawQ: $SYS $BURNIN" >> burnin_log.txt
    done
fi

#NGC188_4965_mcmc_powerlawlgQ_samples.h5:30\
#NGC188_5797_mcmc_powerlawlgQ_samples.h5:30\
bayesian/visualize_emcee.py\
    NGC188_4618_mcmc_powerlawlgQ_samples.h5:60\
    NGC188_4904_mcmc_powerlawlgQ_samples.h5:60\
    NGC188_5015_mcmc_powerlawlgQ_samples.h5:60\
    NGC188_5601_mcmc_powerlawlgQ_samples.h5:30\
    NGC188_5738_mcmc_powerlawlgQ_samples.h5:30\
    NGC188_6171_mcmc_powerlawlgQ_samples.h5:30\
    --log-x\
    --frequency-dependence-plot frequency_dependent_constraints.png\
    --plot-confidence 0.9544997361036416 \
    --frequency-dependence-plot-no-lines

for SYS in NGC188_4618 NGC188_5601 NGC188_6171 NGC188_4904; do
    if [ "$SYS" == "NGC188_4618" -o "$SYS" == "NGC188_4904" ]; then
        BURNIN=60
    else
        BURNIN=30
    fi
    bayesian/visualize_emcee.py\
        ${SYS}_mcmc_powerlawlgQ_samples.h5:${BURNIN}\
        ${SYS}_mcmc_samples.h5:60\
        --log-x\
        --frequency-dependence-plot ${SYS}_comparison.png\
        --plot-confidence 0.6826894921370859\
        --frequency-dependence-plot-no-lines
done

for xexpr in 'orbital_period' 'orbital_period/2.0'; do 
    fname_tag=$(echo "$xexpr"|sed -e 's%/%_div_%g')
    bayesian/visualize_emcee.py\
        NGC188_4080_mcmc_samples.h5:60\
        NGC188_4289_mcmc_samples.h5:60\
        NGC188_4618_mcmc_samples.h5:60\
        NGC188_4904_mcmc_samples.h5:60\
        NGC188_4965_mcmc_samples.h5:60\
        NGC188_5040_mcmc_samples.h5:60\
        NGC188_5463_mcmc_samples.h5:60\
        NGC188_5601_mcmc_samples.h5:60\
        NGC188_5647_mcmc_samples.h5:60\
        NGC188_5733_mcmc_samples.h5:60\
        NGC188_5738_mcmc_samples.h5:60\
        NGC188_5797_mcmc_samples.h5:60\
        NGC188_6171_mcmc_samples.h5:60\
        NGC188_6292_mcmc_samples.h5:60\
        NGC188_880_mcmc_samples.h5:60\
        NGC6819_57004_mcmc_samples.h5:60\
        NGC6819_59003_mcmc_samples.h5:60\
        --errorbar-plot constQ_errorbar_vs_${fname_tag}.eps "$xexpr" 'lgQ_min'\
        --plot-confidence 0.9544997361036416 0.6826894921370859
done
