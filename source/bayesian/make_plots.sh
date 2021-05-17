#!/bin/bash

if [ "$1" == "all" ]; then
    rm burnin_log.txt

    for f in NGC188_*_mcmc_samples.h5 NGC6819_*_mcmc_samples.h5; do
        SYS=${f%_mcmc_samples.h5}

        if [ "$SYS" == "NGC188_6171" -o "$SYS" == "NGC188_6292" ] ; then
            BURNIN=100
        elif [ "$SYS" == "NGC188_5733" ] ; then
            BURNIN=75
        elif [ \
            "$SYS" == "NGC188_5601"\
            -o "$SYS" == "NGC188_4289"\
        ]; then
            BURNIN=65
        elif [ \
            "$SYS" == "NGC188_4618"\
            -o "$SYS" == "NGC188_4904"\
            -o "$SYS" == "NGC188_4965"\
            -o "$SYS" == "NGC6819_57004"\
        ]; then
            BURNIN=60
        elif [\
            "$SYS" == "NGC188_5463"\
        ]; then
            BURNIN=40
        else
            BURNIN=30
        fi

        bayesian/visualize_emcee.py ${f}\
            --trace-plot-fname ${SYS}_traces.eps\
            --max-traces-per-plot 16

        bayesian/visualize_emcee.py ${f}:${BURNIN}\
            --corner-plot-fname ${SYS}_corner.eps\

        echo "constQ: $SYS $BURNIN" >> burnin_log.txt

    done

    for f in NGC*_powerlawlgQ_samples.h5 ; do

        SYS=${f%_mcmc_powerlawlgQ_samples.h5}

        if [ "$SYS" == "NGC188_4618" -o "$SYS" == "NGC188_5015" ] ; then
            BURNIN=125
        elif [ "$SYS" == "NGC188_6171" ] ; then
            BURNIN=100
        elif [ "$SYS" == "NGC188_8775" ] ; then
            BURNIN=75
        elif [ "$SYS" == "NGC188_4904" -o "$SYS" == "NGC188_5733" ]; then
            BURNIN=50
        elif [ "$SYS" == "NGC6819_57004" ] ; then
            BURNIN=45
        elif [ "$SYS" == "NGC188_4289" ] ; then
            BURNIN=40
        else
            BURNIN=30
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
    NGC188_4289_mcmc_powerlawlgQ_samples.h5:40\
    NGC188_4618_mcmc_powerlawlgQ_samples.h5:125\
    NGC188_4904_mcmc_powerlawlgQ_samples.h5:50\
    NGC188_4965_mcmc_powerlawlgQ_samples.h5:30\
    NGC188_5015_mcmc_powerlawlgQ_samples.h5:30\
    NGC188_5463_mcmc_powerlawlgQ_samples.h5:30\
    NGC188_5601_mcmc_powerlawlgQ_samples.h5:30\
    NGC188_5733_mcmc_powerlawlgQ_samples.h5:50\
    NGC188_5738_mcmc_powerlawlgQ_samples.h5:30\
    NGC188_5797_mcmc_powerlawlgQ_samples.h5:30\
    NGC188_6171_mcmc_powerlawlgQ_samples.h5:100\
    NGC188_6292_mcmc_powerlawlgQ_samples.h5:30\
    NGC188_8775_mcmc_powerlawlgQ_samples.h5:75\
    NGC6819_57004_mcmc_powerlawlgQ_samples.h5:45\
    NGC6819_59003_mcmc_powerlawlgQ_samples.h5:30\
    NGC6819_66004_mcmc_powerlawlgQ_samples.h5:30\
    --log-x\
    --frequency-dependence-plot frequency_dependent_constraints.png\
    --plot-confidence 0.9544997361036416 \
    --frequency-dependence-plot-no-lines \
    --frequency-dependence-bounds

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
