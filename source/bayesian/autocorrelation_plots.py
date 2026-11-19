#!/usr/bin/env python3

"""Create autocorrelation diagnostic plots."""

from matplotlib import pyplot
import numpy

from general_purpose_python_modules.emcee_autocorrelation import\
    autocorr_gw2010,\
    average_autocorr,\
    max_likelihood_autocorr

from bayesian.visualize_emcee import get_plot_data

if __name__ == '__main__':
    system_name, blobs, log_prob = get_plot_data(
#        'samples/NGC6819_66016_mcmc_powerlawlgQ_samples.h5',
        'samples/NGC188_4618_mcmc_powerlawlgQ_samples.h5',
        0,
        [('initial_eccentricity', 0.8)]
    )
    print('Blobs shape: ' + repr(blobs.shape))

    plot_nsamples = 2**numpy.arange(3, 8)

    pyplot.plot(numpy.arange(blobs['lgQ_min'].shape[0]),
                blobs['lgQ_min'])
    pyplot.show()

    chains = blobs['lgQ_min'].T
    gw2010 = numpy.empty(len(plot_nsamples))
    new = numpy.empty(len(plot_nsamples))
    ml = max_likelihood_autocorr(chains, 1)
    print('ML: ' + repr(ml))
    for ind, eval_nsamples in enumerate(plot_nsamples):
        gw2010[ind] = autocorr_gw2010(chains[:, :eval_nsamples])
        new[ind] = average_autocorr(chains[:, :eval_nsamples])

    pyplot.loglog(plot_nsamples, gw2010, 'o-', label='G&W 2010')
    pyplot.loglog(plot_nsamples, new, 'o-', label='average')
    pyplot.loglog(plot_nsamples, plot_nsamples / 50.0, '--', label='N/50')
    pyplot.show()
