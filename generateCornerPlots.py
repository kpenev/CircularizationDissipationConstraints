import logging
import os
from datetime import datetime
import numpy
import emcee
import corner
import matplotlib.pyplot as plt
from pypdf import PdfMerger
import EnvelopeEccentricityDistribution

def generate_corner_plot(system):
    mcmc_progress_file_name = '/work/08529/mmmahmud/p0andmcmc/%(system)s/%(system)s_mcmc_progress.h5' % dict(system=system)
    backend = emcee.backends.HDFBackend(mcmc_progress_file_name)
    blobs = backend.get_blobs(flat=True)
    figure = corner.corner(blobs, labels=['\n\nM*',  # Mass of the parent star
                                          '\n\n     age',  # Age
                                          '\n\nRp',  # Planetary radius
                                          '\n\n      Fe/H_*',  # Stellar Metallicity
                                          '\n\nMp',  # Planetary mass
                                          '\n\n         initSpin*',  # Initial stellar spin
                                          '\n\n    lgQpl',
                                          '\n\n   tidal \n   break \n   point',
                                          '\n\n     alpha',
                                          '\n\n    lgQst',
                                          '\n\n       e_now',  # Present eccentricity
                                          '\n\nlog(f(e_now))'],  # log likelihood of present eccentricity
                           quantiles=[0.16, 0.5, 0.84],title_fmt=".2E",
                           show_titles=True, title_kwargs={"fontsize": 8.45}, label_kwargs={"fontsize": 8})
    figure.subplots_adjust(left=0.045,right=0.98, top=0.975, bottom = 0.085)
    for ax in figure.get_axes():
        ax.tick_params(axis='both', labelsize=8)
    figure.suptitle(system)
    return figure
    #figfilename = "/work/08529/mmmahmud/corner_plots/%(system)s_corner_plot.pdf" % dict(system=system)
    #figure.savefig(figfilename, bbox_inches='tight')
    #all_corner_plots_filename = "/work/08529/mmmahmud/corner_plots/all_corner_plots.pdf"
    #all_corner_plots_file_exists = os.path.exists(all_corner_plots_filename)
    #merger = PdfMerger()
    #if all_corner_plots_file_exists:
        #merger.append(all_corner_plots_filename)
        #merger.append(figfilename)
        #merger.write(all_corner_plots_filename)
        #merger.close()
    #else:
        #merger.append(figfilename)
        #merger.write(all_corner_plots_filename)
        #merger.close()

if __name__ == '__main__':
    envelope_eccentricity_distribution_instance = EnvelopeEccentricityDistribution.EnvelopeEccentricityDistribution()
    index = envelope_eccentricity_distribution_instance.print_properties_of_binary_systems_satisfying_constraints()
    merger = PdfMerger()
    i = 0
    systems = []
    while i<len(index):
        measured_values, standard_deviations, system_name = envelope_eccentricity_distribution_instance.properties_of_ith_binary_system_if_satisfies_constraints(index[i])
        #if not system_name in systems:
        #systems.append(system_name)
        if i !=13:
            figure = generate_corner_plot(system_name)
            figfilename = "/work/08529/mmmahmud/corner_plots/%(system)s_corner_plot.pdf" % dict(system=system_name)
            merger.append(figfilename)
        i = i+1
    all_corner_plots_filename = "/work/08529/mmmahmud/corner_plots/all_corner_plotS.pdf"
    merger.write(all_corner_plots_filename)
    merger.close()
