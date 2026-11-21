#!/usr/bin/env python3

"""Plot phase lag vs tidal frequency per mcee samples."""

import sys
from os.path import expanduser

from matplotlib import pyplot
from matplotlib.backends.backend_pdf import PdfPages
from configargparse import ArgumentParser, DefaultsFormatter
import numpy

from orbital_evolution.star_interface import EvolvingStar
from orbital_evolution.transformations import phase_lag

from bayesian.parse_command_line import use_parser
from bayesian.binary_utils import prepare_sampling_common
from bayesian.visualize_emcee import get_plot_data
from bayesian.log_likelihood_binary_stars import LogLikelihoodBinaryStars


sys.path.append(expanduser('~/projects/git/poet/scripts'))
from plot_phase_lag import generate_plots

def parse_command_line(config_fname='plot_sampled_lag.cfg'):
    """Avoid polluting global namespace."""

    parser = ArgumentParser(
        description='Plot phase lag vs  tidal frequency for a collection of '
        'emcee points',
        default_config_files=[config_fname],
        args_for_writing_out_config_file=['--generate-config-file'],
        args_for_setting_config_path=['--config-file', '-c'],
        formatter_class=DefaultsFormatter,
        ignore_unknown_config_file_keys=True
    )
    parser.add_argument(
        'samples_fname',
        help='The filename of the HDF5 file containing the generated samples '
        'to plot from.'
    )
    parser.add_argument(
        '--chain-condition',
        default=[],
        nargs=2,
        action='append',
        metavar=('ATTRIBUTE', 'VALUE'),
        help='Select which chain in the given file to plot by imposing a '
        'condition on on of the attributes of the chain group. The condition '
        'can be a single value in which case the attribute should have '
        'either just that value or be a pair of identical values. '
        'Alternatively, the condition can be multiple values that must match '
        'exactly in the order specified. By default, plots the first chain in '
        'the input file.'
    )
    parser.add_argument(
        '--sample',
        type=int,
        nargs=2,
        action='append',
        default=[],
        help='The iteration and walker indices to plot. Can be specified '
        'multiple times. Plots will be added as pages to the output PDF.'
    )
    parser.add_argument(
        '--plot-random-samples',
        type=int,
        default=0,
        help='Specify a number of randomly selected samples to plot, saved as '
        'pages of the output PDF file.'
    )
    parser.add_argument(
        '--spin-angular-velocity', '--wspin',
        type=float,
        default=1.0,
        help='The spin angular velocity to assume for creating the plot.'
    )
    parser.add_argument(
        '--tidal-angular-velocity', '--wtide',
        type=float,
        nargs=3,
        default=[-5.0, 5.0, 1000.0],
        help='Range and resolution of the tidal period grid for the plot.'
    )
    parser.add_argument(
        '--output', '-o',
        default='phase_lag_samples.pdf',
        help='The name of the PDF file to generate with the specified plots.'
    )
    return use_parser(parser,
                      sampling=False,
                      output=False)


def get_plot_samples(config):
    """Return the samples to plot."""

    samples = get_plot_data(
        config.samples_fname,
        0,
        config.chain_condition
    )[1]
    niterations, nwalkers = samples.shape
    selected_samples = numpy.empty(
        (len(config.sample) + config.plot_random_samples, 2),
        dtype=int
    )
    if config.sample:
        selected_samples[:len(config.sample)] = config.sample
    selected_samples[-config.plot_random_samples:, 0] = numpy.random.randint(
        niterations,
        size=config.plot_random_samples
    )
    selected_samples[-config.plot_random_samples:, 1] = numpy.random.randint(
        nwalkers,
        size=config.plot_random_samples
    )
    return samples[selected_samples[:, 0], selected_samples[:, 1]]


class DissipationGetter(LogLikelihoodBinaryStars):
    """A class for callables that generate dissipation config for a sample."""

    def calculate_log_likelihood(self, parameters, **other_args):
        assert False


    def __init__(self):
        """Prepare the class to use for generating dissipation configuration."""

        super().__init__(
            powerlaw_dissipation=True,
            interpolator=None,
            envelope_eccentricity=None,
            evolution_timeout=None,
            period_search_factor=None,
            scaled_period_guess=None
        )
        self._defaults = dict(
            lgQ_inertial_sharpness=10
        )


    #A little ugly but does the job
    #pylint: disable=arguments-differ
    def __call__(self, sample):
        """Return the dissipation configuration per given sample."""

        parameters = numpy.empty(len(self.parameter_order))
        for param_i, (param_name, _) in enumerate(
                self.parameter_order
        ):
            if param_name in sample.dtype.names:
                parameters[param_i] = sample[param_name]
            else:
                parameters[param_i] = self._defaults.get(param_name, numpy.nan)
        return self.get_dissipation(parameters)
    #pylint: enable=arguments-differ


def main(config):
    """Avoid polluting global namespace."""

    plot_samples = get_plot_samples(config)
    get_dissipation = DissipationGetter()
    star_interp = prepare_sampling_common(config)
    star = EvolvingStar(
        mass=1.0,
        metallicity=0.0,
        wind_strength=0.17,
        wind_saturation_frequency=10.0,
        diff_rot_coupling_timescale=5e-3,
        interpolator=star_interp,
    )


    param_values = dict(
        age=1.0,
        spin_frequency=config.spin_angular_velocity,
        orbital_multiplier=1,
        spin_multiplier=0,
        orbital_frequency=None
    )
    wtide = numpy.linspace(config.tidal_angular_velocity[0],
                           config.tidal_angular_velocity[1],
                           int(config.tidal_angular_velocity[2]))
    with PdfPages(config.output) as pdf:
        for sample in plot_samples:
            star.set_dissipation(zone_index=0,
                                 **get_dissipation(sample)['primary'])
            generate_plots(star, param_values, wtide, integrate=False)
            pyplot.legend()
            pyplot.suptitle(
                (
                    r'$\log_{{10}}Q_{{min}} = {lgQ:.3f}$, '
                    r'$P_{{br}} = {pbr:.3f}$, '
                    r'$\alpha = {alpha:.3f}$, '
                    r'$\gamma = {gamma:.3f}$'
                ).format(
                    lgQ=sample['lgQ_min'],
                    pbr=sample['lgQ_break_period'],
                    alpha=sample['lgQ_powerlaw'],
                    gamma=sample['lgQ_inertial_boost']
                )
            )
            pyplot.axhline(y=phase_lag(sample['lgQ_min']),
                           label=r'$\Delta_{max}$')
            pyplot.axvline(x=2.0 * numpy.pi / sample['lgQ_break_period'],
                           label=r'$\omega_{br}$')
            pyplot.axvspan(xmin=-2.0 * config.spin_angular_velocity,
                           xmax=2.0 * config.spin_angular_velocity,
                           label='Inertial mode range',
                           zorder=-1000,
                           edgecolor='none',
                           facecolor='grey')

            pdf.savefig()
            pyplot.close()

if __name__ == '__main__':
    main(parse_command_line())
