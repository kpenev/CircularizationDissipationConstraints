#!/usr/bin/env python3

"""Create plots of emcee sampling results."""

import logging
from os import path

import matplotlib
from matplotlib import pyplot
from configargparse import\
    ArgumentParser,\
    DefaultsFormatter,\
    Action as ArgparseAction
import numpy
from numpy.lib import recfunctions
import emcee
import h5py
import pandas
from scipy import stats
from asteval import Interpreter
from astropy.table import Table
from astropy import constants

from orbital_evolution.transformations import lgQ

from general_purpose_python_modules.combined_mcmc_constraint import \
    CombinedMCMCConstraint
from general_purpose_python_modules.visuals import make_corner_plot

#Interface specified by argparse module.
#pylint: disable=too-few-public-methods
class ParseGrid(ArgparseAction):
    """Action for parsing linspace arguments."""

    def __call__(self, parser, namespace, values, option_string=None):
        """Parse a grid option to a numpy.arary()."""

        if len(values) == 3:
            expander = numpy.linspace
        else:
            assert len(values) == 4
            expander = getattr(numpy, values[3])
        setattr(namespace,
                self.dest,
                expander(float(values[0]),
                         float(values[1]),
                         int(values[2])))
#pylint: enable=too-few-public-methods

def add_frequency_dependence_plot_config(parser, disable=()):
    """Add command line arguments configuring the frequence dependence plots."""

    for entry in disable:
        assert entry in ['hatch',
                         'bounds',
                         'lgQ_grid',
                         'ptide_grid',
                         'heat_map',
                         'kernel',
                         'confidence',
                         'lines']

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
    if 'hatch' not in disable:
        parser.add_argument(
            '--frequency-dependence-hatch',
            default=False,
            help='If specified frequency dependent plots use hatching as well '
            'as color to distinguish between constraints from different '
            'systems.'
        )
    if 'bounds' not in disable:
        parser.add_argument(
            '--frequency-dependence-bounds',
            default=False,
            action='store_true',
            help='If specified frequency dependent plots draw lines to indicate'
            ' the bounds of the specified confidence intervals.'
        )
    if 'lgQ_grid' not in disable:
        parser.add_argument(
            '--lgQ-grid',
            default=numpy.linspace(4, 12, 100),
            action=ParseGrid,
            nargs=3,
            metavar=('MIN_LGQ', 'MAX_LGQ', 'RES'),
            help='Set the range and resolution in log10(Q) on which to '
            'calculate cobined constraints.'
        )
    if 'heat_map' not in disable:
        parser.add_argument(
            '--combined-constraint-heat-map',
            default='log',
            choices=('log', 'lin'),
            help='Plot a heat map of the combined (log-)likelihood in the '
            'frequnecy dependence plot. If enabled, the confidence interval is '
            'shown using lines only (not filled).'
        )
        parser.add_argument(
            '--heat-map-contrast',
            default=1e-3,
            type=float,
            help='The smallest value to plot on the heat map is this factor '
            'times the maximum value.'
        )
    if 'kernel' not in disable:
        parser.add_argument(
            '--kernel-width',
            default=0.2,
            type=float,
            help='The width of the kernel that gets convolved with the samples '
            'for calculating combined constraints. If not specified, a combined'
            ' cosntraint is not generated.'
        )
    if 'confidence' not in disable:
        parser.add_argument(
            '--plot-confidence',
            type=float,
            nargs='*',
            default=[stats.norm.cdf(2.0) - stats.norm.cdf(-2.0)],
            help='Specify the confidence to display in the dissipation vs tidal'
            ' frequency plot (see --frequency-dependence-plot-fname argument).'
        )
    if 'ptide_grid' not in disable:
        parser.add_argument(
            '--ptide-grid',
            nargs=4,
            default=numpy.logspace(0.0, numpy.log10(50.0), 100),
            action=ParseGrid,
            metavar=('MIN_PERIOD', 'MAX_PERIOD', 'RES', 'GRID_TYPE'),
            help='Set the range and resolution of the tidal period to include '
            'in the frequency dependence plot (see '
            '--frequency-dependence-plot-fname argument).'
        )
    if 'lines' not in disable:
        parser.add_argument(
            '--frequency-dependence-plot-no-lines',
            action='store_true',
            help='If passed, the frequency dependence plot will not include '
            'individual lines.'
        )

def parse_command_line():
    """Parse the command line for what and how to plot."""

    parser = ArgumentParser(
        description=__doc__,
        default_config_files=['visualize_emcee.cfg'],
        formatter_class=DefaultsFormatter,
        ignore_unknown_config_file_keys=False
    )
    parser.add_argument(
        'samples_fnames',
        nargs='+',
        help='The filename(s) containing the stored emcee chain(s) to plot. For'
        ' now only frequency dependence plots support plotting multiple '
        'constraints on top of each other. All other plots only plot the chain '
        'from the last file.'
    )
    parser.add_argument(
        '--corner-plot-fname', '--corner-plot', '--corner',
        default=None,
        help='If specified, a corner plot is created and saved with the given '
        'filename.'
    )
    parser.add_argument(
        '--trace-plot-fname', '--traces-plot', '--traces',
        default=None,
        help='If specified, a trace plot is generated and saved with the given '
        'filaname.'
    )
    parser.add_argument(
        '--frequency-dependence-plot-fname',
        '--frequency-dependence-plot',
        '--frequency',
        default=None,
        help='If specified, a plot of the confidence interval of lgQ vs tidal '
        'frequency is generated and saved with the given filename. Inertial '
        'mode enhancement is ignored.'
    )
    add_frequency_dependence_plot_config(parser)
    parser.add_argument(
        '--errorbar-plot',
        nargs=3,
        metavar=('FILENAME', 'XEXPR', 'YEXPR'),
        default=None,
        help='If specified, for each input system the X and Y expressions are '
        'evaluated from the chain to get confidence intervals (see '
        '--plot-confidence argument) and a plot of the results is saved under '
        'the given file name.'
    )
    parser.add_argument(
        '--corner-plot-params',
        default=None,
        nargs='+',
        help='If specified, only the listed parameters are included in the corner '
        'plot.'
    )
    parser.add_argument(
        '--corner-plot-log-param',
        default=[],
        action='append',
        help='Specify that a log(parameter) should be plotted in corner plot.'
    )

    parser.add_argument(
        '--max-traces-per-plot',
        type=int,
        default=5,
        help='The maximum number of traces to include in a trace plot (more '
        'walkers will results in more panels in the plot).'
    )
    parser.add_argument(
        '--log-x',
        action='store_true',
        help='Switch the x-axis to log-scale.'
    )
    return parser.parse_args()


def get_chain_name(samples_fname, chain_conditions):
    """Return the name of the chain satisfying the given conditions."""

    system_name = None
    with h5py.File(samples_fname, 'r') as chain_file:
        chain_name = None
        for try_chain_name in chain_file.keys():
            match = True
            for attr_name, expected_value in chain_conditions:
                match = (
                    match
                    and
                    (
                        chain_file[try_chain_name].attrs[attr_name]
                        ==
                        numpy.array(float(expected_value))
                    ).all()
                    and
                    chain_file[try_chain_name].attrs['iteration'] > 0
                )
            if match:
                if chain_name is not None:
                    raise RuntimeError(
                        'Multiple chains found in %s that satisify %s: '
                        '%s and %s'
                        %
                        (
                            repr(samples_fname),
                            repr(chain_conditions),
                            repr(chain_name),
                            repr(try_chain_name)
                        )
                    )
                chain_name = try_chain_name
                if 'system' in chain_file[chain_name].attrs:
                    system_name = chain_file[chain_name].attrs['system']
        if chain_name is None:
            return None, None
        if system_name is None:
            fname, ext = path.splitext(path.basename(samples_fname))
            assert ext in ['.h5', '.hdf5']
            if fname.startswith('system'):
                _, system_name = fname.split('_')
                system_name = int(system_name)
            else:
                system_name, fname_end = fname.split('_', 1)
                assert system_name.endswith(' b')
                assert fname_end == 'mcmc_progress'
        return chain_name, system_name


def get_backend(samples_fname, chain_conditions):
    """Return the chain to plot."""

    chain_name, system_name = get_chain_name(samples_fname, chain_conditions)
    print('{system:s} chain name: {chain:s}'.format(system=str(system_name),
                                                  chain=str(chain_name)))
    backend = emcee.backends.HDFBackend(samples_fname,
                                        name=chain_name,
                                        read_only=True)
    if chain_name is None or backend.iteration == 0:
        return None, None

    return backend, system_name

def get_confidence_interval(samples, confidence):
    """
    Return the confidence interval for quantities given sampled values.

    Args:
        samples(2-D array):    The first index should iterate over quantities to
            get confidence interval for, and the second index should iterate
            over values for those quantities at each sample in the MCMC chain.

        confidence(float):    The confidence level at which to find the
            interval.

    Returns:
        1-D array:
            The lower end of the confidence interval for earch input quantity.

        1-D array:
            The upper end of the confidence interval for earch input quantity.
    """

    samples = numpy.sort(samples, 1)

    num_exclude = int(
        numpy.floor(
            (1.0 - confidence) * samples.shape[1]
        )
    )
    lower = samples[:, :num_exclude + 1]
    upper = samples[:, -num_exclude-1:]
    selected_indices = numpy.argmin(upper - lower, axis=1)
    lower = numpy.take_along_axis(lower,
                                  selected_indices[:, None],
                                  axis=1).flatten()
    upper = numpy.take_along_axis(upper,
                                  selected_indices[:, None],
                                  axis=1).flatten()
    return lower, upper

def get_tex_label(quantity):
    """The TeX label to use for the given quantity in corner plots."""

    if quantity == 'lgQ_min':
        return r'$\log_{10}Q_0$'
    if quantity == 'lgQ_break_period':
        return r'$\log_{10}P_0$'
    if quantity == 'lgQ_powerlaw':
        return r'$\alpha$'
    if quantity == 'e_f':
        return r'$e_{max}$'
    if quantity == 'lnP':
        return r'$\ln\mathcal{P}$'
    return quantity

def save_corner_plot(samples, log_probability, config):
    """Create the corner plot specified on the command line."""

    samples = samples.flatten()
    log_probability = log_probability.flatten()
    include_params = config.corner_plot_params or [
        param
        for param in samples.dtype.names
        if (
            samples[param].min() != samples[param].max()
            and
            numpy.isfinite(samples[param]).any()
        )
    ]
    print('Corner plot parameters: ' + repr(include_params))
    plot_data = samples[include_params]
    for param_name in config.corner_plot_log_param:
        plot_data[param_name] = numpy.log10(plot_data[param_name])
    labels = [get_tex_label(q) for q in include_params + ['lnP']]
    plot_data_frame = pandas.DataFrame(plot_data)
    plot_data_frame.insert(len(include_params), 'lnP', log_probability)
    make_corner_plot(plot_data_frame,
                     corner_plot_fname=config.corner_plot_fname,
                     labels=labels,
                     plot_contours=False,
                     bins=30)

def save_trace_plot(samples, log_probability, config):
    """Create the trace plot specified on the command line."""

    num_panels = samples.shape[1] // config.max_traces_per_plot
    _, panel_list = pyplot.subplots(num_panels,
                                    1,
                                    sharex=True,
                                    figsize=[6.4, 2.4 * num_panels])
    finite_probability = numpy.isfinite(log_probability)
    for first, panel in zip(
            range(0, samples.shape[1], config.max_traces_per_plot),
            panel_list
    ):
        for trace in range(first, first + config.max_traces_per_plot):
            plot_y = samples['lgQ_min'][:, trace].flatten()
            plot_x = numpy.arange(plot_y.size)
            plot_solid = finite_probability[:, trace].flatten()
            plot_dotted = numpy.logical_not(plot_solid)
            plot_dotted[numpy.argmax(plot_solid)] = True
            solid_line = panel.plot(plot_x[plot_solid],
                                    plot_y[plot_solid],
                                    '-')[0]
            panel.plot(plot_x[plot_dotted],
                       plot_y[plot_dotted],
                       ':',
                       color=solid_line.get_color())


    pyplot.savefig(config.trace_plot_fname)


def evaluate_lgq(samples, ptide_grid):
    """
    Evaluate the dissipation model that was sampled as function of period.

    Args:
        samples:    The samples directly as returned from the backend.

        ptide_grid:    The grid of tidal periods to evaluate lgQ at.

    Returns:
        2-D array:
            The first index corresponds to `ptide_grid`. The second index
            iterates over points in the MCMC chain.
    """

    return (
        samples['lgQ_min'].flatten()[None, :]
        +
        numpy.maximum(
            0.0,
            (
                samples['lgQ_powerlaw'].flatten()[None, :]
                *
                numpy.log10(
                    ptide_grid[:, None]
                    /
                    samples['lgQ_break_period'].flatten()[None, :]
                )
            )
        )
    )


class FrequencyDependencePlotter:
    """Create a plot showing the constraint of lgQ vs tidal frequency."""

    def __init__(self, num_chains, config):
        """
        Prepare for plotting.

        Args:
            num_chains:    The expected number of chains that will be combined
                (only sets transparency)

            config:    The configuration for plotting (parsed command line).

        Returns:
            None
        """

        self.config = config
        self.transparency = (
            1.0 / (len(config.plot_confidence)
                   *
                   num_chains)
            if getattr(config, 'plot_confidence', None) else None
        )
        self._constraint_index = 0
        self._hatch_list = ['\\\\',
                            '//',
                            '||',
                            '--',
                            '..',
                            'o',
                            '+',
                            'x',
                            '*',
                            'O']
        self._color_list = ['tab:blue',
                            'tab:orange',
                            'tab:green',
                            'tab:red',
                            'tab:purple',
                            'tab:brown',
                            'tab:pink',
                            'tab:gray',
                            'tab:olive',
                            'tab:cyan']
        self.combined_pdf = (
            None if config.kernel_width is None
            else CombinedMCMCConstraint(config.lgQ_grid,
                                        config.kernel_width)
        )


    def add_chain(self,
                  samples,
                  label,
                  prior_range=(-numpy.inf, numpy.inf),
                  period_range=None):
        """
        Overplot another chain of samples and update combined constraint.

        Args:
            samples(array):    The samples of tidal dissipation parameters to
                add.

            label(str):    How to label the area per the new samples added to
                the plot.

            prior_range(2-tuple):    Upper and lower limit on the evaluated
                lg(Q) values beyond which the PDF distribution is replaced by
                the value at those boundaries.

            period_range(None or 2x2 iterable):    If not None, specifies a
                range of periods for which the distribution above and below the
                50-th percentile is considered valid. The invalid part of the
                bistribution is not folded into the combined constraint.
        """

        evaluated_lgq = evaluate_lgq(samples, self.config.ptide_grid)

        assert numpy.isfinite(evaluated_lgq).all()
        if self.combined_pdf is not None:
            if period_range is None:
                self.combined_pdf.add_samples(
                    evaluated_lgq,
                    prior_range,
                )
            else:
                for (min_period, max_period), ignore_side in zip(period_range,
                                                                 [-1, 1]):
                    self.combined_pdf.add_samples(
                        samples=evaluated_lgq,
                        prior_range=prior_range,
                        include_samples=numpy.logical_and(
                            self.config.ptide_grid >= min_period,
                            self.config.ptide_grid <= max_period,
                        ),
                        ignore_past_quantile=(0.5, ignore_side)
                    )


        if not self.config.frequency_dependence_plot_no_lines:
            pyplot.plot(
                self.config.ptide_grid,
                evaluated_lgq[:, ::1],
                '-k',
                linewidth=5,
                alpha=0.01,
                zorder=0
            )

        for conf_index, confidence in enumerate(
                getattr(self.config, 'plot_confidence', [])
        ):
            min_lgq, max_lgq = get_confidence_interval(evaluated_lgq,
                                                       confidence)
            color = self._color_list[self._constraint_index
                                     %
                                     len(self._color_list)]
#            if self.config.frequency_dependence_hatch:
#                plot_kwargs = dict(
#                    #alpha=self.transparency,
#                    edgecolor=color,
#                    facecolor='none',
#                    hatch=self._hatch_list[self._constraint_index
#                                           %
#                                           len(self._hatch_list)],
#                    linewidth=0
#                )
            self._constraint_index += 1

            if conf_index == 0:
                pyplot.fill_between(
                    [numpy.nan, numpy.nan],
                    [numpy.nan, numpy.nan],
                    [numpy.nan, numpy.nan],
                    edgecolor=color,
                    label=label
                )

            if self.config.frequency_dependence_bounds:
                pyplot.plot(self.config.ptide_grid,
                            max_lgq,
                            '--',
                            zorder=20,
                            linewidth=1,
                            color=color)
                pyplot.plot(self.config.ptide_grid,
                            min_lgq,
                            ':',
                            zorder=20,
                            linewidth=1,
                            color=color)

    def plot_combined_constraint(self, label=None, color_list=None):
        """
        Add a plot of the cobmined constraint collected so far.

        Plots potentially multiple confidence intervals with different colors.

        Args:
            label(str):    The label to use for the plot.

            color_list([matplotlib color]):    Set and order of Colors to use
                for different confidence intervals.

        Returns:
            None
        """

        def get_interval(cdf_slice, confidence):
            """Return shortest lgQ confidence interval for fix P slice."""

            lower_indices = numpy.arange(numpy.searchsorted(cdf_slice,
                                                            1.0 - confidence))
            upper_indices = numpy.searchsorted(
                cdf_slice,
                cdf_slice[:lower_indices[-1] + 1]+ confidence
            )
            selected = numpy.argmin(upper_indices - lower_indices)
            lower_bound, upper_bound = self.config.lgQ_grid[
                [lower_indices[selected], upper_indices[selected]]
            ]
            return lower_bound, upper_bound

        assert self.combined_pdf is not None
        pdf = self.combined_pdf()
        cdf = numpy.cumsum(pdf, 0) - 0.5 * pdf

        if self.config.combined_constraint_heat_map is not None:
            self.plot_combined_pdf_heat_map()

        for color_index, confidence in enumerate(
                self.config.plot_confidence
        ):
            combined_bounds = numpy.empty(
                shape=(2, self.config.ptide_grid.size),
                dtype=float
            )
            print(80*'=')
            print('%f%% confidence interval: ' % (100.0 * confidence))
            for ptide_index in range(self.config.ptide_grid.size):
                combined_bounds[:, ptide_index] = get_interval(
                    cdf[:, ptide_index],
                    confidence
                )
                print(
                    '%25.16e %25.16e'
                    %
                    (
                        self.config.ptide_grid[ptide_index],
                        (
                            combined_bounds[1][ptide_index]
                            -
                            combined_bounds[0][ptide_index]
                        )
                    )
                )

            if color_list is None:
                color_list = self._color_list
            color = color_list[color_index % len(color_list)]

            if self.config.combined_constraint_heat_map:
                pyplot.plot(self.config.ptide_grid,
                            combined_bounds[1],
                            '--',
                            zorder=30,
                            linewidth=4,
                            color=color,
                            label=label)
                pyplot.plot(self.config.ptide_grid,
                            combined_bounds[0],
                            ':',
                            zorder=30,
                            linewidth=4,
                            color=color)
            else:
                pyplot.fill_between(self.config.ptide_grid,
                                    combined_bounds[0],
                                    combined_bounds[1],
                                    zorder=30,
                                    facecolor=color,
                                    edgecolor='none',
                                    alpha=0.7,
                                    label=label)



        if self.config.combined_constraint_heat_map:
            pyplot.colorbar()


    def plot_combined_pdf_heat_map(self,
                                   xlabel='Tidal Period [d]',
                                   ylabel=r"$\log_{10}Q_\star'$"):
        """
        Create a 2-D plot of lgQ vs Ptide color coding KDE of PDF.

        Args:
            None

        Returns:
            None
        """

        assert self.combined_pdf is not None

        period_boundaries = numpy.empty(self.config.ptide_grid.size + 1)
        lgq_boundaries = numpy.empty(
            self.config.lgQ_grid.size
            +
            1
        )
        for boundaries, grid in [
                (period_boundaries, self.config.ptide_grid),
                (lgq_boundaries, self.config.lgQ_grid)
        ]:
            boundaries[1:-1] = 0.5 * (grid[1:] + grid[:-1])
            boundaries[0] = 1.5 * grid[0] - 0.5 * grid[1]
            boundaries[-1] = 1.5 * grid[-1] - 0.5 * grid[-2]

        plot_z = self.combined_pdf()
        plot_z_min = plot_z.min()
        if not plot_z_min > plot_z.max() * self.config.heat_map_contrast:
            plot_z_min = plot_z.max() * self.config.heat_map_contrast

        plot_args = dict(shading='flat', cmap='viridis', zorder=10)

        if self.config.combined_constraint_heat_map == 'log':
            with numpy.errstate(divide='ignore'):
                plot_z = numpy.log10(plot_z)
            plot_args['vmin'] = numpy.log10(plot_z_min)
        else:
            plot_args['vmin'] = plot_z_min

        plot_args['vmax'] = plot_z.max()

        pyplot.pcolormesh(
            period_boundaries,
            lgq_boundaries,
            plot_z,
            **plot_args
        )
        if xlabel:
            pyplot.xlabel(xlabel)
        if ylabel:
            pyplot.ylabel(ylabel)


    def plot_combined_quantiles(self,
                                cdf_values,
                                fmt='-',
                                label_fmt='CDF={0:.3f}'):
        """Plot the specified percentiles."""

        assert self.combined_pdf is not None

        data_behind = Table(
            [self.config.ptide_grid],
            names=['Ptide'],
            dtype=[float],
            descriptions=["The tidal period at which the combined Q' constraint"
                          " was evaluated"],
        )

        for cdf in cdf_values:
            label = label_fmt.format(cdf)
            plot_y = self.combined_pdf.ppf(cdf)
            pyplot.plot(
                self.config.ptide_grid,
                plot_y,
                fmt,
                label=label,
                linewidth=3,
                zorder=20
            )
            data_behind[label] = plot_y
            data_behind[label].description = (
                (
                    "The value of log10(Q') at the {0} quantile of the combined"
                    " distribution from all binaries"
                ).format(label)
            )
        return data_behind


    def save(self, filename=None):
        """Save the plot to the file sepecified by the init configuration."""

        pyplot.figure(figsize=(11, 8.5))
        self.plot_combined_constraint()
        pyplot.xlabel(r'$P_{tide}$ [days]')
        pyplot.ylabel(r"$\log_{10}Q_\star'$")
        pyplot.savefig(filename or self.config.frequency_dependence_plot_fname)


#Simplifying would make things less readable
#pylint: disable=too-many-locals
def add_errorbar(samples, config):
    """Add a single error bar per --errorbar-plot configuration argument."""

    x_expression, y_expression = config.errorbar_plot[1:]
    evaluate = Interpreter()
    for quantity in samples.dtype.names:
        evaluate.symtable[quantity] = samples[quantity].flatten()

    x_samples = evaluate(x_expression)
    y_samples = evaluate(y_expression)
    print('X (%d): ' % x_samples.shape + repr(x_samples))
    line_width = 1
    for confidence in reversed(sorted(config.plot_confidence)):
        x = numpy.median(x_samples)
        y = numpy.median(y_samples)
        x_min, x_max = get_confidence_interval(numpy.atleast_2d(x_samples),
                                               confidence)
        y_min, y_max = get_confidence_interval(numpy.atleast_2d(y_samples),
                                               confidence)
        pyplot.errorbar(
            x=x,
            y=y,
            xerr=[[x - x_min], [x_max - x]],
            yerr=[[y - y_min], [y_max - y]],
            elinewidth=line_width,
            color='black'
        )
        line_width += 2
#pylint: enable=too-many-locals


def ensure_uniform_blob_columns(blobs):
    """Ensure output blobs have correct columns for analysis."""

    if blobs.dtype == float:
        blobs = recfunctions.unstructured_to_structured(
            blobs,
            names=['primary_mass',
                   'age',
                   'secondary_radius',
                   'feh',
                   'secondary_mass',
                   'initial_star_period',
                   'lgQ_min',
                   'lgQ_break_period',
                   'lgQ_powerlaw',
                   'lgQ_star',
                   'e_now',
                   'log-likelihood']
        )
        blobs['secondary_radius'] *= constants.R_earth.to_value('R_jup')
        blobs['secondary_mass'] *= constants.M_earth.to_value('M_jup')

    blob_columns = list(blobs.dtype.names)
    if 'lgQ_min' not in blob_columns:
        blobs['break_period'] = 2.0 * numpy.pi / blobs['break_period']
        phase_lag_max = blobs['phase_lag_max']
        two_breaks = blobs['alpha'] > 0
        phase_lag_max[two_breaks] *= numpy.power(
            50.0 / blobs['break_period'][two_breaks],
            blobs['alpha'][two_breaks]
        )
        blobs['phase_lag_max'] = lgQ(phase_lag_max)

        rename_columns = [('phase_lag_max', 'lgQ_min'),
                          ('alpha', 'lgQ_powerlaw'),
                          ('break_period', 'lgQ_break_period')]
        if 'm_sum' in blob_columns and 'mass_ratio' in blob_columns:
            rename_columns.extend([('m_sum', 'primary_mass'),
                                   ('mass_ratio', 'secondary_mass')])
            primary_mass = blobs['m_sum'] / (1.0 + blobs['mass_ratio'])
            blobs['m_sum'] = primary_mass
            blobs['mass_ratio'] *= primary_mass
        if 'metallicity' in blob_columns:
            rename_columns.append(('metallicity', 'feh'))

        for from_col, to_col in rename_columns:
            blob_columns[blob_columns.index(from_col)] = to_col

        blobs.dtype.names = tuple(blob_columns)
#        if 'Wdisk' not in blob_columns:
#            return recfunctions.append_fields(
#                blobs.flatten(),
#                'Wdisk',
#                numpy.full(blobs['lgQ_min'].size, numpy.nan),
#                float
#            ).reshape(blobs.shape)
    return blobs


def get_plot_data(samples_fname_list, burn_in, chain_condition):
    """Return the samples to use for plotting for a single system."""

    combined_blobs = None
    combined_logprob = None
    for samples_fname in samples_fname_list:
        backend, system_name = get_backend(samples_fname, chain_condition)
        assert backend is not None
        blobs = backend.get_blobs()
        print('Blobs: ' + repr(blobs))
        blobs = ensure_uniform_blob_columns(blobs)
        logprob = backend.get_log_prob()

        if combined_blobs is None:
            assert combined_logprob is None
            combined_blobs = blobs
            combined_logprob = logprob
        else:
            print('Adding %s steps to %s old ones' % (blobs.shape,
                                                      combined_blobs.shape))
            final_shape = (combined_blobs.shape[0] + blobs.shape[0],
                           combined_blobs.shape[1])
            combined_blobs = recfunctions.stack_arrays(
                (combined_blobs, blobs)
            ).reshape(final_shape)
            combined_logprob = numpy.concatenate((combined_logprob, logprob))
            print('Ended up with %s steps' % (combined_blobs.shape,))

    nan_indices = numpy.nonzero(
        numpy.logical_not(numpy.isfinite(combined_logprob))
    )[0]
    if nan_indices.size:
        last_nan = nan_indices.max()
    else:
        last_nan = -1
    for colname in combined_blobs.dtype.names:
        if colname == 'Wdisk':
            continue
        nan_indices = numpy.nonzero(
            numpy.logical_not(numpy.isfinite(combined_blobs[colname]))
        )[0]
        if nan_indices.size:
            last_nan = max(
                last_nan,
                nan_indices.max()
            )
            print('Last nan for %s: %d' % (colname, nan_indices.max()))

    print('Discarding %d leading nan steps' % (last_nan + 1))
    print('There are %d burn-in steps' % burn_in)
    combined_blobs = combined_blobs[last_nan + burn_in + 1:]
    combined_logprob = combined_logprob[last_nan + burn_in + 1:]

    return (
        system_name,
        combined_blobs,
        combined_logprob
    )

def main(config):
    """"Avoid polluting global namespace."""

    if config.log_x:
        pyplot.gca().set_xscale('log')

    frequency_dependence_plotter = FrequencyDependencePlotter(
        len(config.samples_fnames),
        config
    )
    for samples_fname in config.samples_fnames:
        burn_in = 0
        if ':' in samples_fname:
            samples_fname, burn_in = samples_fname.split(':')
            burn_in = int(burn_in)

        try:
            system_name, samples, log_probability = get_plot_data(
                [samples_fname],
                burn_in,
                config.chain_condition
            )

            if config.corner_plot_fname:
                save_corner_plot(samples, log_probability, config)
                pyplot.clf()
            if config.trace_plot_fname:
                save_trace_plot(samples, log_probability, config)
                pyplot.clf()
            if config.frequency_dependence_plot_fname:
                frequency_dependence_plotter.add_chain(
                    samples,
                    system_name.replace('_', ' ')
                )
            if config.errorbar_plot:
                add_errorbar(samples, config)
        except RuntimeError:
            print('Empty chain in %s, skipping!' % samples_fname)

    pyplot.ylim(4.0, 12.0)
    pyplot.legend()

    if config.frequency_dependence_plot_fname:
        frequency_dependence_plotter.save()
    if config.errorbar_plot:
        pyplot.savefig(config.errorbar_plot[0])

if __name__ == '__main__':
    matplotlib.use('Agg')
    logging.basicConfig(level=logging.CRITICAL)
    main(parse_command_line())
