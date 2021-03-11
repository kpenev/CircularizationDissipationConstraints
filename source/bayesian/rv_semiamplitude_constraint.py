#!/usr/bin/env python3
"""Interface implementing constraint from RV semi-amplitude measurement."""

from multiprocessing import Pool, set_start_method
import os.path
from pickle import Pickler, Unpickler
from functools import partial
import logging

from matplotlib import pyplot
from scipy import stats
from scipy import integrate
from scipy.optimize import root_scalar
from scipy.interpolate import InterpolatedUnivariateSpline
from astropy import units as u, constants as c
import numpy

from binary_utils import calculate_secondary_mass

#TODO: allow stopping once CDF error falls below some value (small grid step)
class RVSemiAmplitudeConstraint:
    """Secondary mass constraint from observed RV semi-amplitude."""

    _logger = logging.getLogger(__name__)

    def _pdf_integrand(self, cos_inclination, max_rv_semiamplitude):
        """
        The integrand for marginalizing the PDF over inclination.

        Args:
            cos_inclination:    The value of cos(i), where i is the angle
                between the orbital angular momentum and the line of sight.

            max_rv_semiamplitude:    The semiamplitude that would be observed
                for the given orbit if it were viewed edge on.

        Returns:
            float:
                PDF of observing RV given the specified orbit.
        """

        return self._observed_rvk.pdf(
            max_rv_semiamplitude
            *
            numpy.sqrt(1.0 - numpy.square(cos_inclination))
        )

    def _cdf_integrand(self, sin_inclination, max_rv_semiamplitude):
        """The integrand for marginalizing the CDF over inclination."""

        return (self._observed_rvk.cdf(max_rv_semiamplitude * sin_inclination)
                /
                numpy.sqrt(1.0 - numpy.square(sin_inclination)))

    def _sf_integrand(self, sin_inclination, max_rv_semiamplitude):
        """The integrand for marginalizing survival functn over inclination."""

        return (self._observed_rvk.sf(max_rv_semiamplitude * sin_inclination)
                /
                numpy.sqrt(1.0 - numpy.square(sin_inclination)))

    def _get_integration_breaks(self, max_rv_semiamplitude, cdf_step=0.01):
        """Return array of points integration must hit for accuracy."""

        result = (
            self._observed_rvk.ppf(numpy.arange(cdf_step, 1.0, cdf_step))
            /
            max_rv_semiamplitude
        )
        return result[result < 1.0]

    def _upper_bound_equation(self, upper_bound, target_prob):
        """The equation to solve in order to find max RV semi-amplitude."""

        integral = integrate.quad(
            lambda s: (self._observed_rvk.sf(s * upper_bound)
                       /
                       numpy.sqrt(1.0 - numpy.square(s))),
            0,
            1,
            points=self._get_integration_breaks(upper_bound),
            **self._integration_options
        )
        self._logger.debug('Upper bound equation integral at %s: %s',
                           repr(upper_bound),
                           repr(integral))
        return integral[0] - target_prob

    def _marginalize(self,
                     max_rv_semiamplitude,
                     integrand=None,
                     value_only=True):
        """Evaluate the marginalized PDF at the given max RV semi-amplitude."""

        if max_rv_semiamplitude == 0:
            return 0.0

        points = self._get_integration_breaks(max_rv_semiamplitude)
        split = min(self._observed_rvk.isf(1e-6) / max_rv_semiamplitude, 1.0)
        if integrand is None:
            integrand = self._pdf_integrand

            split = numpy.sqrt(1.0 - numpy.square(split))
            points = numpy.sqrt(1.0 - numpy.square(points))


        integration_result = integrate.quad(
            integrand,
            0.0,
            split,
            args=(max_rv_semiamplitude,),
            points=points,
            full_output=True,
            **self._integration_options
        )
        result = numpy.array(integration_result[:2])
        if split < 1.0:
            integration_result = integrate.quad(
                integrand,
                split,
                1.0,
                args=(max_rv_semiamplitude,),
                points=points,
                full_output=True,
                **self._integration_options
            )
            result += numpy.array(integration_result[:2])

        result *= 2.0 / numpy.pi

        if value_only:
            return result[0]

        return result

    def _m2_cdf_integrand(self, secondary_mass):
        """The integrand required to calculate CDF(M2)."""

        return (
            self.rv_semi_amplitude_pdf(
                self.rv_semi_amplitude(secondary_mass * u.M_sun,
                                       cos_inclination=0.0)
            )
            *
            self._fixed['m2_prior'].pdf(secondary_mass)
        )

    @staticmethod
    def _get_interpolation(semi_amplitude_grid, pdf_values):
        """
        Return an interpolation over the given grid of values.

        Interpolates log(pdf_valuse) to ensure strictly positive function.

        Args:
            semi_amplitude_grid((,N) array):    The values of the RV
                semi-amplitude where the the marginalized PDF is known.

            pdf_values((N,2) array):    The values (`pdf_values[:, 0]`) and
                estimated errors of the PDF at the grid points.

        Returns:
            UnivariateSpline:
                Interpolation over the given data that approximates the PDF as a
                function of RV semi-amplitude.
        """

        return InterpolatedUnivariateSpline(semi_amplitude_grid,
                                            pdf_values[:, 0],
                                            k=1)


    def _get_mismatch_indices(self,
                              semi_amplitude_grid,
                              pdf_values,
                              show_mismatch_plot):
        """
        Return odd indices where interpolation based on even grid pts fails.

        Args:
            semi_amplitude_grid(1-D array):    The current set of RV
                semi-amplitude values to use to build the inerpolation.

            pdf_values(1-D array):    The values of the PDF at the
                semi-amplitude grid points.

            show_mismatch_plot(boool):    Should a plot be displayed showing the
                poorly interpolated indices?

        Returns:
            1-D int array:
                The indices where interpolating using only the even entries of
                the grid and PDF values fails when evaluated at the odd grid
                points.
        """

        interpolation = self._get_interpolation(semi_amplitude_grid[::2],
                                                pdf_values[::2])
        differences = numpy.absolute(
            pdf_values[1::2, 0]
            -
            interpolation(semi_amplitude_grid[1::2])
        )
        self._logger.debug(
            'Tuning interpolation: '
            'max error: %g, '
            'min grid step: %g, '
            'grid size: %d',
            numpy.max(differences),
            numpy.min(semi_amplitude_grid[1:] - semi_amplitude_grid[:-1]),
            semi_amplitude_grid.size
        )
        max_differences = numpy.maximum(
            10.0 * pdf_values[1::2, 1],
            (
                self._interpolation_accuracy[0] * pdf_values[:, 0].max()
                +
                self._interpolation_accuracy[1] * pdf_values[1::2, 0]
            )
        )
        result = 2 * numpy.nonzero(differences > max_differences)[0] + 1

        if 0 < result.size < 10:
            bad_interpolated_pdf = interpolation(semi_amplitude_grid[result])
            #As lazy as can be
            #pylint: disable=logging-not-lazy
            self._logger.debug(
                (
                    'Excessive differences at:'
                    +
                    '\n\tK=%g, %g vs %g (diff=%g +- %g)' * result.size
                ),
                *sum(
                    zip(semi_amplitude_grid[result],
                        bad_interpolated_pdf,
                        pdf_values[result, 0],
                        bad_interpolated_pdf - pdf_values[result, 0],
                        pdf_values[result, 1]),
                    ()
                )
            )
            #pylint: enable=logging-not-lazy


        if result.size > 0 and show_mismatch_plot:
            plot_slice = slice(max(result.min() - 1, 0), result.max() + 1)
            pyplot.errorbar(semi_amplitude_grid[plot_slice],
                            pdf_values[plot_slice, 0],
                            pdf_values[plot_slice, 1],
                            fmt='xk')
            interp_x = numpy.empty(2 * semi_amplitude_grid[plot_slice].size - 1)
            interp_x[::2] = semi_amplitude_grid[plot_slice]
            interp_x[1::2] = 0.5 * (semi_amplitude_grid[plot_slice][1:]
                                    +
                                    semi_amplitude_grid[plot_slice][:-1])
            pyplot.plot(interp_x, interpolation(interp_x), '-r')
            pyplot.show()

        return result

    def _tune_interpolation(self, num_parallel_processes, show_mismatch_plot):
        """Find an interpolation that satisfies the specified precision."""

        semi_amplitude_grid = numpy.concatenate(
            (
                self._observed_rvk.ppf(numpy.arange(0.01, 1.0, 0.01)),
                numpy.linspace(*self._support, 100)
            )
        )
        semi_amplitude_grid.sort()
        assert semi_amplitude_grid.size % 2 == 1

        marginalize = partial(self._marginalize, value_only=False)

        with Pool(num_parallel_processes) as workers:
            pdf_values = numpy.array(
                workers.map(marginalize, semi_amplitude_grid)
            )
#        pdf_values = numpy.array([marginalize(K) for K in semi_amplitude_grid])

        while True:
            mismatch_indices = self._get_mismatch_indices(semi_amplitude_grid,
                                                          pdf_values,
                                                          show_mismatch_plot)
            self._logger.debug(
                '%d indices exceed maximum allowed interpolation error.',
                mismatch_indices.size
            )
            if mismatch_indices.size == 0:
                return self._get_interpolation(semi_amplitude_grid, pdf_values)

            new_grid_values = numpy.concatenate(
                (
                    0.5 * (
                        semi_amplitude_grid[mismatch_indices - 1]
                        +
                        semi_amplitude_grid[mismatch_indices]
                    ),
                    0.5 * (
                        semi_amplitude_grid[mismatch_indices + 1]
                        +
                        semi_amplitude_grid[mismatch_indices]
                    )
                )
            )

            with Pool(num_parallel_processes) as workers:
                new_pdf_values = numpy.array(
                    workers.map(marginalize, new_grid_values)
                )

            insert_indices = numpy.concatenate(
                (mismatch_indices, mismatch_indices + 1)
            )
            semi_amplitude_grid = numpy.insert(semi_amplitude_grid,
                                               insert_indices,
                                               new_grid_values)
            pdf_values = numpy.insert(pdf_values,
                                      insert_indices,
                                      new_pdf_values,
                                      axis=0)

            interpolation = self._get_interpolation(semi_amplitude_grid,
                                                    pdf_values)
            self._logger.debug(
                'Tuned interpolation CDF error: %s',
                repr(interpolation.integral(*self._support) - 1.0),
            )

    def _check_for_pickled(self, pickle_fname):
        """Check if the given pickle file contais usable interploation."""

        def check_rvk_distro(pickled_rvk_distro):
            """Check pickled distribution for matching the RVK distribution."""

            return (
                (self._observed_rvk.args == pickled_rvk_distro.args)
                and
                (self._observed_rvk.kwds == pickled_rvk_distro.kwds)
            )


        if not os.path.exists(pickle_fname):
            open(pickle_fname, 'wb').close()
            return None
        try:
            with open(pickle_fname, 'rb') as pickle_file:
                unpickler = Unpickler(pickle_file)
                while True:
                    section, nobjects = unpickler.load()
                    assert isinstance(section, str)
                    assert isinstance(nobjects, int)
                    if section == 'MarginalizedRVKDistribution':
                        assert nobjects == 5
                        pickled_rvk_distro = unpickler.load()
                        max_discarded_probabiity = unpickler.load()
                        interpolation_accuracy = unpickler.load()
                        integration_options = unpickler.load()
                        interpolation = unpickler.load()
                        nobjects = 0
                        if(
                                check_rvk_distro(pickled_rvk_distro)
                                and
                                (
                                    max_discarded_probabiity
                                    <=
                                    self._max_discarded_probability
                                )
                                and
                                (
                                    interpolation_accuracy
                                    >=
                                    self._interpolation_accuracy
                                )
                                and
                                integration_options == self._integration_options
                        ):
                            self._logger.debug('Matching interpolation found.')
                            return interpolation

                        self._logger.debug(
                            'Interpolation config does not match.'
                        )
                    for _ in range(nobjects):
                        unpickler.load()
        except EOFError:
            self._logger.info('None of the pickled marginalized RV '
                              'semi-amplitude distributinos matches.')
            return None

    def _add_to_pickle_file(self, pickle_fname):
        """Pickle the currently set-up interpolation to the given file."""

        with open(pickle_fname, 'ab') as pickle_file:
            pickler = Pickler(pickle_file)
            pickler.dump(('MarginalizedRVKDistribution', 5))
            pickler.dump(self._observed_rvk)
            pickler.dump(self._max_discarded_probability)
            pickler.dump(self._interpolation_accuracy)
            pickler.dump(self._integration_options)
            pickler.dump(self._rv_semiamplitude_pdf_interp)

    def __init__(self,
                 *,
                 observed_rvk,
                 max_discarded_probabiity,
                 interpolation_accuracy,
                 num_parallel_processes,
                 pickle_fname,
                 show_mismatch_plot=False,
                 **integration_options):
        """
        Set-up the integrand given observed RV semi-amplitude distribution.

        Args:
            observed_rvk(rv_continuous):    The empirical distribution of the RV
                semi-amplitude (will be converted to distribution of edge-on
                orbit).

            max_discarded_probabiity(float):    The tails of the RV
                semi-amplitude distribution with weight less than this are
                truncated to define the range for interpolation.

            interplation_accuracy(float, float):    The maximum error allowed
                in the PDF interpolation as a fraction of the largest PDF value,
                and as the PDF at the inteprolated position. Comparison is to
                directly integrated values.

            integration_options(dict):    Passed directly to
                :func:`integrate.quad()` to control integration. Must not
                include `points`.
        """

        #May want to revive if problems occur.
        #pylint: disable=unused-variable
        def plot_pdf():
            """Show a plot of the PDF."""

            plot_rvk = numpy.linspace(0.0 * observed_rvk.ppf(0.99),
                                      10 * observed_rvk.ppf(0.99),
                                      1000)
            with Pool(num_parallel_processes) as workers:
                plot_pdf = numpy.array(
                    workers.map(self.rv_semi_amplitude_pdf,
                                plot_rvk * (u.m / u.s))
                )
            approx_arg = plot_rvk / (2.0**0.5 * observed_rvk.kwds['scale'])
            pyplot.plot(plot_rvk, plot_pdf, '-r')
            pyplot.show()
        #pylint: enable=unused-variable

        self._fixed = dict()
        self._interpolation_accuracy = interpolation_accuracy
        self._max_discarded_probability = max_discarded_probabiity

        self._observed_rvk = observed_rvk

        max_discarded_probabiity /= 4.0

        self._integration_options = integration_options

        upper_bound_solution = root_scalar(
            self._upper_bound_equation,
            args=(max_discarded_probabiity,),
            bracket=(
                0,
                (
                    observed_rvk.isf(max_discarded_probabiity / numpy.pi)
                    /
                    numpy.sin(max_discarded_probabiity / 2.0)
                )
            )
        )
        self._logger.debug('Upper bound for RVKPDF result: %s',
                           repr(upper_bound_solution))
        assert upper_bound_solution.converged
        self._support = (
            observed_rvk.ppf(max_discarded_probabiity),
            upper_bound_solution.root
        )
        print('Support: '+ repr(self._support))

        self._rv_semiamplitude_pdf_interp = self._check_for_pickled(
            pickle_fname
        )
        if self._rv_semiamplitude_pdf_interp is None:
            self._rv_semiamplitude_pdf_interp = self._tune_interpolation(
                num_parallel_processes,
                show_mismatch_plot
            )
            self._add_to_pickle_file(pickle_fname)
        if show_mismatch_plot:
            plot_pdf()

    def rv_semi_amplitude(self,
                          secondary_mass,
                          *,
                          cos_inclination=0.0,
                          primary_mass=None,
                          eccentricity=None,
                          orbital_period=None):
        """
        Return the RV semi-amplitude the given orbit will have.

        Any arguments left at the default value of `None` will be replaced by
        the currently fixed values per :meth:`fix_primary_and_orbit`.
        """

        if primary_mass is None:
            primary_mass = self._fixed['primary_mass']

        if eccentricity is None:
            eccentricity = self._fixed['eccentricity']

        if orbital_period is None:
            orbital_period = self._fixed['orbital_period']

        sin_inclination = numpy.sqrt(1.0 - numpy.square(cos_inclination))
        return numpy.cbrt(
            2.0 * numpy.pi * c.G
            *
            numpy.power(secondary_mass * sin_inclination, 3)
            /
            orbital_period
            /
            numpy.power(1.0 - numpy.square(eccentricity), 1.5)
            /
            numpy.square(primary_mass + secondary_mass)
        )

    def rv_semi_amplitude_pdf(self, rv_semi_amplitude):
        """
        PDF of observed RV semi-amp. marginalized over inclination.

        Args:
            rv_semi_amplitude:    The value of the RV semi-amplitude at which to
                evaluate the PDF.

        Returns:
            float:
                The value of the marginalized PDF evaluated at the given
                arguments.
        """

        if (
                self._support[0]
                <=
                rv_semi_amplitude.to_value('m/s')
                <=
                self._support[1]
        ):
            return self._rv_semiamplitude_pdf_interp(
                #False positive
                #pylint: disable=no-member
                rv_semi_amplitude.to_value('m/s')
                #pylint: enable=no-member
            )

        return 0.0

    def rv_semi_amplitude_cdf(self, rv_semi_amplitude):
        """
        CDF of observed RV semi-amp. marginalized over inclination.

        Args:
            rv_semi_amplitude:    The value of the RV semi-amplitude at which to
                evaluate the CDF.

        Returns:
            float:
                The value of the marginalized CDF.
        """

        return self._marginalize(
            #False positive
            #pylint: disable=no-value-for-parameter
            rv_semi_amplitude.to_value('m/s'),
            #pylint: enable=no-value-for-parameter
            self._cdf_integrand
        )

    def rv_semi_amplitude_sf(self, rv_semi_amplitude):
        """
        Survival function of observed RV semi-amp. marginalized over incl.

        Args:
            rv_semi_amplitude:    The value of the RV semi-amplitude at which to
                evaluate the survival function.

        Returns:
            float:
                The value of the marginalized survival function.
        """

        return self._marginalize(
            #False positive
            #pylint: disable=no-value-for-parameter
            rv_semi_amplitude.to_value('m/s'),
            #pylint: enable=no-value-for-parameter
            self._sf_integrand
        )

    def prepare_secondary_sampling(self,
                                   *,
                                   primary_mass,
                                   eccentricity,
                                   orbital_period,
                                   secondary_mass_prior):
        """
        Get ready to work with the distribution of the secondary mass.

        Fix all parameters except secordary mass, for which only a prior
        distribution is specified. The `secondary_mass_*` methods assume the
        parameters given here.

        Args:
            primary_mass(astropy quantity w/ units):    The mass of the primary
                star.

            eccentricity(numeric):    The eccentricity of the orbit.

            orbital_period(astropy quantity w/ units):    The orbital period.

            secondary_mass_prior:    The conditional distribution of the
                secondary mass in solar masses based on constraints other than
                the radial velocity semi-amplitude (e.g. photometry).
        """

        self._fixed['primary_mass'] = primary_mass
        self._fixed['eccentricity'] = eccentricity
        self._fixed['orbital_period'] = orbital_period
        self._fixed['min_m2'], self._fixed['max_m2'] = (
            secondary_mass_prior.support()
        )
        if not numpy.isfinite(self._fixed['min_m2']):
            self._fixed['min_m2'] = secondary_mass_prior.ppf(
                self._max_discarded_probability
            )
        if not numpy.isfinite(self._fixed['max_m2']):
            self._fixed['max_m2'] = secondary_mass_prior.isf(
                self._max_discarded_probability
            )
        self._fixed['min_m2'] = max(
            self._fixed['min_m2'],
            calculate_secondary_mass(
                primary_mass,
                orbital_period,
                self._support[0] * u.m / u.s,
                eccentricity
            ).to_value(u.M_sun)
        )
        self._fixed['m2_quad_points'] = numpy.array([
            calculate_secondary_mass(primary_mass,
                                     orbital_period,
                                     rvK * u.m / u.s,
                                     eccentricity).to_value(u.M_sun)
            for rvK in self._get_integration_breaks(1.0)
        ])
        self._fixed['m2_prior'] = secondary_mass_prior
        self._fixed['m2_distribution_norm'] = 1.0
        self._fixed['m2_distribution_norm'] = self.secondary_mass_cdf(
            self._fixed['max_m2'] * u.M_sun
        )
        self._logger.debug('Fixed orbit to: %s', repr(self._fixed))

    def secondary_mass_pdf(self, secondary_mass):
        """CDF(M2|M1, Porb, e)."""

        if (
                self._fixed['min_m2']
                <=
                secondary_mass.to_value(u.M_sun)
                <=
                self._fixed['max_m2']
        ):
            return (
                self._m2_cdf_integrand(secondary_mass.to_value(u.M_sun))
                /
                self._fixed['m2_distribution_norm']
            )

        return 0.0

    def secondary_mass_cdf(self, secondary_mass):
        """CDF(M2|M1, Porb, e)."""

        return integrate.quad(
            self._m2_cdf_integrand,
            self._fixed['min_m2'],
            secondary_mass.to_value(u.M_sun),
            points=self._fixed['m2_quad_points'],
            **self._integration_options
        )[0] / self._fixed['m2_distribution_norm']

    def secondary_mass_ppf(self, quantile):
        """Return the secondary mass corresponding to the given CDF quantile."""

        def equation(secondary_mass):
            """The equation to solve to get the function result."""

            return self.secondary_mass_cdf(secondary_mass * u.M_sun) - quantile

        result = root_scalar(
            equation,
            bracket=(
                self._fixed['min_m2'],
                self._fixed['max_m2']
            ),
            fprime=self._m2_cdf_integrand,
        )
        assert result.converged
        return result.root * u.M_sun

def main():
    """Avoid polluting global namespace."""

    logging.basicConfig(level=logging.DEBUG)
    set_start_method('forkserver')

    system = dict(
        ID='vB62',
        OtherIDs=dict(HD=28033),
        type='Single Lined',
        Porb=8.55089 * u.day,
        errPorb=0.00007 * u.day,
        Gamma=38.77 * u.km / u.s,
        errGamma=0.14 * u.km / u.s,
        K1=16.46 * u.km / u.s,
        errK1=0.16 * u.km / u.s,
        Ecc=0.233,
        errEcc=0.012,
        Omega=38.0 * u.deg,
        errOmega=2.8 * u.deg,
        ProjSemimajor1=1.88 * u.Gm,
        errProjSemimajor1=0.02 * u.Gm,
        MassFunc=0.0036 * u.M_sun,
        errMassFunc=0.0001 * u.M_sun,
        member=True,
        ref=2
    )

    rvk_constraint = RVSemiAmplitudeConstraint(
        observed_rvk=stats.norm(system['K1'].to_value('m/s'),
                                system['errK1'].to_value('m/s')),
        max_discarded_probabiity=1e-5,
        interpolation_accuracy=(0, 1e-4),
        num_parallel_processes=4,
        pickle_fname='rvk_constraints.pkl',
        epsabs=0,
        epsrel=1e-8,
        limit=200,
        maxp1=200
    )

    print(
        'K = '
        +
        repr(
            #False positive
            #pylint: disable=no-member
            rvk_constraint.rv_semi_amplitude(
                cos_inclination=0.0,
                primary_mass=(1.2 * u.M_sun),
                secondary_mass=(0.2 * u.M_sun),
                orbital_period=system['Porb'],
                eccentricity=system['Ecc']
            ).to_value('m/s')
            #pylint: enable=no-member
        )
    )

    reference_pdf = None
    left_plot = pyplot.subplot(131)
    middle_plot = pyplot.subplot(132)
    right_plot = pyplot.subplot(133)
    for eccentricity in numpy.linspace(0.5, 0, 6):

        rvk_constraint.prepare_secondary_sampling(
            primary_mass=(1.2 * u.M_sun),
            orbital_period=system['Porb'],
            eccentricity=eccentricity,
            secondary_mass_prior=stats.uniform(0.0, 1.2)
        )

        plot_m2 = numpy.linspace(0.15, 0.4, 1000)

        with Pool(4) as workers:
            plot_pdf = numpy.array(
                workers.map(rvk_constraint.secondary_mass_pdf,
                            plot_m2 * u.M_sun)
            )
            plot_cdf = numpy.array(
                workers.map(rvk_constraint.secondary_mass_cdf,
                            plot_m2 * u.M_sun)
            )
        if reference_pdf is None:
            reference_pdf = plot_pdf

        print('e=' + repr(eccentricity))

        left_plot.plot(plot_m2,
                       plot_pdf,
                       label='e = ' + str(eccentricity))
        middle_plot.semilogy(plot_m2,
                             plot_pdf / reference_pdf,
                             label='e = ' + str(eccentricity))
        right_plot.plot(plot_m2,
                        plot_cdf,
                        )
    middle_plot.set_ylim((0.1, 10.0))
    pyplot.show()

if __name__ == '__main__':
    main()
