#!/usr/bin/env python3

"""Interface for working with mass constraints from color-mag measurements."""

from multiprocessing import Pool, set_start_method
import os.path
from pickle import Pickler, Unpickler
from functools import partial
import logging

from matplotlib import pyplot#, cm
from scipy import integrate, stats, optimize
import numpy

from planetary_system_io import read_cds_pipe_table
#False positive (fixed in __init__.py)
#pylint: disable=import-error
from cmd_utils import CMDPhotometryInterpolator
from command_line_utilities import data_dir
from photometric_secondary_constraint import PhotometricSecondaryConstraint
#pylint: enable=import-error

#Simplifying decreases readability
#pylint: disable=too-many-instance-attributes
class PhotometricConstraint:
    """Constraint on component masses from observed color & magnitude."""

    _logger = logging.getLogger(__name__)

    def _joint_likelihood(self, secondary_mass, primary_mass, return_log=False):
        """The likelihood of the observed colors and magnitudes given masses."""

        if (
                min(primary_mass, secondary_mass) < self.mass_range[0]
                or
                max(primary_mass, secondary_mass) > self.mass_range[1]
        ):
            return -numpy.inf if return_log else 0

        predicted_photometry = dict()
        for interpolator in self._photometry_interpolators:
            predicted_photometry.update(
                zip(
                    interpolator.available_filters,
                    interpolator.get_binary_magnitudes(primary_mass,
                                                       secondary_mass)
                )
            )

        result = (0.0 if return_log else 1.0)
        for quantity, distribution in self._measured_photometry.items():
            if '-' in quantity:
                mag1, mag2 = (mag.strip() for mag in quantity.split('-'))
                predicted_value = (predicted_photometry[mag1]
                                   -
                                   predicted_photometry[mag2])
            else:
                predicted_value = (predicted_photometry[quantity.strip()]
                                   +
                                   self._distance_modulus)

            if return_log:
                result += distribution.logpdf(predicted_value)
            else:
                result *= distribution.pdf(predicted_value)
        return result

    def _prepare_constraint(self):
        """Prepare the constraint for use with current configuration."""

        def integrand(primary_mass, _):
            """The derivative of cumulative m1 likelihood."""

            return integrate.quad(self._joint_likelihood,
                                  self.mass_range[0],
                                  primary_mass,
                                  args=(primary_mass,),
                                  points=self._quad_points,
                                  limit=200,
                                  **self._integration_config)[0]

        bounds = (self.mass_range[0] + 0.001, self.mass_range[1] - 0.001)
        minimization = optimize.minimize(
            lambda masses: -self._joint_likelihood(*masses, return_log=True),
            numpy.full((2,), 0.5 * (self.mass_range[0] + self.mass_range[1])),
            bounds=(bounds, bounds)
        )
        assert minimization.success
        self._logger.debug('Maximum log-likelihood(m1=%s, m2=%s): %s',
                           repr(minimization.x[0]),
                           repr(minimization.x[1]),
                           repr(-minimization.fun))

        result = integrate.solve_ivp(
            integrand,
            self.mass_range,
            numpy.array([0.0]),
            dense_output=True,
            max_step=1e-2,
            rtol=1e-6,
            atol=1e-9 * numpy.exp(-minimization.fun),
            method='LSODA'
        )
        assert result.success
        return result.sol

    def _check_for_pickled(self, pickle_fname):
        """Check if given file contains a re-usable pickle for current setup."""

        def compare_measured_photometry(pickled_photometry):
            """Check if the pickled photometry measurements match config."""

            if self._measured_photometry.keys() != pickled_photometry.keys():
                return False
            for quantity, distribution in self._measured_photometry.items():
                if distribution.args != pickled_photometry[quantity].args:
                    return False
                if distribution.kwds != pickled_photometry[quantity].kwds:
                    return False
            return True


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
                    if section == 'PhotometricConstraint':
                        assert nobjects == 5
                        isochrone_fnames = unpickler.load()
                        measured_photometry = unpickler.load()
                        distance_modulus = unpickler.load()
                        integration_config = unpickler.load()
                        nobjects -= 4

                        if(
                                tuple(
                                    interp.isochrone_fname
                                    for interp in self._photometry_interpolators
                                ) == isochrone_fnames
                                and
                                compare_measured_photometry(measured_photometry)
                                and
                                self._distance_modulus == distance_modulus
                                and
                                self._integration_config == integration_config
                        ):
                            self._logger.debug('Found matching pickle')
                            return unpickler.load()

                        self._logger.debug('Pickled constraint does not '
                                           'match!')

                    for _ in range(nobjects):
                        unpickler.load()

        except EOFError:
            self._logger.info(
                'None of the pickled color magnitude constraints matches.'
            )
            return None

    def _add_to_pickle_file(self, pickle_fname):
        """Pickle a fully set-up constaint to the given file for fast re-use."""

        with open(pickle_fname, 'ab') as pickle_file:
            pickler = Pickler(pickle_file)
            pickler.dump(('PhotometricConstraint', 5))
            pickler.dump(
                tuple(interp.isochrone_fname
                      for interp in self._photometry_interpolators)
            )
            pickler.dump(self._measured_photometry)
            pickler.dump(self._distance_modulus)
            pickler.dump(self._integration_config)
            pickler.dump(self._cumulative_m1_likelihood)

    def __init__(self,
                 photometry_interpolators,
                 measured_photometry,
                 distance_modulus,
                 pickle_fname,
                 **integration_config):
        """
        Set-up the constraint for a given cluster and system data.

        Args:
            photometry_interpolators([CMDPhotometryInterpolator]):    List of
                objects able to predict relevant magnitudes for a given stellar
                mass or binary based on the color-magnitude isochrones for the
                cluster the binary is a member of.

            measured_photometry(dict):    Dictionary containing available
                photometric and color measurements for the system. Keys should
                be either filter names (e.g. `"B"` or `"g'"`) or colors (e.g.
                `"B-V"` or `"u'-g'"`) and values should be distributions
                (instances of `rv_continuous`).

            pickle_fname(str):    The name of a file to save/load the current
                constraint for fast re-use.

            integration_config:    Keyword arguments to pass to scipy numerical
                integration functions used to normalize and marginalize
                likelihoods.
        """

        self._photometry_interpolators = photometry_interpolators
        self.mass_range = (-float('inf'), float('inf'))
        for interpolator in photometry_interpolators:
            self.mass_range = (
                max(self.mass_range[0], interpolator.min_mass),
                min(self.mass_range[1], interpolator.max_mass)
            )

        self._integration_config = integration_config
        self._quad_points = numpy.linspace(*self.mass_range, 30)
        self._measured_photometry = measured_photometry
        self._distance_modulus = distance_modulus
        self._cumulative_m1_likelihood = self._check_for_pickled(pickle_fname)
        if self._cumulative_m1_likelihood is None:
            self._cumulative_m1_likelihood = self._prepare_constraint()
            self._add_to_pickle_file(pickle_fname)

        self._norm = self._cumulative_m1_likelihood(self.mass_range[1])[0]
        self._secondary_norm = None

    def pdf(self, primary_mass, secondary_mass):
        """The joint PDF of component masses given observed colors and mags."""

        if(
                secondary_mass > primary_mass
                or
                secondary_mass < self.mass_range[0]
                or
                primary_mass > self.mass_range[1]
        ):
            return 0.0
        return (
            self._joint_likelihood(secondary_mass, primary_mass)
            /
            self._norm
        )

    def logpdf(self, primary_mass, secondary_mass):
        """Natural log of the joint PDF."""

        if(
                secondary_mass > primary_mass
                or
                secondary_mass < self.mass_range[0]
                or
                primary_mass > self.mass_range[1]
        ):
            return -numpy.inf
        return (
            self._joint_likelihood(secondary_mass, primary_mass, True)
            -
            numpy.log(self._norm)
        )


    def primary_mass_pdf(self, primary_mass):
        """The PDF(m1) marginalized over m2."""

        return (
            integrate.quad(self._joint_likelihood,
                           self.mass_range[0],
                           primary_mass,
                           args=(primary_mass,),
                           points=self._quad_points,
                           **self._integration_config)[0]
            /
            self._norm
        )

    def primary_mass_cdf(self, primary_mass):
        """The CDF(m1) marginalized over m2."""

        return self._cumulative_m1_likelihood(primary_mass)[0] / self._norm

    def primary_mass_ppf(self, quantile):
        """Find the primary mass for which the CDF matches given quantile."""

        def equation(primary_mass):
            return self.primary_mass_cdf(primary_mass) - quantile

        return optimize.root_scalar(equation, bracket=self.mass_range).root

    def get_conditional_secondary_mass_distribution(self, primary_mass):
        """Return scipy style RV for the secondary mass given primary mass."""

        return PhotometricSecondaryConstraint(self, primary_mass)

#pylint: enable=too-many-instance-attributes


def plot_m1_pdf(constraint):
    """Display a plot of the PDF(primary_mass) marginalized over m2."""

    plot_x = numpy.linspace(constraint.mass_range[0],
                            constraint.mass_range[1],
                            1000)
    with Pool(4) as workers:
        plot_z = numpy.array(workers.map(constraint.primary_mass_pdf, plot_x))
    print('PDF(M1): ' + repr(plot_z))
    pyplot.plot(plot_x, plot_z)
    pyplot.show()

def plot_m1_cdf(constraint):
    """Display a plot of the CDF(primary_mass) marginalized over m2."""

    plot_x = numpy.linspace(constraint.mass_range[0],
                            constraint.mass_range[1],
                            1000)
    plot_z = numpy.vectorize(constraint.primary_mass_cdf)(plot_x)
    pyplot.plot(plot_x, plot_z)
    pyplot.show()

def plot_joint_pdf(constraint):
    """Display a 3-D plot of the PDF(m1, m2)."""

    plot_x, plot_y = numpy.meshgrid(
        numpy.linspace(1.00,
                       1.05,
                       300),
        numpy.linspace(0.6,
                       0.8,
                       300)
    )

    with Pool(4) as workers:
        plot_z = numpy.array(
            workers.starmap(constraint.pdf,
                            zip(plot_x.flatten(), plot_y.flatten()))
        ).reshape(plot_x.shape)
    print('Plot z: ' + repr(plot_z))

    axis = pyplot.gca(projection='3d')
    axis.plot_wireframe(plot_x,
                        plot_y,
                        plot_z,
                        color='black')
                        #False positive
                        #pylint: disable=no-member
#                        cmap=cm.coolwarm,
                        #pylint: enable=no-member
#                        linewidth=0,
#                        edgecolor='none')
    pyplot.show()

def calculate_cdf_column(constraint, secondary_masses, primary_mass):
    """Calculane the values of the CDF at all m2 for fixed m1."""

    m2_distro = constraint.get_conditional_secondary_mass_distribution(
        primary_mass
    )
    return m2_distro.cdf(secondary_masses)

def plot_m2_cdf(constraint):
    """Display a 3-D plot of CDF(m2|m1)."""

    secondary_masses = numpy.linspace(constraint.mass_range[0],
                                      constraint.mass_range[1],
                                      3000)

    axis = pyplot.gca(projection='3d')
    primary_masses = numpy.vectorize(
        constraint.primary_mass_ppf
    )(
        numpy.linspace(1e-5, 1.0 - 1e-5, 300)
    )

    with Pool(4) as workers:
        plot_z = numpy.stack(
            workers.map(
                partial(calculate_cdf_column, constraint, secondary_masses),
                primary_masses
            )
        )

    plot_x, plot_y = numpy.meshgrid(secondary_masses, primary_masses)

    axis = pyplot.gca(projection='3d')
    axis.plot_wireframe(plot_x,
                        plot_y,
                        plot_z,
                        color='black',
                        rstride=1,
                        cstride=1,
                        linewidth=0.1)
    pyplot.show()

def main():
    """Avoid polluting global namespace."""

    set_start_method('forkserver')
    logging.basicConfig(level=logging.DEBUG)

    ngc188_photometry = read_cds_pipe_table(
        os.path.join(
            data_dir,
            'Stetson_et_al_04_NGC188_UBVRI_photometry.tsv'
        )
    )

#    ngc188_single_lined_binaries = read_cds_pipe_table(
#        os.path.join(
#            data_dir,
#            'Geller_et_al_2009_WIYN_single_lined_orbits.tsv'
#        )
#    )


    selected_photometry = ngc188_photometry[ngc188_photometry['PKM'] == 4080][0]
    print(repr(selected_photometry))

    measured_photometry = dict()
    print('Membership probability: %s'
          %
          repr(selected_photometry['Memb']))
    for mag in 'UBVRI':
        if numpy.isfinite(selected_photometry[mag + 'mag']):
            measured_photometry[mag] = stats.norm(
                selected_photometry[mag + 'mag'],
                max(selected_photometry['e_' + mag + 'mag'], 0.05)
            )
    print('Measured photometry: ')
    for quantity, distro in measured_photometry.items():
        print('\t%s: %s' % (quantity, repr(distro.args)))

    interpolator = CMDPhotometryInterpolator(
        os.path.join(
            data_dir,
            'CMD_7.0Gyr_FeH0dex_isochrone_Av0.2_UBVRIJHK.dat'
        )
    )

    constraint = PhotometricConstraint([interpolator],
                                       measured_photometry,
                                       11.23,
                                       'photometric_constraints.pkl')

    plot_joint_pdf(constraint)
    plot_m1_cdf(constraint)
    plot_m2_cdf(constraint)

if __name__ == '__main__':
    main()
