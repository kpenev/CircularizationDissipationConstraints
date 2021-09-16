"""Interface for working with mass constraints from color-mag measurements."""

from multiprocessing import Pool
import os.path
from pickle import Pickler, Unpickler
from functools import partial
import logging

from matplotlib import pyplot, cm
from astropy import units, constants
import numpy

#False positive (fixed in __init__.py)
#pylint: disable=import-error
from sample_binary_masses import SampleBinaryMasses
#pylint: enable=import-error
from mass_fitting import fit_binary_masses

#Simplifying decreases readability
#pylint: disable=too-many-instance-attributes
class PhotometricConstraint(SampleBinaryMasses):
    """Constraint on component masses from observed color & magnitude."""

    _logger = logging.getLogger(__name__)

    def _check_constraints(self, primary_mass, secondary_mass):
        """Return True iff the given masses are allowed."""

        if not super()._check_constraints(primary_mass, secondary_mass):
            return False

        for (
                interpolator_index,
                filter_index,
                min_difference
        ) in self._min_magnitude_difference:
            try:
                primary_mag, secondary_mag = self._photometry_interpolators[
                    interpolator_index
                ](
                    numpy.array([primary_mass, secondary_mass])
                )[
                    filter_index
                ]
                if secondary_mag - primary_mag < min_difference:
                    return False
            except ValueError:
                self._logger.critical(
                    'Invalid input masses to photometry interpolator: %s, %s',
                    repr(primary_mass),
                    repr(secondary_mass)
                )
                raise

        return True


    def _joint_likelihood(self, secondary_mass, primary_mass, return_log=False):
        """The likelihood of the observed colors and magnitudes given masses."""

        if not self._check_constraints(primary_mass, secondary_mass):
            return -numpy.inf if return_log else 0

        predicted_photometry = dict()
        for interpolator in self._photometry_interpolators:
            predicted_photometry.update(
                zip(
                    interpolator.available_filters,
                    interpolator.get_binary_magnitudes(float(primary_mass),
                                                       float(secondary_mass))
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
                predicted_value = predicted_photometry[quantity.strip()]

            if return_log:
                result += distribution.logpdf(predicted_value)
            else:
                result *= distribution.pdf(predicted_value)
        return result


    def _secondary_mass_quad_points(self, primary_mass):
        """Points arg to quad required for reliable marginalization oven m2."""

        return self._quad_points


    def _get_max_likelihood(self, min_magnitude_difference):
        """Prepare the constraint for use with current configuration."""

        likelihood_maximization = fit_binary_masses(
            photometry_interpolators=self._photometry_interpolators,
            photometry=self._measured_photometry,
            min_mag_difference=min_magnitude_difference
        )

        while not self._check_constraints(*likelihood_maximization.x):
            likelihood_maximization.x[1] -= 1e-5

        assert likelihood_maximization.success

        result = self._joint_likelihood(likelihood_maximization.x[1],
                                        likelihood_maximization.x[0])

        self._logger.debug(
            'Maximum log-likelihood(m1=%s, m2=%s) = %s',
            repr(likelihood_maximization.x[0]),
            repr(likelihood_maximization.x[1]),
            repr(result)
        )
        return result


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
                    if section == 'PhotometricConstraint' and nobjects == 6:
                        isochrone_fnames = unpickler.load()
                        distance_moduli = unpickler.load()
                        min_magnitude_difference = unpickler.load()
                        measured_photometry = unpickler.load()
                        integration_config = unpickler.load()
                        nobjects -= 5

                        if(
                                tuple(
                                    interp.isochrone_fname
                                    for interp in self._photometry_interpolators
                                ) == isochrone_fnames
                                and
                                tuple(
                                    interp.distance_modulus
                                    for interp in self._photometry_interpolators
                                ) == distance_moduli
                                and
                                (
                                    self._min_magnitude_difference
                                    ==
                                    min_magnitude_difference
                                )
                                and
                                compare_measured_photometry(measured_photometry)
                                and
                                (
                                    self._m2_integration_config
                                    ==
                                    integration_config
                                )
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
            pickler.dump(('PhotometricConstraint', 6))
            pickler.dump(
                tuple(interp.isochrone_fname
                      for interp in self._photometry_interpolators)
            )
            pickler.dump(
                tuple(interp.distance_modulus
                      for interp in self._photometry_interpolators)
            )

            pickler.dump(self._min_magnitude_difference)
            pickler.dump(self._measured_photometry)
            pickler.dump(self._m2_integration_config)
            pickler.dump(self._cumulative_m1_likelihood)

    def _show_joint_likelihood_plot(self):
        """Display a plot of the joint likelihood."""

        plot_x, plot_y = numpy.meshgrid(
            numpy.linspace(0.9,
                           1.1,
                           100),
            numpy.linspace(0.50,
                           0.70,
                           100)
        )
        with Pool(4) as workers:
            plot_z = numpy.array(
                workers.starmap(self._joint_likelihood,
                                zip(plot_x.flatten(), plot_y.flatten()))
            ).reshape(plot_x.shape)

        pyplot.pcolormesh(plot_x,
                          plot_y,
                          plot_z,
                          #False positive
                          #pylint: disable=no-member
                          cmap=cm.coolwarm)
                          #pylint: enable=no-member
        pyplot.colorbar()
        pyplot.show()

    def __init__(self,
                 photometry_interpolators,
                 measured_photometry,
                 pickle_fname,
                 min_magnitude_difference=None,
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

            min_magnitude_difference(dict):    Dictionary indexed by filter name
                specifying a minimum amount by which the predicted magnitude of
                the secondary in the given filter must exceed that of the
                primary.

            pickle_fname(str):    The name of a file to save/load the current
                constraint for fast re-use.

            integration_config:    Keyword arguments to pass to scipy numerical
                integration functions used to normalize and marginalize
                likelihoods.
        """

        self._photometry_interpolators = photometry_interpolators

        self._min_magnitude_difference = []
        if min_magnitude_difference is not None:
            for filter_name, min_difference in min_magnitude_difference.items():
                for interpolator_index, interpolator in enumerate(
                        photometry_interpolators
                ):
                    if filter_name in interpolator.available_filters:
                        self._min_magnitude_difference.append(
                            (
                                interpolator_index,
                                interpolator.available_filters.index(
                                    filter_name
                                ),
                                min_difference
                            )
                        )

        mass_range = (-float('inf'), float('inf'))
        for interpolator in photometry_interpolators:
            mass_range = (
                max(mass_range[0], interpolator.min_mass),
                min(mass_range[1], interpolator.max_mass)
            )


        self._quad_points = numpy.linspace(*mass_range, 30)
        self._measured_photometry = measured_photometry

#        self._show_joint_likelihood_plot()
        super().__init__(
            mass_range=mass_range,
            m2_integration_config=integration_config,
            m1_solve_ivp_config=dict(
                rtol=1e-6,
                atol=1e-9 * self._get_max_likelihood(min_magnitude_difference),
            ),
            pickle_fname=pickle_fname
        )


    def get_component_radius(self, mass):
        """
        Return the radii of a component at the given mass for ecah interpolator.

        Radius is estimated using log10(g)

        Args:
            mass(float):    The mass at which to evaluate the interpolations.

        Returns:
            numpy.array:
                The radius each interpolator predicts for the given mass.
        """

        surface_g = 10.0**numpy.array([
            float(interp.get_interpolated('logg', mass, None))
            for interp in self._photometry_interpolators
        ]) * units.cm / units.s**2
        return numpy.sqrt(
            constants.G * mass * units.M_sun
            /
            surface_g
        ).to_value(units.R_sun)

#pylint: enable=too-many-instance-attributes


def plot_m1_pdf(constraint):
    """Display a plot of the PDF(primary_mass) marginalized over m2."""

    plot_x = numpy.linspace(constraint.mass_range[0],
                            constraint.mass_range[1],
                            1000)
    with Pool(4) as workers:
        plot_z = numpy.array(workers.map(constraint.primary_mass_pdf,
                                         plot_x))
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

def plot_joint_pdf(constraint, literature_masses=None):
    """Display a 3-D plot of the PDF(m1, m2)."""

    plot_x, plot_y = numpy.meshgrid(
        numpy.linspace(0.9,
                       1.1,
                       300),
        numpy.linspace(0.05,
                       1.0,
                       300)
    )

    with Pool(4) as workers:
        plot_z = numpy.array(
            workers.starmap(constraint.pdf,
                            zip(plot_x.flatten(), plot_y.flatten()))
        ).reshape(plot_x.shape)

    if literature_masses is None:
        axis = pyplot.gca(projection='3d')
        axis.plot_wireframe(plot_x,
                            plot_y,
                            plot_z,
                            color='black')
                            #False positive
                            #pylint: disable=no-member
#                            cmap=cm.coolwarm,
                            #pylint: enable=no-member
#                            linewidth=0,
#                            edgecolor='none')
    else:
        pyplot.pcolormesh(plot_x,
                          plot_y,
                          plot_z,
                          #False positive
                          #pylint: disable=no-member
                          cmap=cm.coolwarm)
                          #pylint: enable=no-member
        pyplot.plot([literature_masses[0]],
                    [literature_masses[1]],
                    'xk',
                    markersize=10,
                    markeredgewidth=3)
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
