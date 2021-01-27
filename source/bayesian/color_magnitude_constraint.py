#!/usr/bin/env python3

"""Interface for working with mass constraints from color-mag measurements."""

from multiprocessing import Pool, set_start_method
import os.path
from pickle import Pickler, Unpickler
import logging

from matplotlib import pyplot, cm
from scipy import integrate, stats, optimize
import numpy

from planetary_system_io import read_cds_pipe_table
#False positive (fixed in __init__.py)
#pylint: disable=import-error
from cmd_utils import CMDPhotometryInterpolator
from command_line_utilities import data_dir
#pylint: enable=import-error

def _identity(x):
    return x

class ColorMagnitudeConstraint:
    """Constraint on component masses from observed color & magnitude."""

    _logger = logging.getLogger(__name__)

    def _joint_likelihood(self, secondary_mass, primary_mass, return_log=False):
        """The likelihood of the observed colors and magnitudes given masses."""

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
                predicted_value = predicted_photometry[quantity.strip()]

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
                                  **self._integration_config)[0]

        minimization = optimize.minimize(
            lambda masses: -self._joint_likelihood(*masses, return_log=True),
            numpy.full((2,), 0.5 * (self.mass_range[0] + self.mass_range[1])),
            bounds=(self.mass_range, self.mass_range)
        )
        assert minimization.success
        self._logger.debug('Maximum log-likelihood: %s',
                           repr(-minimization.fun))

        result = integrate.solve_ivp(
            integrand,
            self.mass_range,
            numpy.array([0.0]),
            dense_output=True,
            max_step=1e-2,
            rtol=1e-6,
            atol=1e-9 * numpy.exp(-minimization.fun)
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
                    if section == 'ColorMagnitudeConstraint':
                        assert nobjects == 4
                        nobjects -= 1
                        if(
                                tuple(
                                    interp.isochrone_fname
                                    for interp in self._photometry_interpolators
                                )
                                ==
                                unpickler.load()
                        ):
                            nobjects -= 1
                            if compare_measured_photometry(unpickler.load()):
                                nobjects -= 1
                                if(
                                        self._integration_config
                                        ==
                                        unpickler.load()
                                ):
                                    self._logger.debug('Found matching pickle')
                                    return unpickler.load()
                                self._logger.debug(
                                    'Integration configuration does not '
                                    'match.'
                                )
                        else:
                            self._logger.debug('Interpolator isochrones do not '
                                               'match')

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
            pickler.dump(('ColorMagnitudeConstraint', 4))
            pickler.dump(
                tuple(interp.isochrone_fname
                      for interp in self._photometry_interpolators)
            )
            pickler.dump(self._measured_photometry)
            pickler.dump(self._integration_config)
            pickler.dump(self._cumulative_m1_likelihood)

    def __init__(self,
                 photometry_interpolators,
                 measured_photometry,
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
        self._cumulative_m1_likelihood = self._check_for_pickled(pickle_fname)
        if self._cumulative_m1_likelihood is None:
            self._cumulative_m1_likelihood = self._prepare_constraint()
            self._add_to_pickle_file(pickle_fname)

        self._norm = self._cumulative_m1_likelihood(self.mass_range[1])[0]
        print('Norm: ' + repr(self._norm))

    def joint_pdf(self, primary_mass, secondary_mass):
        """The joint PDF of component masses given observed colors and mags."""

        if(
                secondary_mass > primary_mass
                or
                secondary_mass < self.mass_range[0]
                or
                primary_mass > self.mass_range[1]
        ):
            return 0.0
        return self._joint_likelihood(
            secondary_mass,
            primary_mass
        ) / self._norm

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

    def secondary_mass_cdf(self, primary_mass, secondary_mass):
        """CDF(m2|m1)."""

        return (
            integrate.quad(self._joint_likelihood,
                           self.mass_range[0],
                           secondary_mass,
                           args=(primary_mass,),
                           points=self._quad_points,
                           **self._integration_config)[0]
            /
            integrate.quad(self._joint_likelihood,
                           self.mass_range[0],
                           primary_mass,
                           args=(primary_mass,),
                           points=self._quad_points,
                           **self._integration_config)[0]
        )

def main():
    """Avoid polluting global namespace."""

    set_start_method('forkserver')
    logging.basicConfig(level=logging.DEBUG)

    ngc_188_photometry = read_cds_pipe_table(
        os.path.join(
            data_dir,
            'Stetson_et_al_04_NGC188_UBVRI_photometry.tsv'
        )
    )
    measured_photometry = dict()
    index = 28
    print('Membership probability(%d): %s'
          %
          (index, repr(ngc_188_photometry[index]['Memb'])))
    for mag in 'UBVRI':
        if numpy.isfinite(ngc_188_photometry[index][mag + 'mag']):
            measured_photometry[mag] = stats.norm(
                ngc_188_photometry[index][mag + 'mag'],
                ngc_188_photometry[index]['e_' + mag + 'mag']
            )

    interpolator = CMDPhotometryInterpolator(
        os.path.join(
            data_dir,
            'CMD_7.0Gyr_FeH0dex_isochrone_Av0.2_UBVRIJHK.dat'
        )
    )

    constraint = ColorMagnitudeConstraint([interpolator],
                                          measured_photometry,
                                          'color_magnitude_constraints.pkl')
    plot_x, plot_y = numpy.meshgrid(
        numpy.linspace(constraint.mass_range[0],
                       constraint.mass_range[1],
                       300),
        numpy.linspace(constraint.mass_range[0],
                       constraint.mass_range[1],
                       300)
    )

    with Pool(4) as workers:
        plot_z = numpy.array(
            workers.starmap(constraint.joint_pdf,
                            zip(plot_x.flatten(), plot_y.flatten()))
        ).reshape(plot_x.shape)
    print('Plot z: ' + repr(plot_z))

    axis = pyplot.gca(projection='3d')
    axis.plot_surface(plot_x,
                      plot_y,
                      plot_z,
                      #False positive
                      #pylint: disable=no-member
                      cmap=cm.coolwarm,
                      #pylint: enable=no-member
                      linewidth=0,
                      antialiased=False)
    pyplot.show()
    plot_x = numpy.linspace(constraint.mass_range[0],
                            constraint.mass_range[1],
                            1000)
    with Pool(4) as workers:
        plot_z = numpy.array(workers.map(constraint.primary_mass_pdf, plot_x))
    print('PDF(M1): ' + repr(plot_z))
    pyplot.plot(plot_x, plot_z)
    pyplot.show()

    plot_z = numpy.vectorize(constraint.primary_mass_cdf)(plot_x)
    pyplot.plot(plot_x, plot_z)
    pyplot.show()

    plot_x, plot_y = numpy.meshgrid(
        numpy.linspace(constraint.mass_range[0],
                       constraint.mass_range[1],
                       300),
        numpy.linspace(constraint.mass_range[0],
                       constraint.mass_range[1],
                       300)
    )
    with Pool(4) as workers:
        plot_z = numpy.array(
            workers.starmap(constraint.secondary_mass_cdf,
                            zip(plot_x.flatten(), plot_y.flatten()))
        ).reshape(plot_x.shape)
    print('Plot z: ' + repr(plot_z))

    axis = pyplot.gca(projection='3d')
    axis.plot_surface(plot_x,
                      plot_y,
                      plot_z,
                      #False positive
                      #pylint: disable=no-member
                      cmap=cm.coolwarm,
                      #pylint: enable=no-member
                      linewidth=0,
                      antialiased=False)
    pyplot.show()

if __name__ == '__main__':
    main()
