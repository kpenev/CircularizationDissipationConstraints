"""Define base class for stellar parameter log-likelihood functions."""

from abc import ABC, abstractmethod

from matplotlib import pyplot
import numpy
from scipy.integrate import solve_ivp
from astropy import units as u

from continous_max_age import get_continuous_max_age

class LogLikelihoodBase(ABC):
    """Common interface for log-likelihood functions of stellar paramseters."""

    interpolator = None

    @classmethod
    def set_interpolator(cls, interpolator):
        """Define a shared interpolator to be used by all log-likelihoods."""

        assert cls.interpolator is None
        cls.interpolator = interpolator

    @abstractmethod
    def _age_cdf_integrand(self, age, _, mass, feh):
        """Unnormalized posterior PDF excluding direct [Fe/H] measurement."""


    def __eq__(self, other):
        """Is other mathematicall the same log-likelihood."""

        return (
            self.default_mass == other.default_mass
            and
            self.default_feh == other.default_feh
            and
            self._solve_ivp_options == other._solve_ivp_options
            and
            self.interpolator.name == other.interpolator.name
            and
            self.interpolator.smoothing == other.interpolator.smoothing
            and
            self.interpolator.nodes == other.interpolator.nodes
            and
            self.interpolator.vs_log_age == other.interpolator.vs_log_age
            and
            self.interpolator.log_quantity == other.interpolator.log_quantity
            and
            self.interpolator.track_masses == other.interpolator.track_masses
            and
            self.interpolator.track_feh == other.interpolator.track_feh
        )

    def __init__(self,
                 mass=1.0 * u.M_sun,
                 feh=0.0,
                 **solve_ivp_options):
        """
        Prepare the log-likelihood for use.

        Args:
            interpolator:     The stellar evolution interpolator on which the
                log-likelihood is based.

            mass(astropy quantity):     The default mass to use when evaluating
                the various PDF and CDF methods.

            feh(float):     The default [Fe/H] to assume when evaluating the
                various PDF and CDF methods.

            solve_ivp_options:     Directly passed to :func:`solve_ivp` when
                calculating the integral of the age PDF.
        """

        self.default_mass = mass
        self.default_feh = feh
        self._solve_ivp_options = solve_ivp_options

        self._get_max_age = get_continuous_max_age(self.interpolator)

        self._age_integrals = dict()

    def __setattr__(self, name, value):
        """Return the default mass assumed by conditional distributions."""

        if name == 'defalut_mass':
            assert (self.interpolator.track_masses[0]
                    <
                    value
                    <
                    self.interpolator.track_masses[-1])
        elif name == 'default_feh':
            assert (self.interpolator.track_feh[0]
                    <
                    value
                    <
                    self.interpolator.track_feh[-1])
        super().__setattr__(name, value)

    def __getstate__(self):
        """
        Do not pickle cached integrals or interpolator (must be set manually).
        """

        return (
            self._get_max_age,
            self.default_mass,
            self.default_feh,
            self._solve_ivp_options
        )

    def __setstate__(self, state):
        """
        Load a pickled likelihood (interpolator must already be defined).
        """

        (
            self._get_max_age,
            self.default_mass,
            self.default_feh,
            self._solve_ivp_options
        ) = state
        assert not hasattr(self, '_age_integrals')
        self._age_integrals = dict()

    def plot_age_cdf_integrand(self, mass, feh, save_as=None):
        """
        Plot the integrant used to calculate CDF(age) and its integral.

        Args:
            mass:     The mass at which to create plots, along with units. If
                iterable, multiple plots are created each labeled by the
                mass & [Fe/H].

            feh:     The [Fe/H] at which to plot. Should have the same shape as
                mass.

            save_as:    The filename to save the plot as. If None, the plot is
                displayed instead.

        Returns:
            None
        """

        try:
            mass_feh_zipped = zip(mass, feh)
        except TypeError:
            print('Not arrays')
            mass_feh_zipped = zip([mass], [feh])


        integrand_plot = pyplot.subplot(211)
        integral_plot = pyplot.subplot(212, sharex=integrand_plot)

        for plot_mass, plot_feh in mass_feh_zipped:
            integral = self.age_integral(plot_mass, plot_feh)
            plot_ages = numpy.linspace(integral.t_min, integral.t_max, 1000)

            label = r'$M_\star=%g,\ [Fe/H]=%g$' % (plot_mass.to_value(u.M_sun),
                                                   plot_feh)

            integrand_plot.plot(
                plot_ages,
                [
                    self._age_cdf_integrand(t,
                                            None,
                                            plot_mass.to_value(u.M_sun),
                                            plot_feh)
                    for t in plot_ages
                ],
                '-',
                label=label
            )

            integral_plot.plot(plot_ages,
                               integral(plot_ages).flatten(),
                               '-')

        integrand_plot.legend()

        if save_as is None:
            pyplot.show()
        else:
            pyplot.savefig(save_as)
            pyplot.clf()

    def age_integral(self, mass=None, feh=None, disable_caching=False):
        """
        Return age integral re-using results when possible.

        If mass and/or feh are None, the default values are used.
        """

        if mass is None:
            mass = self.default_mass

        if feh is None:
            feh = self.default_feh

        if (mass, feh) in self._age_integrals:
            return self._age_integrals[(mass, feh)]

        #False positive
        #pylint: disable=not-callable
        min_age = self.interpolator('radius',
                                    mass.to_value(u.M_sun),
                                    feh).min_age
        #pylint: enable=not-callable
        max_age = self._get_max_age(mass, feh)
        solution = solve_ivp(self._age_cdf_integrand,
                             (min_age, max_age),
                             [0.0],
                             args=(mass.to_value(u.M_sun), feh),
                             dense_output=True,
                             **self._solve_ivp_options)
        assert solution.success
        if not disable_caching:
            self._age_integrals[(mass, feh)] = solution.sol
        return solution.sol
