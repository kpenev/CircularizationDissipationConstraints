#!/usr/bin/env python3
"""Define the distribution of RV semi-amplitude marginalized over incl."""

from multiprocessing import Pool
import logging

from matplotlib import pyplot
from scipy.stats import norm
from scipy import integrate
from scipy.optimize import root_scalar
from astropy import units as u, constants as c
import numpy

class MarginalizedRVKPDF:
    """The integrand for marginalizing PDF over inclination."""

    _logger = logging.getLogger(__name__)

    def _integrand(self, cos_inclination, max_rv_semiamplitude):
        """
        Turn RV semi-amplitude into the integrand for marginalizing over incl.

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

    def _upper_bound_equation(self, upper_bound, target_prob):
        """The equation to solve in order to find max RV semi-amplitude."""

        integral = integrate.quad(
            lambda s: (self._observed_rvk.sf(s * upper_bound)
                       /
                       numpy.sqrt(1.0 - numpy.square(s))),
            0,
            1,
            epsabs=1e-8 * target_prob,
            epsrel=1e-8,
            points=self._observed_rvk.ppf(numpy.linspace(0.1, 0.9, 9))
        )
        self._logger.debug('Upper bound equation integral at %s: %s',
                           repr(upper_bound),
                           repr(integral))
        return integral[0] - target_prob

    def __init__(self,
                 observed_rvk,
                 max_discarded_probabiity,
                 pickle_fname=None):
        """Set-up the integrand given observed RV semi-amplitude distro."""

        self._observed_rvk = observed_rvk

        max_discarded_probabiity /= 4.0

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

    @staticmethod
    def rv_semi_amplitude(cos_inclination,
                          primary_mass,
                          secondary_mass,
                          eccentricity,
                          orbital_period):
        """Return the RV semi-amplitude the given orbit will have."""

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


    def __call__(self, *args):
        """
        PDF of observer RV semi-amp. marginalized over inclination.

        Args:
            Same as :meth:`rv_semi_amplitude()` after `cos_inclination`.

        Returns:
            float:
                The value of the marginalized PDF evaluated at the given
                arguments.
        """

        return integrate.romberg(self._integrand,
                                 -1.0,
                                 1.0,
                                 args=args,
                                 tol=0,
                                 rtol=1e-6,
                                 divmax=20)

def main():
    """Avoid polluting global namespace."""

    logging.basicConfig(level=logging.DEBUG)

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

    pdf = MarginalizedRVKPDF(norm(system['K1'].to_value('m/s'),
                                  system['errK1'].to_value('m/s')),
                             max_discarded_probabiity=1e-6)
    print(
        'K = '
        +
        repr(
            #False positive
            #pylint: disable=no-member
            pdf.rv_semi_amplitude(
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
    left_plot = pyplot.subplot(121)
    right_plot = pyplot.subplot(122)
    for eccentricity in numpy.linspace(0.5, 0, 6):
        plot_m2 = numpy.linspace(0.15, 0.5, 100)

        with Pool(4) as workers:
            plot_pdf = numpy.array(
                workers.starmap(
                    pdf,
                    [
                        (
                            1.2 * u.M_sun,
                            m2 * u.M_sun,
                            eccentricity,
                            system['Porb']
                        )
                        for m2 in plot_m2
                    ]
                )
            )
        if reference_pdf is None:
            reference_pdf = plot_pdf

        print('e=' + repr(eccentricity))

        left_plot.plot(plot_m2,
                       plot_pdf,
                       label='e = ' + str(eccentricity))
        right_plot.semilogy(plot_m2,
                            plot_pdf / reference_pdf,
                            label='e = ' + str(eccentricity))
    right_plot.set_ylim((0.1, 10.0))
    pyplot.show()

if __name__ == '__main__':
    main()
