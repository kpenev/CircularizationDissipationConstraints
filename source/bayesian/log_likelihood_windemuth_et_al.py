"""Define a log-likelihood class for Windemuth et. al. (2019) EBs."""

import logging

import numpy
import scipy
from bayesian.log_likelihood_binary_stars import LogLikelihoodBinaryStars
#from general_purpose_python_modules.reproduce_system import find_evolution #TODO: re-enable this when the testing is over
from bayesian.windemuth_eccentricity_distribution import W19EccentricityDistribution
from functools import partial
from stellar_evolution.manager import StellarEvolutionManager
from orbital_evolution.transformations import phase_lag

def find_evolution(parameters,envelope_eccentricity=None):
    initial_eccentricity = parameters['initial_eccentricity']
    final_period = parameters['system'].orbital_period
    final_eccentricity = parameters['system'].eccentricity

    print('initial_eccentricity = %s' % (repr(initial_eccentricity)))
    print('initial_period = %s' % (repr(final_period)))
    print('final_eccentricity = %s' % (repr(final_eccentricity)))

    if initial_eccentricity != 'solve':
        #final_eccentricity = initial_eccentricity**2
        change = initial_eccentricity / 2
        print('change = %s' % (repr(change)))
        final_eccentricity = initial_eccentricity - change if (initial_eccentricity - change) >= 0.1 else 0.1
        delta_eccentricity = 0.01
        initial_eccentricity_2 = initial_eccentricity + delta_eccentricity if (initial_eccentricity + delta_eccentricity) <= 0.8 else initial_eccentricity - delta_eccentricity
        change = initial_eccentricity_2 / 2
        final_eccentricity_2 = initial_eccentricity_2 - change if (initial_eccentricity_2 - change) >= 0.1 else 0.1
    else:
        #initial_eccentricity = final_eccentricity**(1/2) if final_eccentricity < .64 else numpy.nan
        change = final_eccentricity / 2
        print('change = %s' % (repr(change)))
        initial_eccentricity = final_eccentricity + change if (final_eccentricity + change) <= 0.8 else 0.8
        delta_eccentricity = 0.01
        final_eccentricity_2 = final_eccentricity + delta_eccentricity if (final_eccentricity + delta_eccentricity) <= 0.8 else final_eccentricity - delta_eccentricity
        change = final_eccentricity_2 / 2
        initial_eccentricity_2 = final_eccentricity_2 + change if (final_eccentricity_2 + change) <= 0.8 else 0.8
    #numpy.sqrt(initial_eccentricity) / 2 if initial_eccentricity > .25 else initial_eccentricity / 2
    initial_period = final_period * 1.1 if final_period <= 35 else numpy.nan
    print('initial_period = %s' % (repr(initial_period)))

    #INSERT

    print('initial_eccentricity = %s' % (repr(initial_eccentricity)))
    print('initial_period = %s' % (repr(final_period)))
    print('final_eccentricity = %s' % (repr(final_eccentricity)))
    print('initial_eccentricity_2 = %s' % (repr(initial_eccentricity_2)))
    print('final_eccentricity_2 = %s' % (repr(final_eccentricity_2)))

    #ehat_prime = (final_eccentricity - initial_eccentricity) / (final_period - initial_period)
    ehat_prime = (final_eccentricity - final_eccentricity_2) / (initial_eccentricity - initial_eccentricity_2)
    print('ehat_prime = %s' % (repr(ehat_prime)))

    #if envelope_eccentricity is not None:
    #    final_eccentricity = 0.2#envelope_eccentricity * (9/10)

    print((
            numpy.array([final_period,final_eccentricity]),
            numpy.array([ehat_prime,initial_eccentricity])
        )
    )
    return (
        numpy.array([final_period,final_eccentricity]),
        numpy.array([ehat_prime,initial_eccentricity])
    )

class LogLikelihoodWindemuth(LogLikelihoodBinaryStars):
    """The log-likelihood for Windemuth et. al. (2019) EBs."""

    def __init__(self,
                 *parent_args,
                 envelope_eccentricity,
                 observed_eccentricity_distro,
                 de_distro,
                 pe_distro,
                 **parent_kwargs):
        """
        Prepare the log-likelihood function.

        Args:
            parent_args:    Passed directly to parent`s ``__init__``.

        """

        super().__init__(*parent_args,
                         envelope_eccentricity=envelope_eccentricity,
                         **parent_kwargs)
        self.envelope_weights = observed_eccentricity_distro.cdf(
            envelope_eccentricity
        )
        self._observed_eccentricity_distro = observed_eccentricity_distro
        logging.getLogger(__name__).debug(
            'Init: envelope_weights = %s\nSum: %s',
            repr(self.envelope_weights),
            repr(self.envelope_weights.sum())
        )

        self._de = de_distro
        self._pe = pe_distro

    def _choose_solver(self, parameters, solver = '1d', final_e = 0):
        """Handle the parameters passed to the log-likelihood function."""

        if solver == '1d':
            parameters['solve'] = True
            parameters['initial_eccentricity'] = 0.8
        #if 'period' in parameters:
        #    dsd
        #if 'eccentricity' in parameters:
        #    dfdg
        elif solver == '2d':
            parameters['initial_eccentricity'] = 'solve'
            parameters['system'].eccentricity = final_e
        else:
            raise ValueError('solver must be either 1d or 2d')

        return parameters
    
    def _likelihood_integrand(self,e,ehat_approx, GARY):
        ehat_prime = ehat_approx.deriv()
        print('e = %s' % (repr(e)))

        if ehat_approx.degree() == 2:
            ehat_approx_copy = ehat_approx.copy()
            ehat_approx_copy.coef[0] -= e
            inverse_ehat_approx = numpy.polynomial.polynomial.Polynomial((ehat_approx_copy.coef)).roots()
            inverse_e = numpy.real(inverse_ehat_approx[numpy.isreal(inverse_ehat_approx)])
            # print("!~~~~~~DEBUG~~~~~~!")
            # print('ehat_approx = %s' % (repr(ehat_approx)))
            # print('ehat_prime = %s' % (repr(ehat_prime)))
            print('e = %s' % (repr(e)))
            # print('ehat_approx_copy = %s' % (repr(ehat_approx_copy)))
            # print('inverse_ehat_approx = %s' % (repr(inverse_ehat_approx)))
            print('inverse_e = %s' % (repr(inverse_e)))
            for option in inverse_e:
                if option < 0 or option > 0.8:
                    inverse_e = inverse_e[inverse_e != option]
            print('New inverse_e = %s' % (repr(inverse_e)))
            if inverse_e.size == 0:
                #raise ValueError('No real roots found for ehat_approx_copy (inverse_ehat_approx): %s (%s)',repr(ehat_approx_copy),repr(inverse_ehat_approx))
                # this will be an error when we're not doing debug
                #TODO
                print('No in-range real roots found for ehat_approx_copy (inverse_ehat_approx): %s (%s)' % (repr(ehat_approx_copy),repr(inverse_ehat_approx)))
                inverse_e=numpy.nan
            else:
                print('inverse_e-GARY = %s' % (repr(inverse_e-GARY)))
                print('numpy.abs(inverse_e-GARY) = %s' % (repr(numpy.abs(inverse_e-GARY))))
                print('numpy.argmin(numpy.abs(inverse_e-GARY)) = %s' % (repr(numpy.argmin(numpy.abs(inverse_e-GARY)))))
                inverse_e = inverse_e[numpy.argmin(numpy.abs(inverse_e-GARY))]
            print('chosen inverse_e = %s' % (repr(inverse_e)))
        elif ehat_approx.degree() == 1:
            reverse = numpy.polynomial.polynomial.Polynomial((-ehat_approx.coef[0]/ehat_approx.coef[1],1/ehat_approx.coef[1]))
            print('reverse = %s' % (repr(reverse)))
            inverse_e = reverse(e)
            print('inverse_e = %s' % (repr(inverse_e)))
        else:
            raise ValueError('ehat_approx should be a polynomial of order 1 or 2. Something is wrong with ehat_approx: %s',repr(ehat_approx))
        
        if inverse_e > 0.8 or inverse_e < 0:
            raise ValueError('inverse_e should not be outside range: %s',repr(inverse_e))
        
        result = self._de.pdf(e) * self._pe.cdf((ehat_approx(inverse_e))) / ehat_prime(inverse_e) #TODO: am I calling the right Xpfs here?
        print('self._de.pdf(e) = %s' % (repr(self._de.pdf(e))))
        print('ehat_approx(inverse_e) = %s' % (repr(ehat_approx(inverse_e))))
        print('self._pe.cdf((ehat_approx(inverse_e))) = %s' % (repr(self._pe.cdf((ehat_approx(inverse_e))))))
        print('ehat_prime(inverse_e) = %s' % (repr(ehat_prime(inverse_e))))
        print('result = %s' % (repr(result)))
        return result

    def _calculate_likelihood(self, ehat_approx, GARY, e_max): #  ef_max,ei_med,dehat_med
        specific_integrand = partial(self._likelihood_integrand,ehat_approx=ehat_approx, GARY = GARY)
        I = scipy.integrate.quad(specific_integrand, 0, e_max)[0]
        return I

    def calculate_log_likelihood(self,
                                 parameters,
                                 **other_args):
        """Evaluate the log-likelihood at the given model parameters."""

        logger = logging.getLogger(__name__)

        assert 'sample_weights_envelope' in other_args #why? TODO

        # BEGIN PSEUDOCODE
        GARY=0.5
        #Work out parameters such that when we pass it to find_evolution, we get the 1D solver in the way we want
        parameters = self._choose_solver(parameters,'1d')
        #max_final_eccentricity = find_evolution(**parameters)[0].eccentricity[-1]
        max_final_eccentricity = find_evolution(parameters)[0][1]
        negligible = 1e-3
        De_min,De_max = self._de.support()

        print('envelope_eccentricity = %s' % (repr(self.envelope_eccentricity)))

        # Heaviside
        if (
            (max_final_eccentricity > self.envelope_eccentricity)
            or
            (max_final_eccentricity is None)
            or
            numpy.isnan(max_final_eccentricity)
        ):
                print('uh ohhhhh')
                return -numpy.inf
        
        median_e = self._de.ppf(0.5)

        De_away_from_zero = De_min > 0
        De_at_zero = not De_away_from_zero
        De_from_zero_to_max = self._de.cdf(max_final_eccentricity)

        logger.debug('Various values:')
        logger.debug('median_e = %s',repr(median_e))
        logger.debug('max_final_eccentricity = %s',repr(max_final_eccentricity))
        logger.debug('De range = %s',repr((De_min,De_max)))
        logger.debug('Envelope eccentricity = %s',repr(self.envelope_eccentricity))
        logger.debug('De_away_from_zero = %s',repr(De_away_from_zero))
        logger.debug('De_at_zero = %s',repr(De_at_zero))
        logger.debug('De_from_zero_to_max = %s',repr(De_from_zero_to_max))
        print('Various values:')
        print('median_e = %s' % (repr(median_e)))
        print('max_final_eccentricity = %s' % (repr(max_final_eccentricity)))
        print('De range = %s' % (repr((De_min,De_max))))
        print('Envelope eccentricity = %s' % (repr(self.envelope_eccentricity)))
        print('De_away_from_zero = %s' % (repr(De_away_from_zero)))
        print('De_at_zero = %s' % (repr(De_at_zero)))
        print('De_from_zero_to_max = %s' % (repr(De_from_zero_to_max)))

        test_min = 0
        test_max = 0.8

        if (
            #De_from_zero_to_max <= negligible
            #or
            (
                De_away_from_zero
                and
                max_final_eccentricity < De_min
            ) # D_e is negligible near e=0, and e_hat(e_i=0.8) is below D_e.
            or
            (
                De_at_zero
                and
                max_final_eccentricity <= De_max
            ) # D_e(e_final=0) is not negligible, and e_hat(e_i=0.8) is in or below D_e.
        ):
            logger.debug('In a case where we assume ehat is linear from 0 to 0.8')
            print('In a case where we assume ehat is linear from 0 to 0.8')
            print('Case 1: De_from_zero_to_max <= negligible: %s' % (repr(De_from_zero_to_max <= negligible)))
            print('Case 2: De_away_from_zero and max_final_eccentricity < De_min: %s' % (repr(De_away_from_zero and max_final_eccentricity < De_min)))
            print('Case 3: De_at_zero and max_final_eccentricity <= De_max: %s' % (repr(De_at_zero and max_final_eccentricity <= De_max)))
            ehat_approx = numpy.polynomial.polynomial.Polynomial((0,max_final_eccentricity / 0.8))
        elif (
            De_away_from_zero
            and
            (
                self._de.cdf(self.envelope_eccentricity)
                -
                De_from_zero_to_max
            ) <= negligible
        ): # If D_e is clearly away from e_final=0 and e_hat(e_i=0.8) is below the envelope but above D_e
            logger.debug('In a case where we assume ehat is linear in the range where D_e is non-negligible')
            test_min,test_max = De_min,De_max
            print('In a case where we assume ehat is linear in the range where D_e is non-negligible')
            parameters = self._choose_solver(parameters,'2d',median_e)
            #result = find_evolution(**parameters)
            result = find_evolution(parameters)
            print('result = %s' % (repr(result)))
            #init_e,ehat_prime = result[0].eccentricity[0],result[1][0]
            init_e,ehat_prime = result[1][1],result[1][0]
            yintercept = median_e - ehat_prime*init_e
            ehat_approx = numpy.polynomial.polynomial.Polynomial((yintercept,ehat_prime))
        elif (
            (
                De_away_from_zero
                and
                (
                    De_min <= max_final_eccentricity <= De_max
                )
            ) # If D_e is negligible near e=0, but e_hat(e_i=0.8) is sowhere where D_e is not negligible
            or
            (
                De_at_zero
                and
                max_final_eccentricity > De_max
            ) # D_e(e_final=0) is not negligible, and e_hat(e_i=0.8) is above D_e
        ):
            logger.debug('In a case where we want to use quadratic approximation.')
            print('In a case where we want to use quadratic approximation.')
            e_to_match,e_max_initial,e_max_final = (De_min,0.8,max_final_eccentricity) if De_away_from_zero else (De_max,0,0)
            logger.debug('Values if De_away_from_zero: e_to_match = %s, e_max_initial = %s, e_max_final = %s',repr(De_min),repr(0.8),repr(max_final_eccentricity))
            logger.debug('Values if De_at_zero: e_to_match = %s, e_max_initial = %s, e_max_final = %s',repr(De_max),repr(0),repr(0))
            logger.debug('e_to_match = %s',repr(e_to_match))
            logger.debug('e_max_initial = %s',repr(e_max_initial))
            logger.debug('e_max_final = %s',repr(e_max_final))
            print('Values if De_away_from_zero: e_to_match = %s, e_max_initial = %s, e_max_final = %s' % (repr(De_min),repr(0.8),repr(max_final_eccentricity)))
            print('Values if De_at_zero: e_to_match = %s, e_max_initial = %s, e_max_final = %s' % (repr(De_max),repr(0),repr(0)))
            print('e_to_match = %s' % (repr(e_to_match)))
            print('e_max_initial = %s' % (repr(e_max_initial)))
            print('e_max_final = %s' % (repr(e_max_final)))
            parameters = self._choose_solver(parameters,'2d',e_to_match)
            #result = find_evolution(**parameters)
            result = find_evolution(parameters)
            print('result = %s' % (repr(result)))
            #init_e,ehat_prime = result[0].eccentricity[0],result[1][0][0]
            init_e,ehat_prime = result[1][1],result[1][0]
            print('ehat_prime = %s' % (repr(ehat_prime)))
            print('e_to_match = %s' % (repr(e_to_match)))
            print('init_e = %s' % (repr(init_e)))
            GARY=init_e
            A = numpy.matrix([
                                [init_e**2,        init_e,        1],
                                [e_max_initial**2, e_max_initial, 1],
                                [2*init_e,         1,             0]
                            ])
            B = numpy.matrix([e_to_match,e_max_final,ehat_prime])
            fit = scipy.linalg.lstsq(A,B.T)[0]
            ehat_approx = numpy.polynomial.polynomial.Polynomial((fit[2][0],fit[1][0],fit[0][0]))
            print('ehat_approx = %s' % (repr(ehat_approx)))
            print('HEADS UP')
            print('e_to_match = %s' % (repr(e_to_match)))
            print('e_max_initial = %s' % (repr(e_max_initial)))
            print(-ehat_approx.coef[1]/(2*ehat_approx.coef[2]))
            if e_to_match < -ehat_approx.coef[1]/(2*ehat_approx.coef[2]) < e_max_initial:
                raise ValueError('We need to revert back to linear.')
        else:
            logger.error('Something went wrong in the Windemuth likelihood function. Please check the code.')
            raise ValueError('Something went wrong in the Windemuth likelihood function. Please check the code.')
        
        ehat_prime = ehat_approx.deriv() #TODO (after debugging): remove, already being handled by _calculate_likelihood

        logger.debug('ehat_approx = %s',repr(ehat_approx))
        logger.debug('ehat_prime = %s',repr(ehat_prime))
        print('ehat_approx = %s' % (repr(ehat_approx)))
        print('ehat_prime = %s' % (repr(ehat_prime)))

        #for e in numpy.linspace(0,0.8,20):
        #    print('ehat_approx(%s) = %s' % (repr(e),repr(ehat_approx(e))))
        
        #print('INTERCEPTED! WE testing why integrand has discontinuity at max_final_eccentricity')
        for e in numpy.linspace(test_min,test_max,20):
            self._likelihood_integrand(e,ehat_approx,GARY)
        raise ValueError
        
        likelihood = self._calculate_likelihood(ehat_approx,GARY)
        #assert likelihood >= 0
        #if likelihood == 0:
        #    return -numpy.inf
        
        logger.debug('likelihood = %s',repr(likelihood))
        logger.debug('log(likelihood) = %s',repr(numpy.log(likelihood)))
        print('likelihood = %s' % (repr(likelihood)))
        print('log(likelihood) = %s' % (repr(numpy.log(likelihood))))

        return numpy.log(likelihood)
        ########################################### END PSEUDOCODE

        efinal_cdfs = self._observed_eccentricity_distro.eval_sample_cdf(
            final_eccentricity
        )
        numerator = (
            efinal_cdfs
            /
            self.envelope_weights
            *
            other_args['sample_weights_envelope']
        ).sum()

        denominator = other_args['sample_weights_envelope'].sum()

        logger.debug('Observed e CDF(%s) = %s\nSum: %s',
                     repr(final_eccentricity),
                     repr(efinal_cdfs),
                     repr(efinal_cdfs.sum()))
        logger.debug('envelope_weights = %s\nSum: %s',
                     repr(self.envelope_weights),
                     repr(self.envelope_weights.sum()))

        logger.debug('W19 likelihood = %s / %s',
                     repr(numerator),
                     repr(denominator))

        assert numerator >= 0
        if numerator == 0:
            return -numpy.inf

        return numpy.log(numerator) - numpy.log(denominator)
    
    def test_internal_functions(self):
        logger = logging.getLogger(__name__)

        # Test_likelihood_integrand
        logger.debug('Testing _likelihood_integrand')
        print('Testing _likelihood_integrand')
        e_set = numpy.linspace(0,0.8,20)
        ehat_set = (
            #numpy.polynomial.polynomial.Polynomial((1)),
            numpy.polynomial.polynomial.Polynomial((0,1)),
            numpy.polynomial.polynomial.Polynomial((1,1)),
            numpy.polynomial.polynomial.Polynomial((0,0,1)),
            numpy.polynomial.polynomial.Polynomial((-1,0,1)),
            numpy.polynomial.polynomial.Polynomial((0,1,1)),
            numpy.polynomial.polynomial.Polynomial((-1,1,1))
        )
        for e in e_set:
            for ehat in ehat_set:
                logger.debug('e = %s, ehat = %s',repr(e),repr(ehat))
                print('e = %s, ehat = %s' % (repr(e),repr(ehat)))
                logger.debug('Integrand value is %s',repr(self._likelihood_integrand(e,ehat)))
                print('Integrand value is %s' % (repr(self._likelihood_integrand(e,ehat))))

        # Test _calculate_likelihood
        logger.debug('Testing _calculate_likelihood')
        print('Testing _calculate_likelihood')
        ehat_set = (
            #numpy.polynomial.polynomial.Polynomial((1)),
            numpy.polynomial.polynomial.Polynomial((0,1)),
            numpy.polynomial.polynomial.Polynomial((1,1)),
            numpy.polynomial.polynomial.Polynomial((0,0,1)),
            numpy.polynomial.polynomial.Polynomial((-1,0,1)),
            numpy.polynomial.polynomial.Polynomial((0,1,1)),
            numpy.polynomial.polynomial.Polynomial((-1,1,1))
        )
        for ehat in ehat_set:
            logger.debug('ehat = %s',repr(ehat))
            print('ehat = %s' % (repr(ehat)))
            logger.debug('Integral value is %s',repr(self._calculate_likelihood(ehat)))
            print('Integral value is %s' % (repr(self._calculate_likelihood(ehat))))
        
        print('Testing complete.')
        
        return 1
    
    def test_external_functions(self,interpolator):
        logger = logging.getLogger(__name__)
        from astropy import units, constants
        from types import SimpleNamespace

        # Test calculate_log_likelihood
        logger.debug('Testing calculate_log_likelihood')
        # print('Testing calculate_log_likelihood')
        # parameters = dict(
        #     interpolator=interpolator,
        #     #dissipation =314827856756252354432,
        #     system=SimpleNamespace(
        #         primary_mass=1 * units.M_sun,
        #         secondary_mass=1 * units.M_sun,
        #         feh=0.4,
        #         orbital_period=10 * units.day,
        #         age=1.2 * units.Gyr,
        #         eccentricity=0.3
        #     ),
        #     dissipation = dict(
        #         primary = dict(
        #             reference_phase_lag = 0.0
        #         ),
        #         secondary = dict(
        #             reference_phase_lag = 0.0
        #         )
        #     ),
        #     max_age =None,
        #     initial_porb =27.3 * units.day,
        #     initial_eccentricity =0.0,
        #     initial_obliquity =0.0,
        #     disk_period=1 * units.day,
        #     disk_dissipation_age =2e-3 * units.Gyr,
        #     primary_wind_strength =0.17,
        #     primary_wind_saturation =2.78,
        #     primary_core_envelope_coupling_timescale =0.05 * units.Gyr,
        #     secondary_wind_strength =0.0,
        #     secondary_wind_saturation =100.0,
        #     secondary_core_envelope_coupling_timescale =0.05 * units.Gyr,
        #     secondary_disk_period =None,
        #     orbital_period_tolerance =1e-6,
        #     eccentricity_tolerance =1e-6,
        #     obliquity_tolerance =1e-6,
        #     period_search_factor =2.0,
        #     scaled_period_guess =1.0,
        #     eccentricity_upper_limit =0.8,
        #     solve =True,
        #     max_iterations =49,
        #     secondary_is_star =None,
        #     precision = 1e-2
        # )
        parameters = dict(
            system=SimpleNamespace(
                orbital_period=27.3,
                age=5.89,
                eccentricity=0.44
            ),
            initial_porb =35,
            initial_eccentricity =0.7,
            solve =True,
        )
        # There should be other setup but I haven't thought through what the edge cases I need to test are.
        # Q=6.0
        # parameters['dissipation']['primary']['reference_phase_lag'] = phase_lag(Q)
        # parameters['dissipation']['primary']['tidal_frequency_breaks'] = numpy.array((0.0,))
        # parameters['dissipation']['primary']['spin_frequency_breaks'] = numpy.array((0.0,))
        # parameters['dissipation']['primary']['tidal_frequency_powers'] = numpy.array((0.0,))
        # parameters['dissipation']['primary']['spin_frequency_powers'] = numpy.array((0.0,))
        # parameters['dissipation']['secondary']['reference_phase_lag'] = phase_lag(Q)
        # parameters['dissipation']['secondary']['tidal_frequency_breaks'] = numpy.array((0.0,))
        # parameters['dissipation']['secondary']['spin_frequency_breaks'] = numpy.array((0.0,))
        # parameters['dissipation']['secondary']['tidal_frequency_powers'] = numpy.array((0.0,))
        # parameters['dissipation']['secondary']['spin_frequency_powers'] = numpy.array((0.0,))
        # print
        self.calculate_log_likelihood(parameters,sample_weights_envelope=0)

        return 1

if __name__ == '__main__':
    
    # Testing
    interpolator = StellarEvolutionManager('/home/vortebo/ctime/poet/stellar_evolution_interpolators').get_interpolator_by_name(
        'default'
    )
    bob=LogLikelihoodWindemuth(
        envelope_eccentricity=0.7,
        observed_eccentricity_distro=scipy.stats.uniform(loc=0,scale=0.8),
        de_distro=W19EccentricityDistribution(12356914,pickle_fname='/home/vortebo/ctime/CircularizationDissipationConstraints/windemuth_eccentricity_distros.pkl'),
        pe_distro=scipy.stats.uniform(loc=0,scale=0.8),
        powerlaw_dissipation = 5,
        interpolator=interpolator,
        evolution_timeout = 1.0,
        period_search_factor = 2.0,
        scaled_period_guess = 1.0
    )
    #bob.test_internal_functions()
    bob.test_external_functions(interpolator)