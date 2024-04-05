"""Define a log-likelihood class for Windemuth et. al. (2019) EBs."""

import logging

import numpy
import scipy
from bayesian.log_likelihood_binary_stars import LogLikelihoodBinaryStars
from general_purpose_python_modules.reproduce_system import find_evolution
from bayesian.windemuth_eccentricity_distribution import W19EccentricityDistribution
from functools import partial
from stellar_evolution.manager import StellarEvolutionManager
from orbital_evolution.transformations import phase_lag
from astropy import units
from types import SimpleNamespace
from multiprocessing import Manager

class twoLines():
    coef = []
    def __init__(self,existingLine,secondExistingLine=None,belowZero=False,breakPoint=0.5,newY=0.1):
        self.breakPoint = breakPoint
        # Any time we want the coefs, it will be from an ehat_approx rather than an ehat_prime,
        # so this is okay.
        self.coef = existingLine.coef
        if secondExistingLine is None:
            breakY = existingLine(breakPoint)
            if belowZero:
                newX = 0.0
                slope = (breakY-newY)/(breakPoint-newX)
                self.lower_line = numpy.polynomial.polynomial.Polynomial((0.0,slope))
                self.upper_line = existingLine
            else:
                newX = 0.8
                slope = (newY-breakY)/(newX-breakPoint)
                yintercept = newY - slope*newX
                self.lower_line = existingLine
                self.upper_line = numpy.polynomial.polynomial.Polynomial((yintercept,slope))
        else:
            self.lower_line = existingLine
            self.upper_line = secondExistingLine
    def __call__(self,x):
        if x < self.breakPoint:
            return self.lower_line(x)
        else:
            return self.upper_line(x)
    def deriv(self):
        # This miiiiight cause problems due to the discontinuity
        return twoLines(self.lower_line.deriv(),self.upper_line.deriv(),breakPoint=self.breakPoint)
    def degree(self):
        return 1
    def inverse(self,e):
        if e < self.breakPoint:
            current_line =  self.lower_line
        else:
            current_line = self.upper_line
        inverse_line = numpy.polynomial.polynomial.Polynomial((-current_line.coef[0]/current_line.coef[1],1/current_line.coef[1]))
        return inverse_line(e)
    # Canonical string representation
    def __repr__(self):
        return 'twoLines(%s,%s,%s,%s,%s)' % (repr(self.lower_line),repr(self.upper_line),repr(self.breakPoint),repr(self.lower_line(0.5)),repr(self.upper_line(0.5)))

class LogLikelihoodWindemuth(LogLikelihoodBinaryStars):
    """The log-likelihood for Windemuth et. al. (2019) EBs."""

    _lock = Manager().Lock()

    def __init__(self,
                 *parent_args,
                 envelope_eccentricity,
                 observed_eccentricity_distro,
                 de_distro,
                 pe_distro,
                 system_eccentricity,
                 system_name = 'default',
                 nn_path = None,
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
        self.system_eccentricity = system_eccentricity
        self.system_name = system_name

        self.upper_limit = 2.0
        self.lower_limit = -2.0

        self.final_eccentricity = 0.0

        self.median_e = 0.0
        self.approximation = 0

        self.a = 0
        self.c = 0
        self.b = 0

        self.ehat_prime = 0

        self.nn_path = nn_path

    def _choose_solver(self, parameters, solver = '1d', final_e = 0):
        """Handle the parameters passed to the log-likelihood function."""
        print('parameters = %s' % (repr(parameters)))
        if solver == '1d':
            parameters['solve'] = True
            parameters['initial_eccentricity'] = 0.8
        elif solver == '2d':
            parameters['initial_eccentricity'] = 'solve'
            parameters['system'].eccentricity = final_e
        else:
            raise ValueError('solver must be either 1d or 2d')

        return parameters
    
    def _likelihood_integrand(self,e,ehat_approx, e_to_be_near):
        ehat_prime = ehat_approx.deriv()
        
        logger = logging.getLogger(__name__)

        if ehat_approx.degree() == 2:
            ehat_approx_copy = ehat_approx.copy()
            ehat_approx_copy.coef[0] -= e
            inverse_ehat_approx = numpy.polynomial.polynomial.Polynomial((ehat_approx_copy.coef)).roots()
            inverse_e = numpy.real(inverse_ehat_approx[numpy.isreal(inverse_ehat_approx)])
            
            for option in inverse_e:
                if option < 0 or option > 0.8:
                    inverse_e = inverse_e[inverse_e != option]
            
            if inverse_e.size == 0:
                #raise ValueError('No real roots found for ehat_approx_copy (inverse_ehat_approx): %s (%s)',repr(ehat_approx_copy),repr(inverse_ehat_approx))
                # this will be an error when we're not doing debug
                #TODO
                #print('No in-range real roots found for ehat_approx_copy (inverse_ehat_approx): %s (%s)' % (repr(ehat_approx_copy),repr(inverse_ehat_approx)))
                logger.warning('No in-range real roots found for ehat_approx_copy (inverse_ehat_approx): %s (%s)',repr(ehat_approx_copy),repr(inverse_ehat_approx))
                inverse_e=numpy.nan
            else:
                inverse_e = inverse_e[numpy.argmin(numpy.abs(inverse_e-e_to_be_near))]
        elif ehat_approx.degree() == 1:
            if isinstance(ehat_approx, twoLines):
                inverse_e = ehat_approx.inverse(e)
            else:
                reverse = numpy.polynomial.polynomial.Polynomial((-ehat_approx.coef[0]/ehat_approx.coef[1],1/ehat_approx.coef[1]))
                inverse_e = reverse(e)
        else:
            raise ValueError('ehat_approx should be a polynomial of order 1 or 2. Something is wrong with ehat_approx: %s',repr(ehat_approx))
        
        if inverse_e > 0.8 or inverse_e < 0:
            logger.error('inverse_e(e) is outside range: %s(%s)',repr(inverse_e),repr(e))
            raise ValueError('inverse_e should not be outside range: %s',repr(inverse_e))
        
        ehat_prime_result = ehat_prime(inverse_e)
        if ehat_prime_result < 0:
            logger.warning('ehat_prime is negative: %s',repr(ehat_prime_result))
        result = self._de.pdf(e) * self._pe.pdf((inverse_e)) / ehat_prime_result
        
        return result

    def _calculate_likelihood(self, ehat_approx, e_to_be_near, e_min, e_max, breakPoint):
        specific_integrand = partial(self._likelihood_integrand,ehat_approx=ehat_approx, e_to_be_near = e_to_be_near)
        I = scipy.integrate.quad(specific_integrand, e_min, e_max, points=breakPoint)[0]
        return I
    
    def _get_ai_carepackage(self,parameters):
        result = dict()
        result['lgQ_min']=self.get_parameter_value(parameters, 'lgQ_min')
        result['lgQ_break_period']=self.get_parameter_value(parameters, 'lgQ_break_period')
        result['lgQ_powerlaw']=self.get_parameter_value(parameters, 'lgQ_powerlaw')
        result['system_name']=self.system_name
        result['lock']=self._lock
        result['path']= self.nn_path #'/home/vortebo/ctime/ayeye'
        return result

    def calculate_log_likelihood(self,
                                 encoded_parameters,
                                 **other_args):
        """Evaluate the log-likelihood at the given model parameters."""

        logger = logging.getLogger(__name__)

        interpolator = StellarEvolutionManager('/home/vortebo/ctime/poet/stellar_evolution_interpolators').get_interpolator_by_name(
            'default'
        )

        parameters = dict()
        parameters['dissipation'] = self.get_dissipation(encoded_parameters)
        parameters['system'] = SimpleNamespace(
            orbital_period=0,
            age=0,
            eccentricity=0,
            primary_mass=0,
            secondary_mass=0,
            feh=0,
            Rprimary = 0,
            Rsecondary = 0
        )

        for big_term in self.parameter_names_units:
            if big_term == 'evolution':
                for i in range(len(self.parameter_names_units[big_term])):
                    if self.parameter_names_units[big_term][i][0] == 'primary_disk_lock_period':
                        parameters['disk_period'] = self.get_parameter_value(encoded_parameters, self.parameter_names_units[big_term][i][0])
                        continue
                    elif self.parameter_names_units[big_term][i][0] == 'secondary_disk_lock_period':
                        continue
                    parameters[self.parameter_names_units[big_term][i][0]] = self.get_parameter_value(encoded_parameters, self.parameter_names_units[big_term][i][0])
            elif big_term == 'system':
                for i in range(len(self.parameter_names_units[big_term])):
                    if self.parameter_names_units[big_term][i][0] == 'orbital_period':
                        parameters['system'].orbital_period = self.get_parameter_value(encoded_parameters, self.parameter_names_units[big_term][i][0])
                    elif self.parameter_names_units[big_term][i][0] == 'age':
                        parameters['system'].age = self.get_parameter_value(encoded_parameters, self.parameter_names_units[big_term][i][0])
                    elif self.parameter_names_units[big_term][i][0] == 'primary_mass':
                        parameters['system'].primary_mass = self.get_parameter_value(encoded_parameters, self.parameter_names_units[big_term][i][0])
                    elif self.parameter_names_units[big_term][i][0] == 'secondary_mass':
                        parameters['system'].secondary_mass = self.get_parameter_value(encoded_parameters, self.parameter_names_units[big_term][i][0])
                    elif self.parameter_names_units[big_term][i][0] == 'feh':
                        parameters['system'].feh = self.get_parameter_value(encoded_parameters, self.parameter_names_units[big_term][i][0])
                    elif self.parameter_names_units[big_term][i][0] == 'cmd_primary_radius':
                        parameters['system'].Rprimary = self.get_parameter_value(encoded_parameters, self.parameter_names_units[big_term][i][0])
                    elif self.parameter_names_units[big_term][i][0] == 'cmd_secondary_radius':
                        parameters['system'].Rsecondary = self.get_parameter_value(encoded_parameters, self.parameter_names_units[big_term][i][0])
                    else:
                        continue
            else:
                continue
        parameters['system'].eccentricity = self.system_eccentricity
        parameters['interpolator'] = interpolator
        #TODO: how should I actually handle these?:
        parameters['secondary_is_star'] = True
        #TODO: this should be specified somewhere like a user command or something, right?
        parameters['precision'] = 1e-5
        logger.debug('Hey look at this do we ever talk about max time steps?') #TODO
        parameters['carepackage'] = self._get_ai_carepackage(encoded_parameters)
        logger.debug('parameters = %s',repr(parameters))

        assert 'sample_weights_envelope' in other_args #why? TODO

        e_to_be_near=0.5
        #Work out parameters such that when we pass it to find_evolution, we get the 1D solver in the way we want
        parameters = self._choose_solver(parameters,'1d')
        try:
            max_final_eccentricity = find_evolution(**parameters)[0].eccentricity[-1]
        except ValueError:
            logger.error('find_evolution returned a ValueError. Returning -numpy.inf.')
            logger.error('Error: %s',repr(ValueError))
            return -numpy.inf
        self.final_eccentricity = max_final_eccentricity
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
            or
            (max_final_eccentricity < De_min)
        ):
                print('Heaviside says no')
                logger.debug('Heaviside says no')
                logger.debug('max_final_eccentricity > self.envelope_eccentricity: %s',repr(max_final_eccentricity > self.envelope_eccentricity))
                logger.debug('max_final_eccentricity is None: %s',repr(max_final_eccentricity is None))
                logger.debug('numpy.isnan(max_final_eccentricity): %s',repr(numpy.isnan(max_final_eccentricity)))
                logger.debug('max_final_eccentricity < De_min: %s',repr(max_final_eccentricity < De_min))
                return -numpy.inf
        
        self.median_e = self._de.ppf(0.5)

        De_away_from_zero = De_min > 0
        De_at_zero = not De_away_from_zero
        De_from_zero_to_max = self._de.cdf(max_final_eccentricity)

        logger.debug('Various values:')
        logger.debug('median_e = %s',repr(self.median_e))
        logger.debug('max_final_eccentricity = %s',repr(max_final_eccentricity))
        logger.debug('De range = %s',repr((De_min,De_max)))
        logger.debug('Envelope eccentricity = %s',repr(self.envelope_eccentricity))
        logger.debug('De_away_from_zero = %s',repr(De_away_from_zero))
        logger.debug('De_at_zero = %s',repr(De_at_zero))
        logger.debug('De_from_zero_to_max = %s',repr(De_from_zero_to_max))
        print('Various values:')
        print('median_e = %s' % (repr(self.median_e)))
        print('max_final_eccentricity = %s' % (repr(max_final_eccentricity)))
        print('De range = %s' % (repr((De_min,De_max))))
        print('Envelope eccentricity = %s' % (repr(self.envelope_eccentricity)))
        print('De_away_from_zero = %s' % (repr(De_away_from_zero)))
        print('De_from_zero_to_max = %s' % (repr(De_from_zero_to_max)))

        breakPoint = None

        if (
            De_at_zero
            and
            (
                max_final_eccentricity <= De_max
            )
            # D_e(e_final=0) is not negligible, and e_hat(e_i=0.8) is in or below D_e.
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
            #(
            #    self._de.cdf(self.envelope_eccentricity)
            #    -
            #    De_from_zero_to_max
            #) <= negligible
            (
                De_max < max_final_eccentricity < self.envelope_eccentricity
            )
        ): # If D_e is clearly away from e_final=0 and e_hat(e_i=0.8) is below the envelope but above D_e
            logger.debug('In a case where we assume ehat is linear in the range where D_e is non-negligible')
            print('In a case where we assume ehat is linear in the range where D_e is non-negligible')
            parameters = self._choose_solver(parameters,'2d',self.median_e)
            parameters['system'].eccentricity = self.median_e
            result = find_evolution(**parameters)
            print('result = %s' % (repr(result)))
            try:
                init_e,ehat_prime = result[1][1],result[1][0][0]
            except TypeError:
                if numpy.isnan(result[1][0]):
                    return -numpy.inf
                else:
                    raise
            except:
                raise
            print('ehat_prime = %s' % (repr(ehat_prime)))
            print('init_e = %s' % (repr(init_e)))
            yintercept = self.median_e - ehat_prime*init_e
            base_approx = numpy.polynomial.polynomial.Polynomial((yintercept,ehat_prime))
            base_approx_inverse = numpy.polynomial.polynomial.Polynomial((-base_approx.coef[0]/base_approx.coef[1],1/base_approx.coef[1]))
            if base_approx_inverse(De_max) > 0.8:
                logger.debug('Found an instance where we must use twoLines: %s',repr(base_approx_inverse(De_max)))
                logger.debug('De_max = %s',repr(De_max))
                logger.debug('max_final_eccentricity = %s',repr(max_final_eccentricity))
                logger.debug('median_e = %s',repr(self.median_e))
                print('This is where we are using twoLines')
                ehat_approx = twoLines(base_approx,None,False,self.median_e,max_final_eccentricity)
                breakPoint = self.median_e
                self.approximation = 1
            elif base_approx_inverse(De_min) < 0.0:
                logger.debug('Found an instance where we must use twoLines: %s',repr(base_approx_inverse(De_min)))
                logger.debug('De_min = %s',repr(De_min))
                logger.debug('max_final_eccentricity = %s',repr(max_final_eccentricity))
                logger.debug('median_e = %s',repr(self.median_e))
                print('This is where we are using twoLines')
                ehat_approx = twoLines(base_approx,None,True,self.median_e,0.0)
                breakPoint = self.median_e
                self.approximation = -1
            else:
                ehat_approx = base_approx
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
            logger.debug('e_to_match = %s',repr(e_to_match))
            logger.debug('e_max_initial = %s',repr(e_max_initial))
            logger.debug('e_max_final = %s',repr(e_max_final))
            print('e_to_match = %s' % (repr(e_to_match)))
            print('e_max_initial = %s' % (repr(e_max_initial)))
            print('e_max_final = %s' % (repr(e_max_final)))
            parameters = self._choose_solver(parameters,'2d',e_to_match)
            parameters['system'].eccentricity = e_to_match
            result = find_evolution(**parameters)
            print('result = %s' % (repr(result)))
            try:
                init_e,ehat_prime = result[1][1],result[1][0][0]
            except TypeError:
                if numpy.isnan(result[1][0]):
                    return -numpy.inf
                else:
                    raise
            except:
                raise
            print('ehat_prime = %s' % (repr(ehat_prime)))
            print('init_e = %s' % (repr(init_e)))
            e_to_be_near=init_e
            A = numpy.matrix([
                                [init_e**2,        init_e,        1],
                                [e_max_initial**2, e_max_initial, 1],
                                [2*init_e,         1,             0]
                            ])
            B = numpy.matrix([e_to_match,e_max_final,ehat_prime])
            fit = scipy.linalg.lstsq(A,B.T)[0]
            ehat_approx = numpy.polynomial.polynomial.Polynomial((fit[2][0],fit[1][0],fit[0][0]))
            print('Location of maximum: %s' % (repr(-ehat_approx.coef[1]/(2*ehat_approx.coef[2]))))
            if e_to_match < -ehat_approx.coef[1]/(2*ehat_approx.coef[2]) < e_max_initial:
                print('We need to revert back to linear.')
                # Making a straight line between the points (e_max_initial,e_max_final) and (init_e,e_to_match)
                ehat_prime = (e_max_final - e_to_match) / (e_max_initial - init_e)
                yintercept = e_to_match - ehat_prime*init_e
                ehat_approx = numpy.polynomial.polynomial.Polynomial((yintercept,ehat_prime))
                print('(%s,%s) and (%s,%s)' % (repr(e_max_initial),repr(e_max_final),repr(init_e),repr(e_to_match)))
        else:
            logger.error('Something went wrong in the Windemuth likelihood function. Please check the code.')
            raise ValueError('Something went wrong in the Windemuth likelihood function. Please check the code.')
        
        ehat_prime = ehat_approx.deriv() #TODO (after debugging): remove, already being handled by _calculate_likelihood

        logger.debug('ehat_approx = %s',repr(ehat_approx))
        logger.debug('ehat_prime = %s',repr(ehat_prime))
        print('ehat_approx = %s' % (repr(ehat_approx)))
        print('ehat_prime = %s' % (repr(ehat_prime)))
        
        integration_min = max(0,De_min)
        integration_max = min(De_max,max_final_eccentricity)
        #try:
        likelihood = self._calculate_likelihood(ehat_approx,e_to_be_near,integration_min,integration_max,breakPoint)
        #except ValueError:
        #    logger.error('Error in _calculate_likelihood. Returning -numpy.inf.')
        #    logger.error('Error: %s',repr(ValueError))
        #    return -numpy.inf
        
        logger.debug('likelihood = %s',repr(likelihood))
        logger.debug('log(likelihood) = %s',repr(numpy.log(likelihood)))
        print('likelihood = %s' % (repr(likelihood)))
        print('log(likelihood) = %s' % (repr(numpy.log(likelihood))))

        assert likelihood >= 0
        if likelihood == 0:
            return -numpy.inf

        if ehat_approx.degree() == 2:
            self.a = ehat_approx.coef[2]
        self.c = ehat_approx.coef[0]
        self.b = ehat_approx.coef[1]
        self.ehat_prime = ehat_prime.coef[0]

        return numpy.log(likelihood)
    
    def test_internal_functions(self):
        logger = logging.getLogger(__name__)

        # Test_likelihood_integrand
        logger.debug('Testing _likelihood_integrand')
        print('Testing _likelihood_integrand')
        e_set = numpy.linspace(0,0.8,20)
        ehat_set = (
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
        de_distro=W19EccentricityDistribution(3348093,pickle_fname='/home/vortebo/ctime/CircularizationDissipationConstraints/windemuth_eccentricity_distros.pkl'),
        pe_distro=scipy.stats.uniform(loc=0,scale=0.8),
        powerlaw_dissipation = 5,
        interpolator=interpolator,
        evolution_timeout = 1.0,
        period_search_factor = 2.0,
        scaled_period_guess = 1.0
    )
    #bob.test_internal_functions()
    bob.test_external_functions(interpolator)