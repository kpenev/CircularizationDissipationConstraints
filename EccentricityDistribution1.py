from abc import ABCMeta, abstractmethod
from scipy.special import erf
import math
from scipy.stats import rice
import logging
from scipy.optimize import fsolve
import numpy as np
from scipy.special import i0
from scipy import integrate
import matplotlib.pyplot as plt
import argparse
from scipy.stats import norm
import os
import sys
from rice_distribution_utils import rice_from_error_bars
from eccentricity_kde_distro_gen import eccentricity_kde_distro_gen
sys.path.append('/home1/08529/mmmahmud/general_purpose_python_modules')
from rice_distribution_utils import rice_from_error_bars

def phi(z):
    return 0.5 * (1 + erf(z / math.sqrt(2)))


class EccentricityDistribution:
    def __init__(self,
                 measured_eccentricity,
                 eccentricity_upper_uncertainty,
                 eccentricity_lower_uncertainty,
                 eccentricity_flag_limit = 0,
                 system_name = 'Star-Exoplanet',
                 output_directory = "/work/08529/mmmahmud/p0andmcmc",
                 crit=35, logger = None):

        self.measured_e_now = measured_eccentricity
        self.e_now_upper_uncertainty = eccentricity_upper_uncertainty
        self.e_now_lower_uncertainty = eccentricity_lower_uncertainty
        if eccentricity_flag_limit == 1:
            self.confidence_interval_for_eccentricity = phi(1)
        if eccentricity_flag_limit == 2:
            self.confidence_interval_for_eccentricity = 0.95
        if eccentricity_flag_limit == 3:
            self.confidence_interval_for_eccentricity = 0.99
        if eccentricity_flag_limit == 0:
            self.confidence_interval_for_eccentricity = 2*phi(1) - 1

        self.system_name = system_name
        self.logger = logger
        self.eccentricity_distribution_is_delta = False

        self.output_directory = "%(output_directory)s/%(system)s/%(system)s_edist" % dict(output_directory=output_directory, system=system_name)
        if os.path.exists(self.output_directory):
            if self.logger is not None: self.logger.debug("output directory exists: %(directory)s " % dict(directory= self.output_directory))
        if not os.path.exists(self.output_directory):
            if self.logger is not None: self.logger.debug('output directory does not exist. we are going to create it.')
            os.makedirs(self.output_directory)
            if os.path.exists(self.output_directory):
                 if self.logger is not None: self.logger.debug("Now it is created: %(directory)s " % dict(directory=self.output_directory))
        self.eccentricity_distribution = None
        if (measured_eccentricity > 0) and (eccentricity_upper_uncertainty != 0) and (eccentricity_lower_uncertainty != 0) and eccentricity_flag_limit == 0:
            if measured_eccentricity/eccentricity_upper_uncertainty < crit or measured_eccentricity/math.fabs(eccentricity_lower_uncertainty) < crit:
                 if self.logger is not None: self.logger.debug("Eccentricity distribution for this system is a Rice Distribution/2pi.eccentricity")
                 self.rice_distribution = rice_from_error_bars(self.measured_e_now, self.e_now_upper_uncertainty, math.fabs(self.e_now_lower_uncertainty))
                 self.raw_moment1 = self.rice_distribution.moment(1)
                 self.raw_moment2 = self.rice_distribution.moment(2)
                 self.raw_moment3 = self.rice_distribution.moment(3)
                 self.raw_moment4 = self.rice_distribution.moment(4)
                 self.nu_square = math.sqrt(2 * (self.raw_moment2 ** 2) - self.raw_moment4)
                 self.sigma_square = (self.raw_moment2 - self.nu_square)/2
                 self.nu = math.sqrt(self.nu_square)
                 self.eccentricity_distribution = self.get_eccentricity_distribution()

                 rice_params = rice_from_error_bars(self.measured_e_now, self.e_now_upper_uncertainty, math.fabs(self.e_now_lower_uncertainty)).kwds
                 self.nu_ = rice_params['b']*rice_params['scale']
                 self.sigma_square_ = rice_params['scale']**2
                 self.eccentricity_distribution_ = eccentricity_kde_distro_gen([rice_params['b'] * rice_params['scale']], rice_params['scale']).pdf
            else:
                 #self.rice_distribution = rice_from_error_bars(self.measured_e_now, self.e_now_upper_uncertainty, math.fabs(self.e_now_lower_uncertainty))
                 #self.raw_moment1 = self.rice_distribution.moment(1)
                 #self.raw_moment2 = self.rice_distribution.moment(2)
                 #self.raw_moment3 = self.rice_distribution.moment(3)
                 #self.raw_moment4 = self.rice_distribution.moment(4)
                 #self.nu_square = math.sqrt(2 * (self.raw_moment2 ** 2) - self.raw_moment4)
                 #self.sigma_square = (self.raw_moment2 - self.nu_square)/2
                 #self.nu = math.sqrt(self.nu_square)
                 rice_params = rice_from_error_bars(self.measured_e_now, self.e_now_upper_uncertainty, math.fabs(self.e_now_lower_uncertainty)).kwds
                 self.nu = rice_params['b']*rice_params['scale']
                 self.sigma_square = rice_params['scale']**2
                 self.nu_square = self.nu * self.nu
                 self.eccentricity_distribution = self.get_eccentricity_distribution()

                 if self.logger is not None: self.logger.debug("Eccentricity distribution for this system is approximately a Normal distribution.")
                 rice_params = rice_from_error_bars(self.measured_e_now, self.e_now_upper_uncertainty, math.fabs(self.e_now_lower_uncertainty)).kwds
                 self.nu_ = rice_params['b']*rice_params['scale']
                 self.sigma_square_ = rice_params['scale']**2
                 self.eccentricity_distribution_ = eccentricity_kde_distro_gen([rice_params['b'] * rice_params['scale']], rice_params['scale']).pdf

        if (measured_eccentricity == 0) and (eccentricity_upper_uncertainty != 0) and (eccentricity_lower_uncertainty == 0):
            if self.logger is not None: self.logger.debug("The eccentricity upper uncertainty is non zero and eccentricity lower uncertainty is zero.")
            b, s = self.roots_for_Rice_parameters()
            if self.logger is not None: self.logger.debug("b is %(b)f and s is %(s)f" % dict(b=b, s=s))
            self.sigma_square = s*s
            self.nu = b*s
            self.nu_square = self.nu * self.nu
            self.eccentricity_distribution = self.get_eccentricity_distribution()

            rice_params = rice_from_error_bars(self.measured_e_now, self.e_now_upper_uncertainty, math.fabs(self.e_now_lower_uncertainty)).kwds
            self.nu_ = rice_params['b']*rice_params['scale']
            self.sigma_square_ = rice_params['scale']**2
            self.eccentricity_distribution_ = eccentricity_kde_distro_gen([rice_params['b'] * rice_params['scale']], rice_params['scale']).pdf

    def get_eccentricity_distribution(self):
        def eccentricity_distribution(e):
            if math.fabs(self.nu)< 0.000001:
                return (1/2/math.pi/self.sigma_square)*math.exp(-e**2/2/self.sigma_square)
            return (1/2/math.pi/self.sigma_square)*math.exp(-(e**2 + self.nu_square)/2/self.sigma_square) #* i0(e * self.nu/self.sigma_square)
        return eccentricity_distribution

    def equations_to_be_solved_for_Rice_distribution_parameters(self, x):
        b = x[0]
        s = x[1]
        first = rice.cdf((self.measured_e_now + self.e_now_upper_uncertainty), b,
                         scale=s) - phi(1)
        second = rice.cdf((self.measured_e_now + self.e_now_lower_uncertainty), b,
                          scale=s) - (1-phi(1))
        if math.isnan(first) or math.isnan(second):
            if self.logger is not None: self.logger.error('Iteration for working out Rice parameters does not converge')
            self.rice_parameters_are_found = False
        return [first, second]

    def equation_to_be_solved_for_Rice_distribution_parameter_s_when_b_zero(self, x):
        s = x[0]
        eqn = rice.cdf((self.measured_e_now + self.e_now_upper_uncertainty), 0,
                       scale=s) - self.confidence_interval_for_eccentricity
        if math.isnan(eqn):
            if self.logger is not None: self.logger.error('Iteration for working out Rice parameters does not converge')
            self.rice_parameters_are_found = False
        return [eqn]
    def roots_for_Rice_parameters(self):
        estimated_s = self.e_now_upper_uncertainty
        if self.logger is not None: self.logger.debug("estimated s = %(es)f" % dict(es=estimated_s))
        if self.measured_e_now == 0:
            a = rice.ppf(self.confidence_interval_for_eccentricity, 0, loc=0, scale=1)
            estimated_s = self.e_now_upper_uncertainty/a
            test = rice.ppf(self.confidence_interval_for_eccentricity, 0, loc=0, scale=estimated_s)
            if test == self.e_now_upper_uncertainty:
                return [0, estimated_s]
            if self.logger is not None: self.logger.debug("estimated s = %(es)f" % dict(es=estimated_s))
            try:
                if self.logger is not None: self.logger.info("measured e_now is zero")
                s = fsolve(self.equation_to_be_solved_for_Rice_distribution_parameter_s_when_b_zero,
                           np.asarray([estimated_s]))
            except:
                if self.logger is not None: self.logger.error('Rice parameters cannot be worked out')
                self.rice_parameters_are_found = False
                return [math.nan, math.nan]
            else:
                self.rice_parameters_are_found = True
                if self.logger is not None: self.logger.debug("Rice parameters are found.")
                return [0, s[0]]
        if self.e_now_upper_uncertainty !=0:
            estimated_b = self.measured_e_now / (self.e_now_upper_uncertainty)
            if self.logger is not None: self.logger.debug("Estimated b is %(eb)f" % dict(eb=estimated_b))
            roots = [math.nan, math.nan]
            try:
                roots = fsolve(self.equations_to_be_solved_for_Rice_distribution_parameters,
                               np.asarray([estimated_b, estimated_s]))
            except:
                if self.logger is not None: self.logger.error('Rice parameters cannot be worked out')
                self.rice_parameters_are_found = False
            else:
                if self.logger is not None: self.logger.debug("Rice parameters are found.")
                self.rice_parameters_are_found = True
            return roots
        return [math.nan, math.nan]

    def plot_probability_density_of_eccentricity_vs_eccentricity_graph(self):
        eccentricity = np.linspace(0, 1, 100)
        probability_density_of_eccentricity = []
        probability_density_of_eccentricity_ = []
        probability_density_of_eccentricity__ = []
        w = self.eccentricity_distribution(0.1)/self.eccentricity_distribution_(0.1)
        for i in range(0, len(eccentricity)):
            probability_density_of_eccentricity = probability_density_of_eccentricity + [
                self.eccentricity_distribution(eccentricity[i])]
            probability_density_of_eccentricity_ = probability_density_of_eccentricity_ + [
                self.eccentricity_distribution_(eccentricity[i])]
            probability_density_of_eccentricity__ = probability_density_of_eccentricity__ + [
                w * self.eccentricity_distribution_(eccentricity[i])]

        #w = self.eccentricity_distribution(0.1)/self.eccentricity_distribution_(0.1)
        if self.logger is not None: self.logger.info("We are going to save the probability distribution of eccentricity ")
        plt.plot(eccentricity, probability_density_of_eccentricity, label="Probality density of eccentricity (f(e)) vs. eccentricity (e)")
        plt.plot(eccentricity, probability_density_of_eccentricity_)
        plt.plot(eccentricity, probability_density_of_eccentricity__)
        plt.xlabel('Eccentricity (e)')
        plt.ylabel('probability density of eccentricity (f(e)), w=%(w)f' % dict(w=w))
        #plt.title('Probability density of eccentricity vs. eccentricity for %(system)s' % dict(system=self.system_name))
        plt.title('sigmasq1=%(x)f,sigmasq2=%(y)f,nu1=%(z)f,nu2=%(t)f' % dict(x=self.sigma_square,y=self.sigma_square_,z=self.nu,t=self.nu_))
        fname = '%(output_directory)s/Probability density of eccentricity vs. eccentricity for %(system)s.pdf' % dict(output_directory = self.output_directory, system=self.system_name)
        plt.savefig(fname)
        plt.clf()
        return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('eccentricity',
                        help='Store the present measured mean or mod eccentricity of the orbit',
                        type=float)
    parser.add_argument('eccentricity_upper_uncertainty',
                        help='Store the upper uncertainty of the measured present eccentricity',
                        type=float)
    parser.add_argument('eccentricity_lower_uncertainty',
                        help='Store the lower uncertainty of the measured present eccentricity',
                        type=float)
    parser.add_argument('--eccentricity_flag_limit',
                        help='Store the eccentricity flag limit',
                        type=float)
    parser.add_argument('--system', help='Store the name of the star-exoplanet system')
    parser.add_argument('--output_directory', help='Store the path of the output directory')
    args = parser.parse_args()
    e_dist = EccentricityDistribution(measured_eccentricity=args.eccentricity,
                                      eccentricity_upper_uncertainty=args.eccentricity_upper_uncertainty,
                                      eccentricity_lower_uncertainty=args.eccentricity_lower_uncertainty,
                                      eccentricity_flag_limit=args.eccentricity_flag_limit if args.eccentricity_flag_limit else 0.0,
                                      system_name=args.system if args.system else "Star-Exoplanet",
                                      output_directory=args.output_directory if args.output_directory else "/work/08529/mmmahmud/p0andmcmc")
    e_dist.plot_probability_density_of_eccentricity_vs_eccentricity_graph()

