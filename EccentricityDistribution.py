from abc import ABCMeta, abstractmethod
from scipy.special import erf
import math
from scipy.stats import rice
import logging
from scipy.optimize import fsolve
import numpy as np
from scipy.special import i0
from scipy.integrate import nquad
import matplotlib.pyplot as plt
import argparse

def phi(z):
    return 0.5 * (1 + erf(z / math.sqrt(2)))

def alpha(ci):
    return 1 - ci / 100.0

class SuperEccentricityDistribution(metaclass=ABCMeta):
    @abstractmethod
    def probability_density_of_eccentricity(self, e):
        pass

class EccentricityDistribution(SuperEccentricityDistribution):

    def __init__(self,
                 args,
                 percentile_for_e_now_upper_uncertainty=phi(1),  # or sometimes 1 - alpha(68.0)/2
                 percentile_for_e_now_lower_uncertainty=1 - phi(1),  # or sometimes alpha(68.0)/2
                 ):

        self.measured_e_now = args.eccentricity
        self.e_now_upper_uncertainty = args.eccentricity_upper_uncertainty
        self.e_now_lower_uncertainty = args.eccentricity_lower_uncertainty
        if args.system:
            self.system_name = args.system
        else:
            self.system_name = 'Star-Exoplanet system'
        self.e_env = args.envelope_eccentricity

        self.percentile_for_e_now_upper_uncertainty = percentile_for_e_now_upper_uncertainty
        self.percentile_for_e_now_lower_uncertainty = percentile_for_e_now_lower_uncertainty


        self.rice_parameters_are_found = True

        self.b, self.s = self.roots_for_Rice_parameters()
        self.inv_norm = self.cdf(1.0)

    def equations_to_be_solved_for_Rice_distribution_parameters(self, x):
        b = x[0]
        s = x[1]
        first = rice.cdf((self.measured_e_now + self.e_now_upper_uncertainty), b,
                         scale=s) - self.percentile_for_e_now_upper_uncertainty
        second = rice.cdf((self.measured_e_now + self.e_now_lower_uncertainty), b,
                          scale=s) - self.percentile_for_e_now_lower_uncertainty
        if math.isnan(first) or math.isnan(second):
            logging.warning('Iteration does not converge')
            self.rice_parameters_are_found = False
        return [first, second]

    def equation_to_be_solved_for_Rice_distribution_parameter_s_when_b_zero(self, x):
        s = x[0]
        eqn = rice.cdf((self.measured_e_now + self.e_now_upper_uncertainty), 0,
                       scale=s) - self.percentile_for_e_now_upper_uncertainty
        if math.isnan(eqn):
            logging.warning('Iteration does not converge')
            self.rice_parameters_are_found = False
        return [eqn]

    def roots_for_Rice_parameters(self):
        estimated_s = self.e_now_upper_uncertainty
        if self.measured_e_now == 0:
            try:
                s = fsolve(self.equation_to_be_solved_for_Rice_distribution_parameter_s_when_b_zero,
                           np.asarray([estimated_s]))
            except:
                logging.warning('Rice parameters cannot be worked out')
                self.rice_parameters_are_found = False
                return [math.nan, math.nan]
            else:
                self.rice_parameters_are_found = True
                return [0, s[0]]
        estimated_b = self.measured_e_now / (self.e_now_upper_uncertainty)
        roots = [math.nan, math.nan]
        try:
            roots = fsolve(self.equations_to_be_solved_for_Rice_distribution_parameters,
                           np.asarray([estimated_b, estimated_s]))
        except:
            logging.warning('Rice parameters cannot be worked out')
            self.rice_parameters_are_found = False
        else:
            self.rice_parameters_are_found = True
        return roots

    def pdf(self, e):
        val = i0((e / self.s) * self.b)
        if val == math.inf:
            return 0  # Then math.exp(-((e/s)**2+b**2)/2) = 0
        return math.exp(-((e / self.s) ** 2 + self.b ** 2) / 2) * val

    def cdf(self, e_now):
        value = nquad(self.pdf, [[0, e_now]])
        return value[0]

    def cumulative_density_function_of_present_eccentricity(self, e_now):
        if not (math.isnan(self.s) or math.isnan(self.b)):
            if self.inv_norm == 0:
                return math.inf
            value = self.cdf(e_now) / self.inv_norm
            return value
        logging.warning(
            'Cumulative density function of present eccentricity does not exist for the given e_now and its uncertainties')
        return math.nan

    def probability_density_of_eccentricity(self, e):
        if e > 1 or e < 0:
            return 0
        if e <= self.e_env:
            return self.cumulative_density_function_of_present_eccentricity(e)
        return 0

    def plot_probability_density_of_eccentricity_vs_eccentricity_graph(self):
        eccentricity = np.linspace(0, 1, 100)
        probability_density_of_eccentricity = []
        for i in range(0, len(eccentricity)):
            probability_density_of_eccentricity = probability_density_of_eccentricity + [
                self.probability_density_of_eccentricity(eccentricity[i])]
        M_cdf = []
        for i in range(0, len(eccentricity)):
            M_cdf = M_cdf + [self.cumulative_density_function_of_present_eccentricity(eccentricity[i])]
        M_pdf = []
        for i in range(0, len(eccentricity)):
            M_pdf = M_pdf + [self.pdf(eccentricity[i])]
        plt.plot(eccentricity, probability_density_of_eccentricity,
                 label="Probality density of eccentricity (f(e)) vs. eccentricity (e)")
        # naming the x axis
        plt.xlabel('Eccentricity (e)')
        # naming the y axis
        plt.ylabel('probability density of eccentricity (f(e))')
        # giving a title to my graph
        plt.title('Probability density of eccentricity vs. eccentricity for %(system)s.pdf' % dict(system=self.system_name))

        plt.savefig('Probability density of eccentricity vs. eccentricity for %(system)s.pdf' % dict(system=self.system_name))
        # function to show the plot
        plt.show()
        plt.plot(eccentricity, M_cdf, label="cdf of M(e) vs. eccentricity (e)")
        # naming the x axis
        plt.xlabel('Eccentricity (e)')
        # naming the y axis
        plt.ylabel('M_cdf ')
        # giving a title to my graph
        plt.title('CDF of M(e) vs. eccentricity, e for %(system)s.pdf' % dict(system=self.system_name))
        # function to show the plot
        plt.savefig('CDF of M(e) vs. eccentricity, e for %(system)s.pdf' % dict(system=self.system_name))
        plt.show()

        plt.plot(eccentricity, M_pdf, label="pdf of M(e) vs. eccentricity (e)")
        # naming the x axis
        plt.xlabel('Eccentricity (e)')
        # naming the y axis
        plt.ylabel('M_pdf ')
        # giving a title to my graph
        plt.title('PDF of M(e) vs. eccentricity, e for %(system)s.pdf' % dict(system=self.system_name))

        plt.savefig('PDF of M(e) vs. eccentricity, e for %(system)s.pdf' % dict(system=self.system_name))
        # function to show the plot
        plt.show()
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
    parser.add_argument('envelope_eccentricity',
                        help='Store the envelope eccentricity',
                        type=float)
    parser.add_argument('--system', help='Store the name of the star-exoplanet system')
    args = parser.parse_args()
    e_dist = EccentricityDistribution(args=args)
    e_dist.plot_probability_density_of_eccentricity_vs_eccentricity_graph()