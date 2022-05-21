.. Constraining Tidal Dissipation from Observed Circularization documentation master file, created by
   sphinx-quickstart on Wed May 18 10:45:41 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

System by System Tidal Dissipation Constraints from Eccentricity
================================================================

The goal of this project is to measure the tidal dissipation in stars and
planets in binary sistems for which eccentricity data is available. The meathod
for doing so is as follows:

    * a lower limit on the dissipation can be derived by requiring that the 
      measured eccentricity survives to the present age of the system
      
    * an upper limit on the dissipation can be derived by noting that under the
      assumption that initial eccentricity distribution is similar to that at
      longer orbital periods, an identical system had a probability of starting
      with large initial eccentricity, and the fact that we do not observe
      eccentricities above the e(P) envelope means that these systems must be
      circularized to at least below that envelope.
      
This can be applied system by system to exoplanet systems and binary stars and
can cover a broader range of parameters.

We use Bayesian analysis to generate samples for parameters of a tidal
dissipation model for each system that as much as possible accounts for
observational and model uncertainties.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   sampling_binaries
   API <_implementation/modules>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
