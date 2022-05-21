Single Lined Spectroscopic Binaries in Open Clusters
====================================================

Required Environment Variables:
-------------------------------

 * ``PYTHONPATH`` must include:

   * the path to the POET python package

   * the path to general_purpose_python_modules

   * the path to the source directory of this repository

 * ``LIBRARY_PATH`` and ``LD_LIBRARY_PATH`` must both include the path
   containing the POET shared libraries, or they must be added to the system
   libraries through ``ldconfig``

Generating Posterior Samples
----------------------------

Samples from the joint posterior distribution of binary properties and
dissipation parameters is accomplished by::

    python3 <this_repo>/source/bayesian/sample_sb1.py\
        <system_name>\
        -c <config_file>\
        <command line options>

System names are formatted as ``<CLUSTER>_<SYSTEM ID>``. For example,
``NGC188_4618`` refers to the binary with PKM identifier 4618 in NGC 188.
Different clusters use different identifiers (e.g. NGC 188 uses PKM, M 35 and
NGC 6819 use WOCS etc.).

If sampling is interrupted for whatever reason, you can run the same command
again to continue from the last saved emcee step.

At the moment the required eccentricity envelope, cluster properties, and binay
data tables are available for the following clusters:

  * M 35 (150 Myr)

  * NGC 6819 (2.5 Gyr)

  * NGC 188 (7 Gyr)
