==========================================================================
RockyVsGaseousFromTides, UTDallas Summer Physics REU 2019 -- Anna Tetzlaff
==========================================================================

Evolutions using the test_evolution.py script
=============================================
Dr. Kalo and I modified POET’s ``test_evolution.py`` script so that planets can be set with dissipation as well as stars. Using this script I ran test evolutions for dissipative planets and non-dissipative stars and examined the graphs produced by the script to get a better understanding of how a system evolves over time. I also adjusted the spin angular momentum of the planet (``spin_angmom`` under ``planet.configure`` in the ``create_system`` function), by altering the values in this array I could see how a planet synchronizes over time. When I set the value for the spin angular momentum of the planet, I calculated it using the moment of inertia and multiplying it by the angular velocity. The momentum of inertia for our planet is already defined in the script as ``planet_inertia``; when running test evolutions I used the angular velocity of earth which is 6.3 rad/day.

Grid of orbital period and eccentricity evolution plots
=======================================================
After examining some test evolutions, we wanted to create a grid of plots that show the evolution of period and eccentricity over time for two planets, one with a density of 5g/cm^3 and the other with a density of 2g/cm^3. To do this, I first created a txt file where each line contained the density (in g/cm3) and the mass and radius (in solar units) for two planets. Then I created two Python scripts: ``eccentricity_evolution.py`` and ``period_evolution.py``. In the case of the ``eccentricity_evolution.py`` script, the script reads in the data for a planet from the txt file, then, depending on the density, it sets the tidal quality factor to either Q=6.0 (for lower density) or Q=3.0 (for higher density). The script then runs the ``test_evolution`` function with the mass and radius read in from the txt file. In the ``test_evolution`` function of the script, two arrays have been created, one with initial orbital periods and one with initial eccentricities. Using nested ``for`` loops, the script creates plots showing how the initial eccentricities in the array evolve over time for a set initial orbital period. Alternatively, the ``period_evolution.py`` script does the inverse, where it creates plots showing how different orbital periods evolve for a set initial eccentricity. All the graphs are available under `diss_eccentricity_evolutions`_ and `diss_porb_evolutions`_ in the repository. The only plots not available are the orbital period plots for the 2g/cm^3 planet at initial eccentricities 0.4 and 0.5. When trying to run the ``period_evolution.py`` script at initial eccentricity 0.4 the process appears to become unresponsive, while for initial eccentricity 0.5 it returns a Segmentation fault (core dumped) error. I believe this is because I was using the eccentricity_expansion_coef.txt file to pass to ``orbital_evolution_library.read_eccentricity_expansion_coefficients``. In order to correct the Segmentation fault error, the file eccentricity_expansion_coef_O200.txt should be used.

.. _diss_eccentricity_evolutions: https://github.com/kpenev/RockyVsGaseousFromTides/tree/master/diss_eccentricity_evolutions
.. _diss_porb_evolutions: https://github.com/kpenev/RockyVsGaseousFromTides/tree/master/diss_porb_evolutions

Altering the general purpose modules
====================================
So now that I had an understanding of how a planet’s orbital period and eccentricity evolve over time, we wanted to start modelling real systems. To do this I needed to use the Python modules on the GitHub repository `general_purpose_python_modules`_. The first module needed was ``planetary_system_io.py`` which has a function called ``read_nasa_planets``. Using this function, I could read in data from a CSV file downloaded from the NASA Exoplanet Archive by passing the name of the CSV file to the ``read_nasa_planets`` function. The function will return a dictionary with the column names from the input file as keys. If you don’t know what the names of the columns from the CSV file are, the ``read_nasa_planets`` function will print the names of the columns when it runs; the CSV file should also list what the column names are.

.. _general_purpose_python_modules: https://github.com/kpenev/general_purpose_python_modules

The ``planetary_system_io.py`` script can already be used as is and requires no alterations. The other script, ``reproduce_system.py``, did need some changes to be made before I could use it. (My first altered version of the script, which I simply titled `reproduce_system_nasa.py`_ is on the repository) Originally, ``reproduce_system.py`` was written to use a HATSouth info file. All the instances where the script tries to access data from the system dictionary needed to be changed so that it accesses data using keys from the NASA Exoplanet Archive CSV file. For instance, the original ``reproduce_system.py`` script tries to access the mass of the planet using: ``system.Mseconday``, but the Archive CSV file uses the column name ``pl_massj`` . Therefore, I needed to change ``system.Mseconday`` to ``system.pl_massj[i]`` where ``[i]`` is the index number of the system dictionary to be accessed. The ``[i]`` is necessary otherwise it will access the whole array of planet masses instead of one planet mass. Additionally, the original ``reproduce_system.py`` script set the units for different variables using ``to.(units.Gyr).value`` and ``units.days``, these are unnecessary for the NASA Exoplanet Archive CSV file so I simply commented them out. Another change that needed to be made, was the ``disk_period`` argument in the function ``find_evolution`` needed to be set. If the argument is not set, the function tries to calculate the disk period using the rotational velocity and radius of the star. While these parameters can be included in a search on the NASA Exoplanet Archive, there are not many data sets that have values for these parameters. Instead, the disk period should just be set, I set mine to 2.5.

.. _reproduce_system_nasa.py: https://github.com/kpenev/RockyVsGaseousFromTides/blob/master/reproduce_system_nasa.py

Find data on the NASA Exoplanet Archive and finding stellar parameters
======================================================================
The ``reproduce_system.py`` script and my altered versions of it are the scripts that require data from the NASA Exoplanet Archive. First, there are a few parameters that need limits to be set as stated in `README.rst`_. For my experiment, I only used Transit planets and I constricted the semimajor axis to <0.12 AU and the radius of the planet to <4.0 Earth radii (Dr. Kalo and I agreed to expand the radius parameter to 4.0 rather than 3.0). After constricting those parameters on the Archive, there are four parameters that I set:

- Planet mass to ‘not null’
- Metallicity to ‘not null’ (both [M/H] and [Fe/H] ratios can be used)
- Orbital period to ‘not null’
- Orbital eccentricity to >0.0

Planet density is not necessary for ``reproduce_system.py`` and its variants to run, however I did use it to set the lgQ for high density or low density planets. With the parameters I already set, most of the planets found did have a density. If a data set did not provide a density, I simply solved for it within the main body of the script using the provided mass and radius of the planet. To run evolutions, ``reproduce_system.py`` also needs a few other stellar parameters:

- Star mass set to be within the range [0.4, 1.2] solar masses
- Star age

However, while most of the data sets I found on the Archive did include a stellar mass, many did not have a stellar age listed and requiring stellar age would greatly reduce the sample size. In order to find the stellar age I needed to use the ``change_variables.py`` script included with POET. I wrote a script `star_params.py`_ that just has a function called ``find_star_params`` that accepts the dictionary created by the ``read_nasa_planets`` function, the interpolator, the stellar parameter to for ``change_variables.py`` to use, and the index number for the system you are finding the stellar parameters for. When using ``change_variables.py``, it might return nothing, one solution, or multiple solutions. If nothing was returned then I had the script skip that data set; if one solution was returned the script checks if the age of the star is feasible and if so the evolution is run. If multiple solutions were returned, initially, the script would ask the user to input the solution to use; however, in my experience the first solution provided was always the best one. In order to not have to constantly babysit my script as it ran, I simply had my script always use the first solution in the case of multiple solutions. All this is for cases in which there is no mass and age provided on the archive, my scripts are written so that if there is a stellar mass and age provided the evolution will use that, but if there is no mass and age provided then it will use ``change_variables``.

In the end, the systems I found on the Archive all included stellar mass and age. The data file I used to run ``find_lgq.py`` and make my log(Q) graphs is `system_data.csv`_ and can be found on the repository.

.. _README.rst: https://github.com/kpenev/RockyVsGaseousFromTides/blob/master/README.rst
.. _star_params.py: https://github.com/kpenev/RockyVsGaseousFromTides/blob/master/star_params.py
.. _system_data.csv: https://github.com/kpenev/RockyVsGaseousFromTides/blob/master/system_data.csv

Using reproduce_system_nasa.py and find_lgq.py
==============================================
Essentially, ``reproduce_system_nasa.py`` works exactly the same as the original ``reproduce_system.py`` but it now reads a NASA Exoplanet Archive CSV file. When running ``reproduce_system_nasa.py``, I was hoping to solve for the initial eccentricity of the system given the current state of the system. However, in running the script it appears to just set the current day eccentricity as the initial eccentricity. Unfortunately, I did not have enough time to get the script working as intended, however I did upload it to the repository in case it could be of some use.
| This script that I was able to finish and seems to work correctly is `find_lgq.py`_. ``find_lgq.py`` is an altered version of ``reproduce_system.py``, that uses class inheritance to use the functions of the pre-existing ``EccentricitySolverCallable`` class for the new class I wrote ``lgQSolverCallable``. ``lgQSolverCallable`` has two functions of its own: ``__call__`` and ``__init__``. I did, however, have to make some changes to the functions in ``EccentricitySolverCallable``, like changing the ``create_system_components`` and ``create_planet`` to take lgQ as an argument. ``find_lgq.py`` solves for the lower limit that could be placed on the log of the tidal quality factor (log(Q)) using the scipy optimize function ``brentq``. ``find_lgq.py`` gives all systems an initial eccentricity of 0.5. Then, starting with the lower range of log(Q) argument passed to ``brentq``, it solves for an initial orbital period to reproduce the current one. Once it finds an initial orbital period to match the current, it compares the evolution's calculated final eccentricity with the actual current day eccentricity. If the calculated and actual eccentricities are equal, then the log(Q) used for that evolution will be returned by the ``find_lgq``. If the calculated and actual eccentricities are not the same, then the evolution will be started over with a different value for log(Q) and continued until equal eccentricities are found. One problem I ran into was evolutions that would not find the correct final age. In the ``initial_conditions_solver.py`` script, on line 99 it has the code: ``assert final_state.age == self.target.age``. However, if the found final age and actual final age are not equal, the script will throw an error and stop running. In order to avoid this, I added the following code before the ``assert`` statement:

.. code-block:: python
    
    if final_state.age != self.target.age:
        print("Target age: " + self.target.age)
        print("Found age: " + final_state.age)
        raise Exception

If the found and actual ages are not the same then ``initial_conditions_solver.py`` raises an ``Exception``. The ``Exception`` is then caught by the ``try/except`` statements in the ``__call__`` function of ``lgQSolverCallable``. When the ``Exception`` is caught, another is raised in the ``__call__`` function which is then caught in the ``find_lgq`` function by another ``try/except`` statement. If an ``Exception`` is thrown in the ``find_lgq`` function, the entire evolution will be started again with a different disk dissipation age which seems to avoid the problem most of the time. This solution is not the most elegant, but as I was running low on time is was a sufficient enough quick fix for what I needed. Finally, once all the log(Q) are found a graph of log(Q) vs. Planet Mass and log(Q) vs. Planet Density are created. Currently those are the only two graph to be created by the script, but afterwards I also made graphs for orbital period, eccentricity, planet radius, and semi-major axis/planet radius. However, I made these other graphs after the fact by just taking the found log(Q) and quick writing up another script, ``lgq_graph.py``, that reads in the system data and has a list of the log(Q) and creates the plots I need, that way I don't need to run ``find_lgq.py`` again. ``lgq_graph.py`` is unfortunately a very messy script, but I had to throw it together rather quickly to get all the final graphs done. It is available in the `logQ_graphs`_ folder on the repository. All of the graphs created using ``find_lgq.py`` and ``lgq_graph.py`` are also available on the repository under the folder logQ_graphs. The graphs include error bars (for most of them anyway) and the graphs whose titles begin with 'ss' include planets from our solar system.

| It should be noted that ``find_lgq.py`` takes a LONG time to run. For the seven data sets I had I believe it took about 11 hours or more to finish. I recommend running the script overnight.

| Finally it should be noted that all the changes and code written are commented (hopefully sufficiently) in the scripts.

.. _find_lgq.py: https://github.com/kpenev/RockyVsGaseousFromTides/blob/master/find_lgq.py
.. _logQ_graphs: https://github.com/kpenev/RockyVsGaseousFromTides/tree/master/logQ_graphs

Conclusion
==========
Obviously, the data shown by the graphs of log(Q) are a bit different than we were expecting, like the log(Q) vs. density graph showing the opposite of the planets in the solar system. Additionally there is a relationship between orbital period and log(Q) as well as the ratio of semi-major axis/radius and log(Q) for the exoplanets and not for the planets in our solar system. To try and explain these relationships we thought of these ideas:

- Since we are only solving for a lower limit on Q, it is possible the low density planets have larger Q and they just coicidentally all solved for a low log(Q)
- For some of our exoplanets, the eccentricity error bars show that the eccentricity may be consistent with an eccentricity of 0 in which case we can not infer anything about Q
- Some of the exoplanets we looked at are not alone in their system and their interactions with the other bodies in the system could be affecting their orbital evolution as gravitational interactions with that companion could be increasing the eccentricity while tides are decreasing it

That is pretty much all the work I was able to get done on this project over the summer. I'm hoping to maybe continue to work on it during the school year; however, I might be too busy with my class schedule. I'll leave off with a list of ways to continue the project:

- Find more planets on the NASA Exoplanet Archive with a wider variety of eccentricities, especially larger eccentricities
- Creating a model to examine systems with than two bodies instead of assuming there is only one planet in the system

Hopefully the work I did this summer will be helpful in continuing this project, and if there are ever any questions about something I did on this project please feel free to email me at: annamtetz@gmail.com

-- Anna Tetzlaff, 28 July 2019
