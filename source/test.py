#!/usr/bin/env python3

from ctypes import cdll
from ctypes.util import find_library
import os
import sys
from multiprocessing import Pool, Queue, Process
from time import sleep
from configargparse import ArgumentParser, DefaultsFormatter

from matplotlib import pyplot
from scipy import stats
import numpy

from dynesty import NestedSampler, plotting as dyplot

cstdlib = cdll.LoadLibrary(find_library('libc'))

#Work around emcee limitation
#pylint: disable=too-few-public-methods
class UnchunkedPool:
    """Disable chunking in Pool.map."""

    def __init__(self, pool):
        """Wrap around the given pool's map."""

        self._pool = pool

    def map(self, *args, **kwargs):
        """Delegate everything to parent, but set chunksize to 1."""

        self._pool.map(*args, **kwargs, chunksize=1)
#pylint: enable=too-few-public-methods

def prior_transform(unit_cube):
    """Insert fixd value between each input rv."""

    result = numpy.empty(unit_cube.size * 2 - 1)
    result[::2] = unit_cube
    result[2] = stats.rice.ppf(unit_cube[1], b = 0.775, scale=2.0)
    result[1::2] = numpy.arange(unit_cube.size - 1)
    return result

def log_likelihood(parameters):
    """For now just gaussian mean=0.5, sigma=0.1 for the first param."""

    return stats.norm.logpdf(parameters[0], loc=0.5, scale=0.1)

def sample(nrandom):
    """Create and run the sampler return the resluts."""

    sampler = NestedSampler(log_likelihood,
                            prior_transform,
                            ndim=(2*nrandom - 1),
                            npdim=nrandom,
                            nlive=5000)
    sampler.run_nested(dlogz=0.001)
    return sampler.results

def corner_plots(results):
    """Create and display a corner plot."""

    dyplot.cornerplot(results, show_titles=True, smooth=20)
    pyplot.show()

def stdout_redirect():
    """Demonstrate redirecting stdout of a process to file even from library."""

    destination = os.open(
        'test_redirect_%(pid)s.out' % dict(pid=os.getpid()),
        os.O_WRONLY | os.O_TRUNC | os.O_CREAT | os.O_DSYNC
    )
    os.dup2(destination, 1)
    os.dup2(destination, 2)

def test_redirect(value):
    """Check that stdout is properly redirected."""

    cstdlib.printf(
        (
            'Printing from C: %d  pid %d\n'
            %
            (value, os.getpid())
        ).encode('ascii')
    )
    print('Printing from python: ' + repr(value) + ' pid: ' + repr(os.getpid()))
    assert 1==0

def test_multiprocessing_io():
    """Test the scheme for redirecting multiprocessing I/O."""

    stdout_redirect()
    stdout_redirect()

    cstdlib.printf(
        (
            'C printing from global process (pid: %d)\n'
            %
            os.getpid()
        ).encode('ascii')
    )
    print('Python printing from global process (pid: %d)'
          %
          os.getpid())
    with Pool(4, initializer=stdout_redirect) as pool:
        workers = UnchunkedPool(pool)
        workers.map(test_redirect, range(10))

def f(x):
    """Function to run multiprocessing on."""

    print('f(%f)' % x)
    sleep(x)
    return x

def worker(input_queue, output_queue):
    """Get a task and run it for multiprocessing."""

    while True:
        task = input_queue.get()
        result = f(task)
        output_queue.put(result)

if __name__ == '__main__':
    tasks = Queue()
    results = Queue()

    for i in range(3):
        tasks.put(numpy.random.rand())

    workers = [Process(target=worker, args=(tasks, results))
               for i in range(4)]

    for process in workers:
        process.start()

    sevens_found = 0
    while sevens_found != 3:
        print('Sevens found: ' + repr(sevens_found))
        next_result = results.get()
        if int(next_result * 10) == 7:
            print('* ', end='')
            sevens_found += 1
        else:
            tasks.put(numpy.random.rand())
        print(next_result)

    for process in workers:
        process.terminate()
