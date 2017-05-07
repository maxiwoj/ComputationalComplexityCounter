import logging
import signal
import time
import numpy as np
import os
import multiprocessing
from scipy.optimize import least_squares

from complexity_counter import TimeItResult
from complexity_counter import TestedAlgorithmError

base_types = 2
bases = list()
for base_size in range(5):
    bases += [[(lambda y: np.vectorize(lambda x, p: p * x ** y))(i)
               for i in [base_size]],
              [(lambda y: (lambda x, p: p * x ** y * np.log(x)))(i)
               for i in [base_size]]]


def model(p, x, base):
    """Function that gives value for a point in base of functions with p as
    factors """
    return sum([fun(x, p[i]) for i, fun in enumerate(base)])


def residuals(p, x, y, base):
    """Function returning difference bettwen value of the function and the
    actual value """
    return y - model(p, x, base)


def complexity_test(algorithm, log_level=logging.WARNING, timeout=30):
    """This function is the function to test the complexity for a given
    algorithm
    Attributes: algorithm - class implementing Algorithm class,
    that is wanted to be tested log_level - if set to logging.INFO - prints
    all the logs from testing the algorithm timeout - maximal time,
    the complexity testing must fit in.
    """
    algorithm = algorithm()
    if 'is_decorated' not in dir(algorithm):
        raise TestedAlgorithmError("Provided class is not decorated by "
                                   "@Complex_count")

    logging.basicConfig(level=log_level)

    fully_tested = True
    data, timings = test_timings(algorithm, timeout)

    if -1 in data or -1 in timings:
        first = np.where(data == -1)
        data = data[:first[0][0]]
        timings = timings[:first[0][0]]
        fully_tested = False

    x0_start_point = np.zeros(1)

    results = [least_squares(residuals, x0_start_point,
                             args=(data, timings, base)) for base in bases]

    costs = [result.cost for result in results]  # create list of costs for
    # every result
    base_index = costs.index(min(costs))  # minimal cost points the best result
    factors = list(results[base_index].x)  # get the factors for the
    # corresponding base

    computation_complexity = user_friendly_complexity(fully_tested,
                                                      base_index)

    return TimeItResult(computation_complexity, factors, bases[base_index],
                        data, timings)


def test_timings(algorithm, timeout):
    """This function tests the times of a given algorithm"""

    signal.signal(signal.SIGABRT, signalhandler)

    range_of_tests = 5
    number_of_tries = 5

    data = np.array([5 ** (x % range_of_tests)
                     for x in range(1, number_of_tries * range_of_tests + 1)])
    timings = np.zeros(range_of_tests * number_of_tries)

    for i, data_size in enumerate(data):
        start = time.time()

        p = start_timeout(timeout)
        try:
            algorithm.before(data_size)
            timings[i] = algorithm.run(data_size)
        except TimedOutExc:
            data[i:] = [-1] * len(data[i:])
            timings[i:] = [-1] * len(timings[i:])
            signal.signal(signal.SIGABRT, signal.SIG_DFL)
            return data, timings
        finally:
            p.terminate()
            if algorithm.need_to_clean:
                algorithm.after(data_size)
        timeout = timeout - (time.time() - start)

    for i in range(30):
        if timings[-1] > 3000:
            break
        data_size = int(9 ** 3 * 1.5 ** i)
        for j in range(2):
            start = time.time()

            data = np.append(data, data_size)

            p = start_timeout(timeout)
            try:
                algorithm.before(data_size)
                timings = np.append(timings, algorithm.run(data_size))
            except TimedOutExc:
                if len(data) != len(timings):
                    data = data[:-1]
                signal.signal(signal.SIGABRT, signal.SIG_DFL)
                return data, timings
            finally:
                p.terminate()
                if algorithm.need_to_clean:
                    algorithm.after(data_size)
            timeout = timeout - (time.time() - start)

    signal.signal(signal.SIGABRT, signal.SIG_DFL)
    return data, timings


def start_timeout(timeout):
    p = multiprocessing.Process(target=timeout_fun,
                                args=(timeout, os.getpid()))
    p.start()
    return p


def user_friendly_complexity(fully_tested, base_index):
    """Gives computational complexity in user friendly form using notation: 
    O(f(n)). Returned value is a string"""
    complexities = {
        0: {0: "O(c)", 1: "O(n)"},  # polynomial complexity
        1: {0: "O(log(n))", 1: "O(n log(n))"},  # logarithmic complexity
    }
    if not fully_tested:
        base_index -= 1  # make assumption that it was worse than the maximum
        #  of the smaller complexities

    complexity = base_index // base_types  # get the index specifying
    # the complexity (bases are ordered alternately)

    if base_index % base_types == 0:  # bases are ordered alternately,
        # so base_index % base_types = 0 means it is polynomial complexity
        computation_complexity = complexities[base_index % base_types].get(
            complexity, str.format("O(n^{})", complexity))
    else:  # since there are only 2 base_types everything is logarithmic
        # complexity
        computation_complexity = complexities[base_index % base_types].get(
            complexity, str.format("O(n^{} log(n))", complexity))

    if not fully_tested:
        computation_complexity = "Computation Complexity worse than: " + \
                                 computation_complexity
    return computation_complexity


def signalhandler(signum, frame):
    raise TimedOutExc("End of time")


class TimedOutExc(Exception):
    """
    Raised when a timeout happens
    """


def timeout_fun(timeout, pid):
    end_time = time.time() + timeout
    while end_time > time.time():
        time.sleep(0.01)
    os.kill(pid, signal.SIGABRT)


