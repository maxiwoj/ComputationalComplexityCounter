import logging
from time import time
import numpy as np
from scipy.optimize import least_squares

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
    try:
        algorithm.is_decorated()
    except AttributeError:
        from complexity_counter import TestedAlgorithmError
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

    results = list()
    x0_start_point = np.zeros(1)

    for base in bases:
        results.append(least_squares(residuals, x0_start_point,
                                     args=(data, timings, base)))

    costs = [result.cost for result in results]
    base_index = costs.index(min(costs))
    factors = list(results[base_index].x)
    complexity_index = int(base_index / base_types)

    computation_complexity = user_friendly_complexity(fully_tested,
                                                      complexity_index,
                                                      base_index)

    from complexity_counter import TimeItResult
    return TimeItResult(computation_complexity, factors, bases[base_index],
                        data, timings)


def test_timings(algorithm, timeout):
    """This function tests the times of a given algorithm"""

    time_left = timeout

    range_of_tests = 5
    number_of_tries = 5

    data = np.array([5 ** (x % range_of_tests)
                     for x in range(1, number_of_tries * range_of_tests + 1)])
    timings = np.zeros(range_of_tests * number_of_tries)
    import signal
    signal.signal(signal.SIGALRM, signalhandler)

    for i, number_of_data in enumerate(data):
        start = time()

        time_to_wait = int(round(time_left))
        if time_to_wait < 1:
            data[i:] = [-1] * len(data[i:])
            timings[i:] = [-1] * len(timings[i:])
            signal.signal(signal.SIGALRM, signal.SIG_DFL)
            return data, timings

        signal.alarm(time_to_wait)
        try:
            algorithm.before(number_of_data)
            timings[i] = algorithm.run(number_of_data)
        except TimedOutExc:
            data[i:] = [-1] * len(data[i:])
            timings[i:] = [-1] * len(timings[i:])
        finally:
            signal.alarm(0)
            if algorithm.need_to_clean:
                algorithm.after(number_of_data)
        time_left = time_left - (time() - start)

    i = 0
    while timings[len(timings) - 1] < 3000 and i < 30:
        number_of_data = int(9 ** 3 * 1.5 ** i)
        i += 1
        for j in range(2):
            start = time()

            time_to_wait = int(round(time_left))
            if time_to_wait < 1:
                signal.signal(signal.SIGALRM, signal.SIG_DFL)
                return data, timings

            data = np.append(data, number_of_data)
            signal.alarm(time_to_wait)

            try:
                algorithm.before(number_of_data)
                timings = np.append(timings, algorithm.run(number_of_data))
            except TimedOutExc:
                if len(data) != len(timings):
                    data = data[:-1]
            finally:
                signal.alarm(0)
                if algorithm.need_to_clean:
                    algorithm.after(number_of_data)
            time_left = time_left - (time() - start)

    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    return data, timings


def user_friendly_complexity(fully_tested, complexity, base_index):
    complexities = {
        0: {0: "O(c)", 1: "O(n)"},
        1: {0: "O(log(n))", 1: "O(n log(n))"},
    }
    if not fully_tested:
        complexity -= 1

    if base_index % base_types == 0:
        computation_complexity = complexities[base_index % base_types].get(
            complexity, str.format("O(n^{})", complexity))
    else:
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
