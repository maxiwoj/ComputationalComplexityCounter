import logging
import numpy as np

from scipy.optimize import least_squares

base_types = 2
bases = list()
for base_size in range(5):
    bases += [[(lambda y: np.vectorize(lambda x, p: p * x ** y))(i) for i in [base_size]],
              [(lambda y: (lambda x, p: p * x ** y * np.log(x)))(i) for i in [base_size]]]

complexities = {
    0: {0: "O(c)", 1: "O(n)"},
    1: {0: "O(log(n))", 1: "O(n log(n))"},
    2: {0: "O(c)", 1: "O(c)"}
}


def model(p, x, base):
    """Function that gives value for a point in base of functions with p as factors"""
    return sum([fun(x, p[i]) for i, fun in enumerate(base)])


def residuals(p, x, y, base):
    """Function returning difference bettwen value of the function and the actual value"""
    return y - model(p, x, base)


def complexity_test(algorithm, log_level=logging.WARNING, timeout=30):
    """This function is the function to test the complexity for a given algorithm
    Attributes:
        algorithm - class implementing Algorithm class, that is wanted to be tested
        log_level - if set to logging.INFO - prints all the logs from testing the algorithm
        timeout - maximal time, the complexity testing must fit in.
    """
    algorithm = algorithm()
    try:
        algorithm.is_decorated()
    except AttributeError:
        from complexity_counter import TestedAlgorithmError
        raise TestedAlgorithmError("Provided class is not decorated by @Complex_count")

    logging.basicConfig(level=log_level)

    number_of_tests = 5
    data, timings = test_timings(algorithm, number_of_tests)
    #     data = np.array([x for x in range(1,100)])
    #     timings = np.array([(10 *x)**2 + 100 * (10 * x) + 10 for x in range(1,100)])

    results = list()
    x0_start_point = np.zeros(1)

    for base in bases:
        results.append(least_squares(residuals, x0_start_point, args=(data, timings, base)))

    costs = [result.cost for result in results]
    base = costs.index(min(costs))
    factors = list(results[base].x)
    complexity = int(base / base_types)

    if base % base_types == 0:
        computation_complexity = complexities[base % base_types].get(complexity, str.format("O(n^{})", complexity))
    else:  # base % base_types == 1:
        computation_complexity = complexities[base % base_types].get(complexity,
                                                                     str.format("O(n^{} log(n))", complexity))
    # elif base % base_types == 2:
    #     computation_complexity = complexities[base % base_types].get(complexity, str.format("O({}^n)", complexity))

    from complexity_counter import TimeItResult
    return TimeItResult(computation_complexity, factors, bases[base], data, timings)


def test_timings(algorithm, number_of_tests):
    """This function tests the times of a given algorithm"""
    number_of_repeats = 5
    data = np.array([5 ** (x % number_of_repeats) for x in range(1, number_of_tests * number_of_repeats + 1)])
    timings = np.zeros(number_of_repeats * number_of_tests)

    for i, number_of_data in enumerate(data):
        algorithm.before(number_of_data)
        timings[i] = algorithm.run(number_of_data)
        algorithm.after(number_of_data)

    i = 0
    while timings[len(timings) - 1] < 3000 and i < 10:
        number_of_data = int(9**4 * 1.5 ** i)
        i += 1
        for j in range(2):
            data = np.append(data, number_of_data)
            algorithm.before(number_of_data)
            timings = np.append(timings, algorithm.run(number_of_data))
            algorithm.after(number_of_data)

    return data, timings
