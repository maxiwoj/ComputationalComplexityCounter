import logging
import numpy as np
from scipy.optimize import least_squares

from complexity_counter.atributes import TestedAlgorithmError
from complexity_counter.result import TimeItResult

logger = logging.getLogger(__name__)

base_types = 2
bases = list()
# for base_size in range(5):
#     bases += [[(lambda y: np.vectorize(lambda x, p: p * x**y))(i) for i in [0,base_size]],
#          [(lambda y: (lambda x, p: p * x**y * np.log(x)))(i) for i in [0, base_size]],
#          [(lambda y: np.vectorize(lambda x, p: p * y**x))(i) for i in [0, base_size]]]

for base_size in range(5):
    bases += [[(lambda y: np.vectorize(lambda x, p: p * x ** y))(i) for i in [0, base_size]],
              [(lambda y: (lambda x, p: p * x ** y * np.log(x)))(i) for i in [0, base_size]]]

complexities = {
    0: {0: "O(c)", 1: "O(n)"},
    1: {0: "O(log(n))", 1: "O(n log(n))"},
    2: {0: "O(c)", 1: "O(c)"}
}


def model(p, x, base):
    return sum([fun(x, p[i]) for i, fun in enumerate(base)])


def residuals(p, x, y, base):
    return y - model(p, x, base)


def test(algorithm, log_level=logging.warning, timeout=30):
    algorithm = algorithm()
    try:
        algorithm.isDecorated()
    except AttributeError:
        raise TestedAlgorithmError("Provided class is not decorated by @Complex_count")

    logging.basicConfig(level=log_level)

    number_of_tests = 5
    data, timings = test_timings(algorithm, number_of_tests)
    #     data = np.array([x for x in range(1,100)])
    #     timings = np.array([(10 *x)**2 + 100 * (10 * x) + 10 for x in range(1,100)])

    # print(data)
    # print(timings)
    results = list()
    x0_start_point = np.zeros(2)

    for base in bases:
        results.append(least_squares(residuals, x0_start_point, args=(data, timings, base)))

    costs = [result.cost for result in results]
    base = costs.index(min(costs))
    factors = list(results[base].x)
    complexity = int(base / base_types)  # factors.index(max(factors))

    if base % base_types == 0:
        computation_complexity = complexities[base % base_types].get(complexity, str.format("O(n^{})", complexity))
    else: # base % base_types == 1:
        computation_complexity = complexities[base % base_types].get(complexity, str.format("O(n^{} log(n))", complexity))
    # elif base % base_types == 2:
    #     computation_complexity = complexities[base % base_types].get(complexity, str.format("O({}^n)", complexity))

    return TimeItResult(computation_complexity, factors, bases[base])


def test_timings(algorithm, number_of_tests):
    data = np.array([5 ** x for x in range(1, number_of_tests + 1)])
    timings = np.zeros(number_of_tests)

    for i, number_of_data in enumerate(data):
        algorithm.before(number_of_data)
        timings[i] = algorithm.run(number_of_data)
        algorithm.after(number_of_data)

    print(timings)
    i = len(timings) - 3
    while timings[len(timings) - 1] < 3000 and len(timings) < 15:
        number_of_data = 9 ** (len(timings) - i)
        data = np.append(data, number_of_data)
        timings = np.append(timings, 0)
        algorithm.before(number_of_data)
        timings[len(timings) - 1] = algorithm.run(number_of_data)
        algorithm.after(number_of_data)

    return data, timings
