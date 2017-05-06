from time import time
import numpy as np
import logging

logger = logging.getLogger(__name__)


class TestedAlgorithmError(BaseException):
    """class for wrong implementations of the algorithm to be tested computation complexity

    Attributes:
        message -- explanation of why the specific transition is not allowed
    """

    def __init__(self, message):
        self.message = message


def complex_count(cls):
    """Class decorator allowing to test complexity of given algorithm"""
    if not issubclass(cls, Algorithm):
        raise TestedAlgorithmError("Provided class is not a subclass of Algorithm class")

    class Wrapped(object):
        def __init__(self):
            self.clsInstance = cls()

        def before(self, complexity):
            logger.info(str.format("setting up environment for complexity: {}", complexity))
            self.clsInstance.before(complexity)

        @check_time
        def run(self, complexity):
            logger.info(str.format("checking time for complexity: {}", complexity))
            self.clsInstance.run(complexity)

        def after(self, complexity):
            logger.info(str.format("cleaning up environment for complexity: {}", complexity))
            self.clsInstance.after(complexity)

        def is_decorated(self):
            return True

    return Wrapped


def check_time(fun):
    """Function decorator allowing to test time of the decorated function"""
    def checked_time_fun(self, number_of_args):
        start = time()
        fun(self, number_of_args)
        end = time()
        time_taken = 1000 * (end - start)
        logger.info(str.format("time: {}", time_taken))
        return time_taken

    return checked_time_fun


class Algorithm(object):
    """This class should be implemented with your 
    own algorithm you want to test computation complexity"""

    def before(self, number_of_data):
        """This method is responsible for preparation data for algorithm to test"""
        raise NotImplementedError("Tested class should override methods of Algorithm class: before, run, after")

    def run(self, number_of_data):
        """The main method for testing the time of algorithm"""
        raise NotImplementedError("Tested class should override methods of Algorithm class: before, run, after")

    def after(self, number_of_data):
        """Method responsible for cleaning up after testing the time of the algorithm"""
        raise NotImplementedError("Tested class should override methods of Algorithm class: before, run, after")


class TimeItResult:
    """Object of this class is the result of the test() function
    
    Attributes:
        data - tested number_of_data
        timings - time of the algorithm for corresponding number_of_data
        base - base of the function modelling the algorithm complexity
        factors - corrensponding factors for the base functions
        computation_complexity - string representation of the complexity in notation: O(f(n))
    """
    def __init__(self, computation_complexity, factors, base, data, timings):
        self.data = data
        self.timings = timings
        self.computation_complexity = computation_complexity
        self.factors = factors
        self.base = base

    def time_predict(self, number_of_data):
        """This function allows to predict the time needed to 
        complete the algorithm for given number_of_data. Note, 
        that it's results for small and much bigger number_of_data
        may differ from the real time, that the algorithm needs.
        """
        from complexity_counter import model
        return model(self.factors, np.array([number_of_data]), self.base)[0]
        # return sum([fun(np.array([x]), self.factors[i]) for i, fun in enumerate(self.base)])[0]

    def max_complexity_predict(self, time):
        """This function allows to predict maximal number_of_data
        for the algorithm to complete in given time
        """
        xmax = 10
        xmin = 0
        while time - self.time_predict(xmax) > 0:
            xmax = xmax * 10
        est = int((xmax + xmin) / 2)
        while abs(time - self.time_predict(est)) > 0.0001 and xmax - xmin > 1:
            if (time - self.time_predict(est)) > 0:
                xmin = int((xmax + xmin) / 2)
            else:
                xmax = est
            est = int((xmax + xmin) / 2)
        return est
