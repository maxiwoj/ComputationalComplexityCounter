from time import time

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
