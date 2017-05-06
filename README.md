computational complexity counter
===============================

version number: 0.0.1
author: Maksymilian Wojczuk

Overview
--------

Computational complexity counter is a package to test the computational complexity of a given algorithm.

Installation
--------------------

To install use pip:

    $ pip install git+https://github.com/AGHPythonCourse2017/zad2-maxiwoj


Or clone the repo:

    $ git clone https://github.com/AGHPythonCourse2017/zad2-maxiwoj
    $ python setup.py install
    
Usage
--------------------
To test the algorithm prepare class that implements class Algorithm from complexity_counter. Your class should implement those three methods: before, run, after, and should be decorated by complex_count. Basically it should look like: 

```
@complex_count
class YourClass(Algorithm):
def before(self, number_of_data):
        """This method is responsible for preparation data for algorithm to test"""
    def run(self, number_of_data):
        """The main method for testing the time of algorithm"""
    def after(self, number_of_data):
        """Method responsible for cleaning up after testing the time of the algorithm"""
```

To test the computation complexity of the algorithm prepared in method run() You need to run:

```
result = complex_test(YourClass)
```
    
As a result you will get a class, that holds information about the complexity of your algorithm.
To get the complexity in user-friendly form:
```
result.computation_complexity
```
To predict the time needed to complete the algorithm for given number_of_data type
```
result.time_predict(number_of_data)
```
Note, that it's results for small and much bigger number_of_data than tested
may differ from the real time, that the algorithm needs.

To predict how much data can the tested algorithm process in given time:
 ```
result.max_complexity_predict(number_of_data)
 ```


Contributing
------------

TBD

Example
-------

Script testing the computation complexity for build-in Python function - sorted:

```python
import complexity_counter as complexity
import random

@complexity.complex_count
class Sort(complexity.Algorithm):
    def before(self, number_of_data):
        self.rand_list = random.sample(range(number_of_data), number_of_data)

    def run(self, number_of_data):
        sorted(self.rand_list)

    def after(self, number_of_data):
        del self.rand_list
        
result = complexity.complexity_test(Sort)
```
