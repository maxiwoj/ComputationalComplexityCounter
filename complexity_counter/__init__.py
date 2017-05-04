from complexity_counter.atributes import *
from complexity_counter.testers import *




# import logging
#
# from complexity_counter import atributes, testers
# import random
#
# from complexity_counter.testers import test
#
#
# @atributes.complex_count
# class Sort(atributes.Algorithm):
#     def before(self, number_of_data):
#         self.rand_list = random.sample(range(number_of_data), number_of_data)
#
#     def run(self, number_of_data):
#         sorted(self.rand_list)
#
#     def after(self, number_of_data):
#         del self.rand_list
#
# result = testers.test(Sort, log_level=logging.INFO)
# print(result.computation_complexity)
# print(result.base)
# print(result.factors)
# print(result.max_complexity_predict(2000))
# print(result.time_predict(1000000))
# print(result.time_predict(5))