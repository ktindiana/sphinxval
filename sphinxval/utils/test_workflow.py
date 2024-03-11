# SUBROUTINES RELATED TO WORKFLOW UNIT TESTING
from . import object_handler as objh
from . import config
from . import validation
from . import metrics
from . import time_profile as profile
from . import resume
from . import plotting_tools as plt_tools

import unittest

""" utils/test_workflow.py contains subroutines to run unit tests on SPHINX workflow
"""



def basic_addition(a, b):
    return a + b

class test_workflow(unittest.TestCase):
    
    '''
    Notes:

    def setUp(self): <-- this function is a hook that is called once per test, useful for defining common parameters
        pass

    def test*(self): <-- this is the format required for unittest.main() to run tests automatically

    def tearDown(self): <-- this function is another hook that is called once per test, useful for removing variables from the self.* space
    '''


    def test_basic_addition(self):
        self.assertEqual(basic_addition(1, 2), 3, 'The sum is wrong.')





def run_all_tests():
    unittest.main(__name__)

