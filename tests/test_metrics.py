import sphinxval.utils.metrics as metrics
# import sphinxval.sphinx
import unittest
import numpy as np
import math
import pandas as pd
from sklearn.metrics import brier_score_loss
from sklearn.utils.validation import check_consistent_length
from sklearn.utils.validation import check_array
import datetime

__version__ = "1.0"
__author__ = "Clayton Allison"


# Updated July 31, 2024

# test_metrics.py
# Metrics.py unittest file
# test_metrics.py is structured into four different classes, one for
# each type of validation currently done: flux, probability, contigency
# tables, and time. Each of these classes contains multiple tests for
# each metric/skill score that is calculated for multiple example cases.
# For example for contigency table metrics, there is are tests for a
# table of only hits, only misses, only false alarms, only correct
# negatives, a mixed table of one of each classification, and a complex
# table of a different number of each classification. For each test across
# the unittest, there is an assert statement for the result from
# metrics.py to be equal to a hand calculation of what the metric should
# be (sometimes this hand calculation is in the comments to show exactly
# what the math should be). There are instances where instead of being
# equal to a hand calculation, the assert statement is AssertTrue(isnan),
# and in these cases there is a comment of why the result should be a nan
# (stuff like 0/0 results or functions not liking what the input you give
# it). As more metrics are added to SPHINX (and metrics.py) add more tests
# to the corresponding class or add an additional class if it doesn't fit
# in the current classes.

# To run the unittest, use the following command in the command line,
# while in the main sphinxval directory:
# python -m unittest discover -v
# This command will find any unittests in any of the directories and run
# them, meaning once we start populating more unittests we may want to
# run only specific ones (not using discover). The -v is a shortened
# verbose statement, meaning the output to command line will be more
# detailed than a normal unittest.

# The output from running this command is:
# test_cont_complex_table (tests.test_metrics.ContigencyMetricsTestCase.
#     test_cont_complex_table) ... ok
# test_cont_mixed_table (tests.test_metrics.ContigencyMetricsTestCase.
#     test_cont_mixed_table) ... ok
# test_cont_only_correct_negatives (tests.test_metrics.
#     ContigencyMetricsTestCase.test_cont_only_correct_negatives) ... ok
# test_cont_only_false_alarms (tests.test_metrics.
#     ContigencyMetricsTestCase.test_cont_only_false_alarms) ... ok
# test_cont_only_hits (tests.test_metrics.ContigencyMetricsTestCase.
#     test_cont_only_hits) ... ok
# test_cont_only_misses (tests.test_metrics.ContigencyMetricsTestCase.
#     test_cont_only_misses) ... ok
# test_all_flux_metrics_switch_func (tests.test_metrics.
#     FluxMetricsTestCase.test_all_flux_metrics_switch_func) ... ok
# test_flux_metric_absolute_error_calcs (tests.test_metrics.
#     FluxMetricsTestCase.test_flux_metric_absolute_error_calcs) ... ok
# test_flux_metric_absolute_log_error_calcs (tests.test_metrics.
#     FluxMetricsTestCase.test_flux_metric_absolute_log_error_calcs)
#          ... ok
# test_flux_metric_absolute_percent_error_calcs (tests.test_metrics.
#     FluxMetricsTestCase.test_flux_metric_absolute_percent_error_calcs)
#          ... ok
# test_flux_metric_error_calcs (tests.test_metrics.FluxMetricsTestCase.
#     test_flux_metric_error_calcs) ... ok
# test_flux_metric_log_error_calcs (tests.test_metrics.
#     FluxMetricsTestCase.test_flux_metric_log_error_calcs) ... ok
# test_flux_metric_mean_acc_ratio_calcs (tests.test_metrics.
#     FluxMetricsTestCase.test_flux_metric_mean_acc_ratio_calcs) ... ok
# test_flux_metric_med_symm_acc_calcs (tests.test_metrics.
#     FluxMetricsTestCase.test_flux_metric_med_symm_acc_calcs) ... ok
# test_flux_metric_pearson_linear_corr_calcs (tests.test_metrics.
#     FluxMetricsTestCase.test_flux_metric_pearson_linear_corr_calcs)
#          ... ok
# test_flux_metric_pearson_log_corr_calcs (tests.test_metrics.
#     FluxMetricsTestCase.test_flux_metric_pearson_log_corr_calcs) ... ok
# test_flux_metric_percent_error_calcs (tests.test_metrics.
#     FluxMetricsTestCase.test_flux_metric_percent_error_calcs) ... ok
# test_flux_metric_root_mean_squared_error_calcs (tests.test_metrics.
#     FluxMetricsTestCase.test_flux_metric_root_mean_squared_error_calcs)
#         ... ok
# test_flux_metric_root_mean_squared_log_error_calcs (tests.test_metrics.
#     FluxMetricsTestCase.test_flux_metric_root_mean_squared_
#         log_error_calcs) ... ok
# test_flux_metric_spearman_corr_calcs (tests.test_metrics.
#     FluxMetricsTestCase.test_flux_metric_spearman_corr_calcs) ... ok
# test_flux_metric_squared_error_calcs (tests.test_metrics.
#     FluxMetricsTestCase.test_flux_metric_squared_error_calcs) ... ok
# test_flux_metric_squared_log_error_calcs (tests.test_metrics.
#     FluxMetricsTestCase.test_flux_metric_squared_log_error_calcs)
#         ... ok
# test_flux_metric_symmetric_absolute_percent_error_calcs (tests.
#     test_metrics.FluxMetricsTestCase.test_flux_metric_symmetric_
#         absolute_percent_error_calcs) ... ok
# test_flux_metric_symmetric_percent_error_calcs (tests.test_metrics.
#     FluxMetricsTestCase.test_flux_metric_symmetric_percent_error_calcs)
#         ... ok
# test_prob_area_under_roc_curve (tests.test_metrics.
#     ProbabilityMetricsTestCase.test_prob_area_under_roc_curve) ...
#         C:\Users\cfalliso\AppData\Local\Programs\Python\Python311
#         Lib\site-packages\sklearn\metrics\_ranking.py:1124:
#         UndefinedMetricWarning: No negative samples in y_true, false
#         positive value should be meaningless
#     warnings.warn(
#     ok
# test_prob_brier (tests.test_metrics.
#     ProbabilityMetricsTestCase.test_prob_brier) ... ok
# test_prob_brier_skill (tests.test_metrics.
#     ProbabilityMetricsTestCase.test_prob_brier_skill) ... ok
# test_prob_pearson_lin (tests.test_metrics.
#     ProbabilityMetricsTestCase.test_prob_pearson_lin) ... ok
# test_time_mean_absolue_error (tests.test_metrics.
#     TimeMetricsTestCase.test_time_mean_absolue_error) ... ok
# test_time_mean_error (tests.test_metrics.
#     TimeMetricsTestCase.test_time_mean_error) ... ok
# test_time_median_absolute_error (tests.test_metrics.
#     TimeMetricsTestCase.test_time_median_absolute_error) ... ok
# test_time_median_error (tests.test_metrics.
#     TimeMetricsTestCase.test_time_median_error) ... ok

# ----------------------------------------------------------------------
# Ran 32 tests in 0.042s

# OK

# More on Unittests can be found at:
# https://docs.python.org/3/library/unittest.html


class FluxMetricsTestCase(unittest.TestCase):
     
    # Error ###############################################################################################
    def test_flux_metric_error_calcs(self):        
        y_true = [10]
        y_pred = [1]
        result = metrics.calc_E(y_true, y_pred)
        hand_calc =  y_pred[0] - y_true[0]
        self.assertAlmostEqual(result[0], hand_calc)


        y_true = [10, 10, 10, 10]
        y_pred = [11, 1, 11, 1]
        zipped = zip(y_pred, y_true)
        temp = []
        for elements in zipped:
            temp.append(elements[0] - elements[1])
        hand_calc = np.mean(temp)
        result = np.mean(metrics.calc_E(y_true, y_pred))
        self.assertAlmostEqual(result, hand_calc)

        y_true = [10, 10, 10, 10]
        y_pred = [1, 1, 1, 1]
        zipped = zip(y_pred, y_true)
        temp = []
        for elements in zipped:
            temp.append(elements[0] - elements[1])
        hand_calc = np.mean(temp)
        result = np.mean(metrics.calc_E(y_true, y_pred))
        self.assertTrue(result < 0.0)

        y_true = [10, 10, 10, 10]
        y_pred = [11, 11, 11, 11]
        zipped = zip(y_pred, y_true)
        temp = []
        for elements in zipped:
            temp.append(elements[0] - elements[1])
        hand_calc = np.mean(temp)
        result = np.mean(metrics.calc_E(y_true, y_pred))
        self.assertTrue(result > 0.0)
        



    # Absolute Error ###############################################################################################
    def test_flux_metric_absolute_error_calcs(self):
        y_true = [10]
        y_pred = [1]
        result = metrics.calc_AE(y_true, y_pred)
        hand_calc =  np.abs(y_pred[0] - y_true[0])
        self.assertAlmostEqual(result[0], hand_calc)


        y_true = [10, 10, 10, 10]
        y_pred = [11, 1, 11, 1]
        zipped = zip(y_pred, y_true)
        temp = []
        for elements in zipped:
            temp.append(np.abs(elements[0] - elements[1]))
        hand_calc = np.mean(temp)
        result = np.mean(metrics.calc_AE(y_true, y_pred))
        self.assertAlmostEqual(result, hand_calc)

        y_true = [10, 10, 10, 10]
        y_pred = [1, 1, 1, 1]
        zipped = zip(y_pred, y_true)
        temp = []
        for elements in zipped:
            temp.append(np.abs(elements[0] - elements[1]))
        hand_calc = np.mean(temp)
        result = np.mean(metrics.calc_AE(y_true, y_pred))
        self.assertTrue(result > 0.0)

        y_true = [10, 10, 10, 10]
        y_pred = [11, 11, 11, 11]
        zipped = zip(y_pred, y_true)
        temp = []
        for elements in zipped:
            temp.append(np.abs(elements[0] - elements[1]))
        hand_calc = np.mean(temp)
        result = np.mean(metrics.calc_AE(y_true, y_pred))
        self.assertTrue(result > 0.0)




#     # # Log Error ################################################################################
    def test_flux_metric_log_error_calcs(self):
        y_true = [10]
        y_pred = [1]
        result = metrics.calc_LE(y_true, y_pred)
        hand_calc = np.log10(y_pred[0]) - np.log10(y_true[0])
        self.assertAlmostEqual(result[0], -1)


        y_true = [10, 10, 10, 10]
        y_pred = [11, 1, 11, 1]
        zipped = zip(y_pred, y_true)
        temp = []
        for elements in zipped:
            temp.append(np.log10(elements[0]) - np.log10(elements[1]))
        hand_calc = np.mean(temp)
        result = np.mean(metrics.calc_LE(y_true, y_pred))
        self.assertAlmostEqual(result, hand_calc)

        y_true = [10, 10, 10, 10]
        y_pred = [1, 1, 1, 1]
        zipped = zip(y_pred, y_true)
        temp = []
        for elements in zipped:
            temp.append(np.log10(elements[0]) - np.log10(elements[1]))
        hand_calc = np.mean(temp)
        result = np.mean(metrics.calc_LE(y_true, y_pred))
        self.assertTrue(result < 0.0)

        y_true = [10, 10, 10, 10]
        y_pred = [11, 11, 11, 11]
        zipped = zip(y_pred, y_true)
        temp = []
        for elements in zipped:
            temp.append(np.log10(elements[0]) - np.log10(elements[1]))
        hand_calc = np.mean(temp)
        result = np.mean(metrics.calc_LE(y_true, y_pred))
        self.assertTrue(result > 0.0)

        y_true = [-10]
        y_pred = [11]
        with self.assertRaises(ValueError):
            result = metrics.calc_LE(y_true, y_pred)


#     # # Abs Log Error ################################################################################
    def test_flux_metric_absolute_log_error_calcs(self):
        y_true = [10]
        y_pred = [1]
        result = metrics.calc_ALE(y_true, y_pred)
        hand_calc = np.abs(np.log10(y_pred[0]) - np.log10(y_true[0]))
        self.assertAlmostEqual(result[0], hand_calc)


        y_true = [10, 10, 10, 10]
        y_pred = [11, 1, 11, 1]
        zipped = zip(y_pred, y_true)
        temp = []
        for elements in zipped:
            temp.append(np.abs(np.log10(elements[0]) - np.log10(elements[1])))
        hand_calc = np.mean(temp)
        result = np.mean(metrics.calc_ALE(y_true, y_pred))
        self.assertAlmostEqual(result, hand_calc)

        y_true = [10, 10, 10, 10]
        y_pred = [1, 1, 1, 1]
        zipped = zip(y_pred, y_true)
        temp = []
        for elements in zipped:
            temp.append(np.abs(np.log10(elements[0]) - np.log10(elements[1])))
        hand_calc = np.mean(temp)
        result = np.mean(metrics.calc_ALE(y_true, y_pred))
        self.assertTrue(result > 0.0)

        y_true = [10, 10, 10, 10]
        y_pred = [11, 11, 11, 11]
        zipped = zip(y_pred, y_true)
        temp = []
        for elements in zipped:
            temp.append(np.abs(np.log10(elements[0]) - np.log10(elements[1])))
        hand_calc = np.mean(temp)
        result = np.mean(metrics.calc_ALE(y_true, y_pred))
        self.assertTrue(result > 0.0)

        y_true = [-10]
        y_pred = [11]
        with self.assertRaises(ValueError):
            result = metrics.calc_ALE(y_true, y_pred)

#     # # Squared Error ################################################################################
    def test_flux_metric_squared_error_calcs(self):
        y_true = [10]
        y_pred = [1]
        result = metrics.calc_SE(y_true, y_pred)
        hand_calc = (y_pred[0] - y_true[0])**2
        self.assertAlmostEqual(result[0], hand_calc)


        y_true = [10, 10, 10, 10]
        y_pred = [11, 1, 11, 1]
        zipped = zip(y_pred, y_true)
        temp = []
        for elements in zipped:
            temp.append((elements[0] - elements[1])**2)
        hand_calc = np.mean(temp)
        result = np.mean(metrics.calc_SE(y_true, y_pred))
        self.assertAlmostEqual(result, hand_calc)

        y_true = [10, 10, 10, 10]
        y_pred = [1, 1, 1, 1]
        zipped = zip(y_pred, y_true)
        temp = []
        for elements in zipped:
            temp.append((elements[0] - elements[1])**2)
        hand_calc = np.mean(temp)
        result = np.mean(metrics.calc_SE(y_true, y_pred))
        self.assertTrue(result > 0.0)

        y_true = [10, 10, 10, 10]
        y_pred = [11, 11, 11, 11]
        zipped = zip(y_pred, y_true)
        temp = []
        for elements in zipped:
            temp.append((elements[0] - elements[1])**2)
        hand_calc = np.mean(temp)
        result = np.mean(metrics.calc_SE(y_true, y_pred))
        self.assertTrue(result > 0.0)



#     # # Squared Log Error ################################################################################
    def test_flux_metric_squared_log_error_calcs(self):
        y_true = [10]
        y_pred = [1]
        result = metrics.calc_SLE(y_true, y_pred)
        hand_calc = (np.log10(y_pred[0]) - np.log10(y_true[0]))**2
        self.assertAlmostEqual(result[0], hand_calc)


        y_true = [10, 10, 10, 10]
        y_pred = [11, 1, 11, 1]
        zipped = zip(y_pred, y_true)
        temp = []
        for elements in zipped:
            temp.append((np.log10(elements[0]) - np.log10(elements[1]))**2)
        hand_calc = np.mean(temp)
        result = np.mean(metrics.calc_SLE(y_true, y_pred))
        self.assertAlmostEqual(result, hand_calc)

        y_true = [10, 10, 10, 10]
        y_pred = [1, 1, 1, 1]
        zipped = zip(y_pred, y_true)
        temp = []
        for elements in zipped:
            temp.append((np.log10(elements[0]) - np.log10(elements[1]))**2)
        hand_calc = np.mean(temp)
        result = np.mean(metrics.calc_SLE(y_true, y_pred))
        self.assertTrue(result > 0.0)

        y_true = [10, 10, 10, 10]
        y_pred = [11, 11, 11, 11]
        zipped = zip(y_pred, y_true)
        temp = []
        for elements in zipped:
            temp.append((np.log10(elements[0]) - np.log10(elements[1]))**2)
        hand_calc = np.mean(temp)
        result = np.mean(metrics.calc_SLE(y_true, y_pred))
        self.assertTrue(result > 0.0)
    
        y_true = [-10]
        y_pred = [11]
        with self.assertRaises(ValueError):
            result = metrics.calc_SLE(y_true, y_pred)
    
    
#     # # Root Mean Squared Error ################################################################################
    def test_flux_metric_root_mean_squared_error_calcs(self):
        y_true = [10]
        y_pred = [1]
        result = metrics.calc_RMSE(y_true, y_pred)
        hand_calc = math.sqrt((y_pred[0] - y_true[0])**2)
        # hand_calc = math.sqrt(sum(error)/len(error))
        
        self.assertAlmostEqual(result, hand_calc)


        y_true = [10, 10, 10, 10]
        y_pred = [11, 1, 11, 1]
        zipped = zip(y_pred, y_true)
        temp = []
        for elements in zipped:
            temp.append((elements[0] - elements[1])**2)
        hand_calc = math.sqrt(sum(temp)/len(temp))
        result = np.mean(metrics.calc_RMSE(y_true, y_pred))
        self.assertAlmostEqual(result, hand_calc)

        y_true = [10, 10, 10, 10]
        y_pred = [1, 1, 1, 1]
        zipped = zip(y_pred, y_true)
        temp = []
        for elements in zipped:
            temp.append((elements[0] - elements[1])**2)
        hand_calc = math.sqrt(sum(temp)/len(temp))
        result = np.mean(metrics.calc_RMSE(y_true, y_pred))
        self.assertTrue(result > 0.0)

        y_true = [10, 10, 10, 10]
        y_pred = [11, 11, 11, 11]
        zipped = zip(y_pred, y_true)
        temp = []
        for elements in zipped:
            temp.append((elements[0] - elements[1])**2)
        hand_calc = math.sqrt(sum(temp)/len(temp))
        result = np.mean(metrics.calc_RMSE(y_true, y_pred))
        self.assertTrue(result > 0.0)



   
#     # # Root Mean Squared Log Error ################################################################################
    def test_flux_metric_root_mean_squared_log_error_calcs(self):
        y_true = [10]
        y_pred = [1]
        result = metrics.calc_RMSLE(y_true, y_pred)
        hand_calc = math.sqrt((np.log10(y_pred[0]) - np.log10(y_true[0]))**2)
        self.assertAlmostEqual(result, hand_calc)


        y_true = [10, 10, 10, 10]
        y_pred = [11, 1, 11, 1]
        zipped = zip(y_pred, y_true)
        temp = []
        for elements in zipped:
            temp.append((np.log10(elements[0]) - np.log10(elements[1]))**2)
        hand_calc = math.sqrt(sum(temp)/len(temp))
        result = np.mean(metrics.calc_RMSLE(y_true, y_pred))
        self.assertAlmostEqual(result, hand_calc)

        y_true = [10, 10, 10, 10]
        y_pred = [1, 1, 1, 1]
        zipped = zip(y_pred, y_true)
        temp = []
        for elements in zipped:
            temp.append((np.log10(elements[0]) - np.log10(elements[1]))**2)
        hand_calc = math.sqrt(sum(temp)/len(temp))
        result = np.mean(metrics.calc_RMSLE(y_true, y_pred))
        self.assertTrue(result > 0.0)

        y_true = [10, 10, 10, 10]
        y_pred = [11, 11, 11, 11]
        zipped = zip(y_pred, y_true)
        temp = []
        for elements in zipped:
            temp.append((np.log10(elements[0]) - np.log10(elements[1]))**2)
        hand_calc = math.sqrt(sum(temp)/len(temp))
        result = np.mean(metrics.calc_RMSLE(y_true, y_pred))
        self.assertTrue(result > 0.0)
  
        y_true = [-10]
        y_pred = [11]
        with self.assertRaises(ValueError):
            result = metrics.calc_RMSLE(y_true, y_pred)
    

#     # # Percent Error ################################################################################
    def test_flux_metric_percent_error_calcs(self):
        y_true = [10]
        y_pred = [1]
        result = metrics.calc_PE(y_true, y_pred)
        hand_calc = (y_pred[0] - y_true[0]) / y_true[0]
        self.assertAlmostEqual(result[0], hand_calc)


        y_true = [10, 10, 10, 10]
        y_pred = [11, 1, 11, 1]
        zipped = zip(y_pred, y_true)
        temp = []
        for elements in zipped:
            temp.append((elements[0] - elements[1]) / elements[1])
        hand_calc = np.mean(temp)
        result = np.mean(metrics.calc_PE(y_true, y_pred))
        self.assertAlmostEqual(result, hand_calc)

        y_true = [10, 10, 10, 10]
        y_pred = [1, 1, 1, 1]
        zipped = zip(y_pred, y_true)
        temp = []
        for elements in zipped:
            temp.append((elements[0] - elements[1]) / elements[1])
        hand_calc = np.mean(temp)
        result = np.mean(metrics.calc_PE(y_true, y_pred))
        self.assertTrue(result < 0.0)

        y_true = [10, 10, 10, 10]
        y_pred = [11, 11, 11, 11]
        zipped = zip(y_pred, y_true)
        temp = []
        for elements in zipped:
            temp.append((elements[0] - elements[1]) / elements[1])
        hand_calc = np.mean(temp)
        result = np.mean(metrics.calc_PE(y_true, y_pred))
        self.assertTrue(result > 0.0)

        y_true = [0]
        y_pred = [11]
        with self.assertRaises(ValueError):
            result = metrics.calc_PE(y_true, y_pred)

    

#     # # Abs Percent Error ################################################################################
    def test_flux_metric_absolute_percent_error_calcs(self):
        y_true = [10]
        y_pred = [1]
        result = metrics.calc_APE(y_true, y_pred)
        hand_calc = np.abs(y_pred[0] - y_true[0]) / y_true[0]
        self.assertAlmostEqual(result[0], hand_calc)


        y_true = [10, 10, 10, 10]
        y_pred = [11, 1, 11, 1]
        zipped = zip(y_pred, y_true)
        temp = []
        for elements in zipped:
            temp.append(np.abs(elements[0] - elements[1]) / elements[1])
        hand_calc = np.mean(temp)
        result = np.mean(metrics.calc_APE(y_true, y_pred))
        self.assertAlmostEqual(result, hand_calc)

        y_true = [10, 10, 10, 10]
        y_pred = [1, 1, 1, 1]
        zipped = zip(y_pred, y_true)
        temp = []
        for elements in zipped:
            temp.append(np.abs(elements[0] - elements[1]) / elements[1])
        hand_calc = np.mean(temp)
        result = np.mean(metrics.calc_APE(y_true, y_pred))
        self.assertTrue(result > 0.0)

        y_true = [10, 10, 10, 10]
        y_pred = [11, 11, 11, 11]
        zipped = zip(y_pred, y_true)
        temp = []
        for elements in zipped:
            temp.append(np.abs(elements[0] - elements[1]) / elements[1])
        hand_calc = np.mean(temp)
        result = np.mean(metrics.calc_APE(y_true, y_pred))
        self.assertTrue(result > 0.0)

        y_true = [0]
        y_pred = [11]
        with self.assertRaises(ValueError):
            result = metrics.calc_APE(y_true, y_pred)

#     # # Symmetric Percent Error ################################################################################
    def test_flux_metric_symmetric_percent_error_calcs(self):
        y_true = [10]
        y_pred = [1]
        result = metrics.calc_SPE(y_true, y_pred)
        hand_calc =  2.0 * (y_pred[0] - y_true[0]) / (y_pred[0] + y_true[0])
        self.assertAlmostEqual(result[0], hand_calc)


        y_true = [10, 10, 10, 10]
        y_pred = [11, 1, 11, 1]
        zipped = zip(y_pred, y_true)
        temp = []
        for elements in zipped:
            temp.append(2.0* (elements[0] - elements[1]) / (elements[0] + elements[1]))
        hand_calc = np.mean(temp)
        result = np.mean(metrics.calc_SPE(y_true, y_pred))
        self.assertAlmostEqual(result, hand_calc)

        y_true = [10, 10, 10, 10]
        y_pred = [1, 1, 1, 1]
        zipped = zip(y_pred, y_true)
        temp = []
        for elements in zipped:
            temp.append(2.0* (elements[0] - elements[1]) / (elements[0] + elements[1]))
        hand_calc = np.mean(temp)
        result = np.mean(metrics.calc_SPE(y_true, y_pred))
        self.assertTrue(result < 0.0)

        y_true = [10, 10, 10, 10]
        y_pred = [11, 11, 11, 11]
        zipped = zip(y_pred, y_true)
        temp = []
        for elements in zipped:
            temp.append(2.0* (elements[0] - elements[1]) / (elements[0] + elements[1]))
        hand_calc = np.mean(temp)
        result = np.mean(metrics.calc_SPE(y_true, y_pred))
        self.assertTrue(result > 0.0)
    
        y_true = [-11]
        y_pred = [11]
        with self.assertRaises(ValueError):
            result = metrics.calc_SPE(y_true, y_pred)
        # ValueError("Symmetric Percent Error cannot be used when predicted targets and true targets sum to zero.") 


#     # # Symmetric Abs Percent Error ################################################################################
    def test_flux_metric_symmetric_absolute_percent_error_calcs(self):
        y_true = [10]
        y_pred = [1]
        result = metrics.calc_SAPE(y_true, y_pred)
        hand_calc =  2.0 * np.abs(y_pred[0] - y_true[0]) / (y_pred[0] + y_true[0])
        self.assertAlmostEqual(result[0], hand_calc)


        y_true = [10, 10, 10, 10]
        y_pred = [11, 1, 11, 1]
        zipped = zip(y_pred, y_true)
        temp = []
        for elements in zipped:
            temp.append(2.0* np.abs(elements[0] - elements[1]) / (elements[0] + elements[1]))
        hand_calc = np.mean(temp)
        result = np.mean(metrics.calc_SAPE(y_true, y_pred))
        self.assertAlmostEqual(result, hand_calc)

        y_true = [10, 10, 10, 10]
        y_pred = [1, 1, 1, 1]
        zipped = zip(y_pred, y_true)
        temp = []
        for elements in zipped:
            temp.append(2.0* np.abs(elements[0] - elements[1]) / (elements[0] + elements[1]))
        hand_calc = np.mean(temp)
        result = np.mean(metrics.calc_SAPE(y_true, y_pred))
        self.assertTrue(result > 0.0)

        y_true = [10, 10, 10, 10]
        y_pred = [11, 11, 11, 11]
        zipped = zip(y_pred, y_true)
        temp = []
        for elements in zipped:
            temp.append(2.0* np.abs(elements[0] - elements[1]) / (elements[0] + elements[1]))
        hand_calc = np.mean(temp)
        result = np.mean(metrics.calc_SAPE(y_true, y_pred))
        self.assertTrue(result > 0.0)

        y_true = [0]
        y_pred = [0]
        with self.assertRaises(ValueError):
            result = metrics.calc_SAPE(y_true, y_pred)
        # ValueError("Symmetric Absolute Percent Error cannot be used when predicted targets and true targets sum to zero.") 

#     # # Linear Pearson Correlation Coefficient ################################################################################
    def test_flux_metric_pearson_linear_corr_calcs(self):
        y_true = [10]
        y_pred = [1]
        result = metrics.calc_pearson(y_true, y_pred)
        self.assertTrue(math.isnan(result[0]))
        # Gives a nan since the scipy pearson does not like it when you only give it one obs/pred pair. 
        

        y_true = [10, 1, 1, 10]
        y_pred = [11, 1, 11, 1]
        zipped = zip(y_pred, y_true)
        result = metrics.calc_pearson(y_true, y_pred)
        self.assertAlmostEqual(result[0], 0)

        y_true = [10, 11, 12, 13]
        y_pred = [1, 2, 3, 4]
        zipped = zip(y_pred, y_true)
        result = metrics.calc_pearson(y_true, y_pred)
        self.assertAlmostEqual(result[0], 1)

        y_true = [10, 11, 12, 13]
        y_pred = [4, 3, 2, 1]
        result = metrics.calc_pearson(y_true, y_pred)
        self.assertAlmostEqual(result[0], -0.9999999999999999)

    
#     # # Log Pearson Correlation Coefficient ################################################################################
    def test_flux_metric_pearson_log_corr_calcs(self):
        y_true = [10]
        y_pred = [1]
        result = metrics.calc_pearson(y_true, y_pred)
        # Asserts NaN since pearson needs more than one obs/pred pair to find a correlation
        self.assertTrue(math.isnan(result[1]))
        

        y_true = [10, 1, 1, 10]
        y_pred = [11, 1, 11, 1]
        zipped = zip(y_pred, y_true)
        result = metrics.calc_pearson(y_true, y_pred)
        self.assertAlmostEqual(result[1], 0)

        y_true = [1, 10, 100, 1000]
        y_pred = [1, 10, 100, 1000]
        zipped = zip(y_pred, y_true)
        result = metrics.calc_pearson(y_true, y_pred)
        self.assertAlmostEqual(result[1], 1)
       

        y_true = [1, 10, 100, 1000]
        y_pred = [1000, 100, 10, 1]
        result = metrics.calc_pearson(y_true, y_pred)
        self.assertAlmostEqual(result[1], -0.9999999999999999)




#     # # Mean Accuracy Ratio ################################################################################
    def test_flux_metric_mean_acc_ratio_calcs(self):
        y_true = [10]
        y_pred = [1]
        result = metrics.calc_MAR(y_true, y_pred)
        hand_calc = y_pred[0] / y_true[0]
        self.assertAlmostEqual(result, hand_calc)
        

        y_true = [10, 1, 1, 10]
        y_pred = [11, 1, 11, 1]
        zipped = zip(y_pred, y_true)
        temp = []
        for elements in zipped:
            temp.append(elements[0] / elements[1])
        hand_calc = np.mean(temp)
        result = metrics.calc_MAR(y_true, y_pred)
        self.assertAlmostEqual(result, hand_calc)

        y_true = [10, 11, 12, 13]
        y_pred = [0, 1, 2, 3]
        zipped = zip(y_pred, y_true)
        temp = []
        for elements in zipped:
            temp.append(elements[0] / elements[1])
        hand_calc = np.mean(temp)
        result = metrics.calc_MAR(y_true, y_pred)
        self.assertAlmostEqual(result, hand_calc)

        y_true = [10, 11, 12, 13]
        y_pred = [3, 2, 1, 0]
        zipped = zip(y_pred, y_true)
        temp = []
        for elements in zipped:
            temp.append(elements[0] / elements[1])
        hand_calc = np.mean(temp)
        result = metrics.calc_MAR(y_true, y_pred)
        self.assertAlmostEqual(result, hand_calc)

        y_true = [10]
        y_pred = [-1]
        with self.assertRaises(ValueError):
            result = metrics.calc_MAR(y_true, y_pred)
        


#      # # Median Symmetric Accuracy ################################################################################
    def test_flux_metric_med_symm_acc_calcs(self):
        y_true = [10]
        y_pred = [1]
        result = metrics.calc_MdSA(y_true, y_pred)
        hand_calc = (np.exp(np.median(np.abs(np.log(y_true[0] / y_pred[0])))) - 1.0)
        self.assertAlmostEqual(result, hand_calc)
        

        y_true = np.array([10, 1, 1, 10])
        y_pred = np.array([11, 1, 11, 1])
        zipped = zip(y_pred, y_true)
        temp = []
        hand_calc = (np.exp(np.median(np.abs(np.log(y_true/ y_pred)))) - 1.0)
        result = metrics.calc_MdSA(y_true, y_pred)
        self.assertAlmostEqual(result, hand_calc)

        y_true = np.array([10, 11, 12, 13])
        y_pred = np.array([1, 2, 3, 4])
        hand_calc = (np.exp(np.median(np.abs(np.log(y_true/ y_pred)))) - 1.0)
        result = metrics.calc_MdSA(y_true, y_pred)
        self.assertAlmostEqual(result, hand_calc)

        y_true = np.array([10, 11, 12, 13])
        y_pred = np.array([4, 3, 2, 1])
        hand_calc = (np.exp(np.median(np.abs(np.log(y_true/ y_pred)))) - 1.0)
        result = metrics.calc_MdSA(y_true, y_pred)
        self.assertAlmostEqual(result, hand_calc)

        y_true = [10]
        y_pred = [-1]
        with self.assertRaises(ValueError):
            result = metrics.calc_MdSA(y_true, y_pred)


# # # Spearman Correlation Coefficient ################################################################################
    def test_flux_metric_spearman_corr_calcs(self):
        y_true = [10]
        y_pred = [1]
        result = metrics.calc_spearman(y_true, y_pred)
        self.assertTrue(math.isnan(result))
        # Gives nan since spearman needs more than one obs/pred pair
        

        y_true = [10, 1, 1, 10]
        y_pred = [11, 1, 11, 1]
        zipped = zip(y_pred, y_true)
        result = metrics.calc_spearman(y_true, y_pred)
        self.assertAlmostEqual(result, 0)

        y_true = [10, 10]
        y_pred = [1]
        with self.assertRaises(ValueError):
            result = metrics.calc_spearman(y_true, y_pred)
        # Gives error since spearman needs equal numbers of obs/pred pairs



# # # Testing All ################################################################################
    def test_all_flux_metrics_switch_func(self):
        y_true = np.array([10, 1, 1, 10])
        y_pred = np.array([11, 1, 11, 1])
        metrics_list = ['E', 'AE', 'LE', 'ALE', 'SE', 'SLE', 'RMSE', 'RMSLE', 'PE', 'APE', 'SPE', 'SAPE', 'r_lin', 'r_log', 'MAR', 'MdSA', 'spearman']
        for metric in metrics_list:
            if metric == 'r_lin':
                metric = 'r'
                func_call = metrics.switch_error_func(metric, y_true, y_pred)[0]
                metric = 'r_lin'
            elif metric == 'r_log':
                metric = 'r'
                func_call = metrics.switch_error_func(metric, y_true, y_pred)[1]
                metric = 'r_log'
            else:
                func_call = metrics.switch_error_func(metric, y_true, y_pred)
            if metric == 'E':
                hand_calc = np.mean(y_pred - y_true)
                self.assertAlmostEqual(np.mean(func_call), hand_calc)
            elif metric == 'AE':
                hand_calc = np.mean(np.abs(y_pred - y_true))
                self.assertAlmostEqual(np.mean(func_call), hand_calc)
            elif metric == 'LE':
                hand_calc = np.mean(np.log10(y_pred) - np.log10(y_true))
                self.assertAlmostEqual(np.mean(func_call), hand_calc)
            elif metric == 'ALE':
                hand_calc = np.mean(np.abs(np.log10(y_pred) - np.log10(y_true)))
                self.assertAlmostEqual(np.mean(func_call), hand_calc)
            elif metric == 'SE':
                hand_calc = np.mean((y_pred - y_true)**2)
                self.assertAlmostEqual(np.mean(func_call), hand_calc)
            elif metric == 'SLE':
                hand_calc = np.mean((np.log10(y_pred) - np.log10(y_true))**2)
                self.assertAlmostEqual(np.mean(func_call), hand_calc)
            elif metric == 'RMSE':
                error = (y_pred - y_true)**2
                hand_calc = math.sqrt(sum(error)/ len(error))
                self.assertAlmostEqual(np.mean(func_call), hand_calc)
            elif metric == 'RMSLE':
                error = (np.log10(y_pred) - np.log10(y_true))**2
                hand_calc = math.sqrt(sum(error)/len(error))
                self.assertAlmostEqual(np.mean(func_call), hand_calc)
            elif metric == 'PE':
                hand_calc = np.mean((y_pred - y_true) / y_true)
                self.assertAlmostEqual(np.mean(func_call), hand_calc)
            elif metric == 'APE':
                hand_calc = np.mean(np.abs(y_pred - y_true) / y_true)
                self.assertAlmostEqual(np.mean(func_call), hand_calc)
            elif metric == 'SPE':
                hand_calc = np.mean(2.0 * (y_pred - y_true) / (y_pred + y_true))
                self.assertAlmostEqual(np.mean(func_call), hand_calc)
            elif metric == 'SAPE':
                hand_calc = np.mean(2.0 * np.abs(y_pred - y_true) / (y_pred + y_true))
                self.assertAlmostEqual(np.mean(func_call), hand_calc)
            elif metric == 'r_lin':
                self.assertAlmostEqual(func_call, 0)
            elif metric == 'r_log':
                self.assertAlmostEqual(func_call, 0)
            elif metric == 'MAR':
                hand_calc = np.mean(y_pred / y_true)
                self.assertAlmostEqual(func_call, hand_calc)
            elif metric == 'MdSA':
                hand_calc = (np.exp(np.median(np.abs(np.log(y_true / y_pred)))) - 1.0)
                self.assertAlmostEqual(func_call, hand_calc)
            elif metric == 'spearman':
                self.assertAlmostEqual(func_call, 0)


        metric = 'not_a_metric' # Testing the error raise       
        with self.assertRaises(ValueError): #, msg = 'not_a_metric is an invalid metric.')
            func_call = metrics.switch_error_func(metric, y_true, y_pred)    

class ProbabilityMetricsTestCase(unittest.TestCase):
#   # Brier Score
    def test_prob_brier(self):
        y_true = [1]
        y_pred = [0.1]
        result = metrics.calc_brier(y_true, y_pred)
        hand_calc =  (y_pred[0] - y_true[0])**2
        self.assertAlmostEqual(result, hand_calc)


        y_true = [1., 1., 0., 0.]
        y_pred = [1., 0.1, 1., 0.1]
        zipped = zip(y_pred, y_true)
        temp = []
        for elements in zipped:
            temp.append((elements[0] - elements[1])**2)
        hand_calc = np.mean(temp)
        result = np.mean(metrics.calc_brier(y_true, y_pred))
        self.assertAlmostEqual(result, hand_calc)

        y_true = [1, 1, 1, 1]
        y_pred = [0.1, 0.1, 0.1, 0.1]
        zipped = zip(y_pred, y_true)
        temp = []
        for elements in zipped:
            temp.append((elements[0] - elements[1])**2)
        hand_calc = np.mean(temp)
        result = np.mean(metrics.calc_brier(y_true, y_pred))
        self.assertTrue(result > 0.0)

        y_true = [0, 0, 0, 0]
        y_pred = [1, 1, 1, 1]
        zipped = zip(y_pred, y_true)
        temp = []
        for elements in zipped:
            temp.append((elements[0] - elements[1])**2)
        hand_calc = np.mean(temp)
        result = np.mean(metrics.calc_brier(y_true, y_pred))
        self.assertTrue(result > 0.0)


        y_true = [1., 1., 0., 0.]
        y_pred = [1., None, 1., None]
        y_true_none = [1,0]
        y_pred_none = [1,1]
        zipped = zip(y_pred_none, y_true_none)
        temp = []
        for elements in zipped:
            temp.append((elements[0] - elements[1])**2)
        hand_calc = np.mean(temp)
        result = np.mean(metrics.calc_brier(y_true, y_pred))
        self.assertAlmostEqual(result, hand_calc)
    
    
#   # Brier Skill Score    
    def test_prob_brier_skill(self):
        hb_clim = 0.033 #Hazel's climatology (Bain et al. 2021)
        y_true = [1]
        y_pred = [0.1]
        clim = np.ndarray(np.size(y_pred))
        clim.fill(hb_clim)
        clim_score = brier_score_loss(y_true, clim)
        result = metrics.calc_brier_skill(y_true, y_pred)
        hand_calc =  1 - ((y_pred[0] - y_true[0])**2 / clim_score)
        self.assertAlmostEqual(result, hand_calc)


        y_true = [1, 1, 0, 0]
        y_pred = [1, 0.1, 1, 0.1]
        zipped = zip(y_pred, y_true)
        clim = np.ndarray(np.size(y_pred))
        clim.fill(hb_clim)
        temp = []
        for elements in zipped:
            temp.append((elements[0] - elements[1])**2)
        clim_score = brier_score_loss(y_true, clim)
        hand_calc = 1 - (np.mean(temp) / clim_score)
        result = metrics.calc_brier_skill(y_true, y_pred)
        self.assertAlmostEqual(result, hand_calc)

        y_true = [1, 1, 1, 1]
        y_pred = [0.1, 0.1, 0.1, 0.1]
        zipped = zip(y_pred, y_true)
        clim = np.ndarray(np.size(y_pred))
        clim.fill(hb_clim)
        temp = []
        for elements in zipped:
            temp.append((elements[0] - elements[1])**2)
        clim_score = brier_score_loss(y_true, clim)
        hand_calc = 1 - (np.mean(temp) / clim_score)
        result = np.mean(metrics.calc_brier_skill(y_true, y_pred))
        self.assertTrue(result > 0.0)

        y_true = [0, 0, 0, 0]
        y_pred = [1, 1, 1, 1]
        zipped = zip(y_pred, y_true)
        clim = np.ndarray(np.size(y_pred))
        clim.fill(hb_clim)
        temp = []
        for elements in zipped:
            temp.append((elements[0] - elements[1])**2)
        clim_score = brier_score_loss(y_true, clim)
        hand_calc = 1 - (np.mean(temp) / clim_score)
        result = np.mean(metrics.calc_brier_skill(y_true, y_pred))
        self.assertTrue(result < 0.0)
        
#   # Linear Pearson Correlation Coefficient
    def test_prob_pearson_lin(self):
        y_true = [1]
        y_pred = [0.1]
        result = metrics.calc_pearson(y_true, y_pred)
        self.assertTrue(math.isnan(result[0]))
        # Asserts NaN since pearson needs more than one obs/pred pair to find a correlation
        

        y_true = [0, 1, 1, 0]
        y_pred = [0.1, 1, 0.1, 1]
        result = metrics.calc_pearson(y_true, y_pred, 'linear')
        self.assertAlmostEqual(result[0], 0)

        y_true = [0.0, 1.0, 1.0, 1.0]
        y_pred = [0.0, 0.1, 0.2, 0.3]
        result = metrics.calc_pearson(y_true, y_pred, 'linear')
        self.assertTrue(result[0] > 0.0)

        y_true = [0.0, 1.0, 1.0, 1.0]
        y_pred = [0.3, 0.2, 0.1, 0.0]
        result = metrics.calc_pearson(y_true, y_pred, 'linear')
        self.assertTrue(result[0] < 0.0)

    
#   # Receiver Operator Characteristic Area Under the Curve
    def test_prob_area_under_roc_curve(self):
        from scipy import integrate
        import sklearn.metrics as skl
        y_true = [1.0, 0.0]
        y_pred = [1.0, 0.0]
        result, _ = metrics.receiver_operator_characteristic(y_true, y_pred, 'Test')
        self.assertAlmostEqual(result, 1)

        y_true = [1.0, 1.0, 1.0, 1.0]
        y_pred = [1.0, 1.0, 1.0, 1.0]
        result, _ = metrics.receiver_operator_characteristic(y_true, y_pred, 'Test')
        self.assertWarns(Warning, msg= 'No negative samples in y_true, false positive value should be meaningless')
        # self.assertTrue(math.isnan(result))
        # Will give the following warning message:
        # UndefinedMetricWarning: No negative samples in y_true, false positive value should be meaningless



class ContigencyMetricsTestCase(unittest.TestCase):
    # Since this part of metrics.py does not have seperate
    # subroutines for each metric but calculates them all at once
    # so my unittest for this just builds six example contigency
    # tables that it'll calculate the metrics for each to test.
    def test_cont_only_hits(self):
        y_true = [10]
        y_pred = [11]
        thresh = 10
        # contigency table 1 0
        #                  0 0
        result = metrics.calc_contingency(y_true, y_pred, thresh)
        
        # Iterating through result dictionary
        for score in result:
            # print(score)
            if score == 'TP':
                self.assertAlmostEqual(result[score], 1)
                # 1 hit
            elif score == 'FN':
                self.assertAlmostEqual(result[score], 0)
                # 0 Misses
            elif score == 'FP':
                self.assertAlmostEqual(result[score], 0)
                # 0 False Alarms
            elif score == 'TN':
                self.assertAlmostEqual(result[score], 0)
                # 0 True Negatives
            elif score == 'PC':
                self.assertAlmostEqual(result[score], 1)
                # PC = (h+c)/1 = 1+0/1
            elif score == 'B':
                self.assertAlmostEqual(result[score], 1)
                # B = h+f / h+m = 1+0/1+0
            elif score == 'H':
                self.assertAlmostEqual(result[score], 1)
                # H = h/h+m = 1 / 1+0 
            elif score == 'FAR':
                self.assertAlmostEqual(result[score], 0)
                # FAR = f / h+f = 0 / 1+0
            elif score == 'F':
                self.assertTrue(math.isnan(result[score]))
                # F = f / f+c = 0/0
            elif score == 'FOH':
                self.assertAlmostEqual(result[score], 1)
                # FOH = h / h+f = 1 / 1+0
            elif score == 'FOM':
                self.assertAlmostEqual(result[score], 0)
                # FOM = m / h+m = 0 / 1+0
            elif score == 'POCN':
                self.assertTrue(math.isnan(result[score]))
                # POCN = c / c+f = 0 / 0
            elif score == 'DFR':
                self.assertTrue(math.isnan(result[score]))
                # DFR = m / m+c = 0/0
            elif score == 'FOCN':
                self.assertTrue(math.isnan(result[score]))
                # FOCN = c / m+c = 0/0
            elif score == 'TS': 
                self.assertAlmostEqual(result[score], 1)
                # TS = h / h+f+m = 1/1+0+0
            elif score == 'OR':
                self.assertTrue(math.isnan(result[score]))
                # OR = hc / fm = 1*0/0*0
            elif score == 'GSS':
                self.assertTrue(math.isnan(result[score]))
                # GSS = h-(h+f)*(h+m)/n / h+f+m-(h+f)*(h+m)/n = 1-(1+0)*(1+0)/1 / 1+0+0-(1+0)(1+0)/1
            elif score == 'TSS':
                self.assertTrue(math.isnan(result[score]))
                # TSS = h / h+m - f / f+c = 1 / 1+0 - 0 / 0
            elif score == 'HSS':
                self.assertTrue(math.isnan(result[score]))
                # HSS = 2.0*(h*c-f*m) / (h+m)*(m+c)+(h+f)*(f+c) = 2*(1*0 - 0*0) / (1+0)(0+0)+(1+0)(0+0)
            elif score == 'ORSS':
                self.assertTrue(math.isnan(result[score]))
                # ORSS = h*c-m*f / h*c+m*f = 1*0-0*0 / 1*0+0+0
            elif score == 'SEDS':
                self.assertTrue(math.isnan(result[score]))
                # SEDS = (log((h+f)/n)+log((h+m)/n) / log(h/n)) - 1 = (log((1+0)/1)+log((1+0)/1) / log(1/1)) - 1
            elif score == 'FONE':
                self.assertAlmostEqual(result[score], 1)
            elif score == 'FTWO':
                self.assertAlmostEqual(result[score], 1)
            elif score == 'FHALF':
                self.assertAlmostEqual(result[score], 1)
                # F-Scores (Beta = 0.5, 1, 2) = ((1+ Beta^2)* h) / ((1+Beta^2)*h + Beta^2 * m + f)
                # F(Beta = 0.5) = ((1+0.5^2)*0) / ((1+0.5^2)*0 + 1 + 0)
                # F(Beta = 1) = ((1+1^2)*1) / ((1+1^2)*1 + 0 + 0)
                # F(Beta = 2) = ((1+2^2)*1) / ((1+2^2)*1 + 0 + 0)
            elif score == 'PREV':
                self.assertAlmostEqual(result[score], 1)
                # Prevalence = (h+m)/n = (1+0)/1
            elif score == 'MCC':
                self.assertTrue(math.isnan(result[score]))
                # Matthew Correlation Coefficient = (h*c-f*m)/Sqrt((h+f)*(h+m)*(c+f)*(f+m)) = (1*0-0*0)/sqrt((1+0)*(1+0)*(0+0)*(0+0))
            elif score == 'INFORM':
                self.assertTrue(math.isnan(result[score]))
                # Informedness = h/(h+m) + c/(f+c) - 1 = 1/(1+0) + 0/(0+0) - 1
            elif score == 'MARK':
                self.assertTrue(math.isnan(result[score]))
                # Markedness = h/(h+f) + c/(f+c) - 1 = 1/(1+0) + 0/(0+0) - 1
            elif score == 'PT':
                self.assertTrue(math.isnan(result[score]))
                # Prevalence Threshold = (Sqrt(h/(h+m)*f/(f+c))-(f/(f+c))) / (h/(h+m)-f/(f+c)) = (Sqrt(1/(1+0)*0/(0+0))-(0/(0+0))) / (1/(1+0)-0/(0+0))
            elif score == 'BA':
                self.assertTrue(math.isnan(result[score]))
                # Balanced Accuracy = (1/2)*(h/(h+m)+c/(f+c) = 1/2 * (1/(1+0)+ 0/(0+0))
            elif score == 'FM':
                self.assertAlmostEqual(result[score],1)
                # Fowlkes-Mallows Index = Sqrt((h/(h+f))*(h/(h+m))) = sqrt((1/(1+0))*((1/(1+0)))
            elif score == 'FAER':
                self.assertAlmostEqual(result[score],0)
                # FAER = f / (h + m) = 0 / (1 + 0)
            elif score == 'Tau':
                self.assertTrue(math.isnan(result[score]))
                # Tau = 1 - (np.sqrt((f/(c + f))**2 + (m/(h + m))**2)/np.sqrt(2)) = 1 - sqrt((0 / 0)^2 + (0/(1 + 0))^2 )



    def test_cont_only_misses(self):
        y_true = [10]
        y_pred = [1]
        thresh = 10
        # contigency table 0 0
        #                  1 0
        result = metrics.calc_contingency(y_true, y_pred, thresh)
        
        for score in result:
            # print(score)
            if score == 'TP':
                self.assertAlmostEqual(result[score], 0)
                # 0 hits
            elif score == 'FN':
                self.assertAlmostEqual(result[score], 1)
                # 1 miss
            elif score == 'FP':
                self.assertAlmostEqual(result[score], 0)
                # 0 false alarms
            elif score == 'TN':
                self.assertAlmostEqual(result[score], 0)
                # 0 correct negatives
            elif score == 'PC':
                self.assertAlmostEqual(result[score], 0)
                # PC = h+c / n = 0+0 / 1
            elif score == 'B':
                self.assertAlmostEqual(result[score], 0)
                # B = h+f / h+m = 0+0 / 0+1
            elif score == 'H':
                self.assertAlmostEqual(result[score], 0)
                # H = h / h+m = 0 / 1+0
            elif score == 'FAR':
                self.assertTrue(math.isnan(result[score]))
                # FAR = f / h+f = 0 / 0+0
            elif score == 'F':
                self.assertTrue(math.isnan(result[score]))
                # f = f / f+c = 0 / 0+0
            elif score == 'FOH':
                self.assertTrue(math.isnan(result[score]))
                # FOH = h / h+f = 0/ 0+0
            elif score == 'FOM':
                self.assertAlmostEqual(result[score], 1)
                # FOM = m / h+m = 1 / 0+1
            elif score == 'POCN':
                self.assertTrue(math.isnan(result[score]))
                # POCN = c / f+c = 0 / 0+0
            elif score == 'DFR':
                self.assertAlmostEqual(result[score], 1)
                # DFR = m / m+c = 1 / 1+0
            elif score == 'FOCN':
                self.assertAlmostEqual(result[score], 0)
                # FOCN = c / m+c = 0 / 1+0
            elif score == 'TS': 
                self.assertAlmostEqual(result[score], 0)
                # TS = h / h+f+m = 0 / 1+0+0
            elif score == 'OR':
                self.assertTrue(math.isnan(result[score]))
                # OR = hc / fm = 0*0 / 0*1
            elif score == 'GSS':
                self.assertAlmostEqual(result[score], 0)
                # GSS = h-(h+f)*(h+m)/n / h+f+m-(h+f)*(h+m)/n = 0-(0+0)*(0+1)/1 / 0+0+1-(0+0)(0+1)/1
            elif score == 'TSS':
                self.assertTrue(math.isnan(result[score]))
                # TSS = h / h+m - f / f+c = 0 / 1+0 - 0 / 0
            elif score == 'HSS':
                self.assertAlmostEqual(result[score], 0)
                # HSS = 2.0*(h*c-f*m) / (h+m)*(m+c)+(h+f)*(f+c) = 2*(0*0 - 0*1) / (1+0)(1+0)+(0+0)(0+0)
            elif score == 'ORSS':
                self.assertTrue(math.isnan(result[score]))
                # ORSS = h*c-m*f / h*c+m*f = 0*0-1*0 / 0*0+1*0
            elif score == 'SEDS':
                self.assertTrue(math.isnan(result[score]))
                # SEDS = (log((h+f)/n)+log((h+m)/n) / log(h/n)) - 1 = (log((0+0)/1)+log((0+0)/1) / log(0/1)) - 1
                # Will throw out the following error messages (but will pass unittest)
                # RuntimeWarning: divide by zero encountered in log
                # RuntimeWarning: invalid value encountered in scalar divide
            elif score == 'FONE':
                self.assertAlmostEqual(result[score], 0)
            elif score == 'FTWO':
                self.assertAlmostEqual(result[score], 0)
            elif score == 'FHALF':
                self.assertAlmostEqual(result[score], 0)
                # F-Scores (Beta = 0.5, 1, 2) = ((1+ Beta^2)* h) / ((1+Beta^2)*h + Beta^2 * m + f)
                # F(Beta = 0.5) = ((1+0.5^2)*0) / ((1+0.5^2)*0 + 1 + 0)
                # F(Beta = 1) = ((1+1^2)*0) / ((1+1^2)*0 + 1 + 0)
                # F(Beta = 2) = ((1+2^2)*0) / ((1+2^2)*0 + 1 + 0)
            elif score == 'PREV':
                self.assertAlmostEqual(result[score], 1)
                # Prevalence = (h+m)/n = (0+1)/1
            elif score == 'MCC':
                self.assertTrue(math.isnan(result[score]))
                # Matthew Correlation Coefficient = (h*c-f*m)/Sqrt((h+f)*(h+m)*(c+f)*(f+m)) = (0*0-0*1)/sqrt((0+0)*(0+1)*(0+0)*(0+1))
            elif score == 'INFORM':
                self.assertTrue(math.isnan(result[score]))
                # Informedness = h/(h+m) + c/(f+c) - 1 = 0/(0+1) + 0/(0+0) - 1
            elif score == 'MARK':
                self.assertTrue(math.isnan(result[score]))
                # Markedness = h/(h+f) + c/(f+c) - 1 = 0/(0+0) + 0/(0+0) - 1
            elif score == 'PT':
                self.assertTrue(math.isnan(result[score]))
                # Prevalence Threshold = (Sqrt(h/(h+m)*f/(f+c))-(f/(f+c))) / (h/(h+m)-f/(f+c)) = (Sqrt(0/(0+1)*0/(0+0))-(0/(0+0))) / (0/(0+1)-0/(0+0))
            elif score == 'BA':
                self.assertTrue(math.isnan(result[score]))
                # Balanced Accuracy = (1/2)*(h/(h+m)+c/(f+c) = 1/2 * (1/(1+0)+ 0/(0+0))
            elif score == 'FM':
                self.assertTrue(math.isnan(result[score]))
                # Fowlkes-Mallows Index = Sqrt((h/(h+f))*(h/(h+m))) = sqrt((0/(0+0))*((0/(0+1)))
            elif score == 'FAER':
                self.assertAlmostEqual(result[score],0)
                # FAER = f / (h + m) = 0 / (0 + 1)
            elif score == 'Tau':
                self.assertTrue(math.isnan(result[score]))
                # Tau = 1 - (np.sqrt((f/(c + f))**2 + (m/(h + m))**2)/np.sqrt(2)) = 1 - sqrt((0 / 0)^2 + (1/(1 + 0))^2 )

        

    def test_cont_only_false_alarms(self):
        y_true = [1]
        y_pred = [10]
        thresh = 10
        result = metrics.calc_contingency(y_true, y_pred, thresh)
        # contigency table 0 1
        #                  0 0
        for score in result:
            # print(score)
            if score == 'TP':
                self.assertAlmostEqual(result[score], 0)
                # 0 hits
            elif score == 'FN':
                self.assertAlmostEqual(result[score], 0)
                # 0 miss
            elif score == 'FP':
                self.assertAlmostEqual(result[score], 1)
                # 1 false alarms
            elif score == 'TN':
                self.assertAlmostEqual(result[score], 0)
                # 0 correct negatives
            elif score == 'PC':
                self.assertAlmostEqual(result[score], 0)
                # PC = h+c / n = 0+0 / 1
            elif score == 'B':
                self.assertTrue(result[score], 0)
                # B = h+f / h+m = 0+1 / 0+0
            elif score == 'H':
                self.assertTrue(math.isnan(result[score]))
                # H = h / h+m = 0 / 0+0
            elif score == 'FAR':
                self.assertTrue(result[score], 1)
                # FAR = f / h+f = 1 / 0+1
            elif score == 'F':
                self.assertTrue(result[score], 1)
                # f = f / f+c = 1 / 1+0
            elif score == 'FOH':
                self.assertAlmostEqual(result[score], 0)
                # FOH = h / h+f = 0/ 0+1
            elif score == 'FOM':
                self.assertTrue(math.isnan(result[score]))
                # FOM = m / h+m = 0 / 0+0
            elif score == 'POCN':
                self.assertAlmostEqual(result[score], 0)
                # POCN = c / f+c = 0 / 1+0
            elif score == 'DFR':
                self.assertTrue(math.isnan(result[score]))
                # DFR = m / m+c = 0 / 0+0
            elif score == 'FOCN':
                self.assertTrue(math.isnan(result[score]))
                # FOCN = c / m+c = 0 / 0+0
            elif score == 'TS': 
                self.assertAlmostEqual(result[score], 0)
                # TS = h / h+f+m = 0 / 0+1+0
            elif score == 'OR':
                self.assertTrue(math.isnan(result[score]))
                # OR = hc / fm = 0*0 / 0*1
            elif score == 'GSS':
                self.assertAlmostEqual(result[score], 0)
                # GSS = h-(h+f)*(h+m)/n / h+f+m-(h+f)*(h+m)/n = 0-(0+1)*(0+0)/1 / 0+1+0-(0+1)(0+0)/1
            elif score == 'TSS':
                self.assertTrue(math.isnan(result[score]))
                # TSS = h / h+m - f / f+c = 0 / 0+0 - 1 / 1+0
            elif score == 'HSS':
                self.assertAlmostEqual(result[score], 0)
                # HSS = 2.0*(h*c-f*m) / (h+m)*(m+c)+(h+f)*(f+c) = 2*(0*0 - 0*1) / (0+0)(0+0)+(0+1)(1+0)
            elif score == 'ORSS':
                self.assertTrue(math.isnan(result[score]))
                # ORSS = h*c-m*f / h*c+m*f = 0*0-0*1 / 0*0+1*0
            elif score == 'SEDS':
                self.assertTrue(math.isnan(result[score]))
                # SEDS = (log((h+f)/n)+log((h+m)/n) / log(h/n)) - 1 = (log((0+1)/1)+log((0+0)/1) / log(0/1)) - 1
                # Will throw out the following error messages (but will pass unittest)
                # RuntimeWarning: divide by zero encountered in log
                # RuntimeWarning: invalid value encountered in scalar divide
            elif score == 'FONE':
                self.assertAlmostEqual(result[score], 0)
            elif score == 'FTWO':
                self.assertAlmostEqual(result[score], 0)
            elif score == 'FHALF':
                self.assertAlmostEqual(result[score], 0)
                # F-Scores (Beta = 0.5, 1, 2) = ((1+ Beta^2)* h) / ((1+Beta^2)*h + Beta^2 * m + f)
                # F(Beta = 0.5) = ((1+0.5^2)*0) / ((1+0.5^2)*0 + 0 + 1)
                # F(Beta = 1) = ((1+1^2)*0) / ((1+1^2)*0 + 0 + 1)
                # F(Beta = 2) = ((1+2^2)*0) / ((1+2^2)*0 + 0 + 1)
            elif score == 'PREV':
                self.assertAlmostEqual(result[score], 0)
                # Prevalence = (h+m)/n = (0+0)/1
            elif score == 'MCC':
                self.assertTrue(math.isnan(result[score]))
                # Matthew Correlation Coefficient = (h*c-f*m)/Sqrt((h+f)*(h+m)*(c+f)*(f+m)) = (0*0-1*0)/sqrt((1+0)*(0+0)*(0+1)*(1+0))
            elif score == 'INFORM':
                self.assertTrue(math.isnan(result[score]))
                # Informedness = h/(h+m) + c/(f+c) - 1 = 0/(0+0) + 0/(1+0) - 1
            elif score == 'MARK':
                self.assertTrue(math.isnan(result[score]))
                # Markedness = h/(h+f) + c/(m+c) - 1 = 0/(0+1) + 0/(0+0) - 1
            elif score == 'PT':
                self.assertTrue(math.isnan(result[score]))
                # Prevalence Threshold = (Sqrt(h/(h+m)*f/(f+c))-(f/(f+c))) / (h/(h+m)-f/(f+c)) = (Sqrt(0/(0+0)*1/(1+0))-(1/(1+0))) / (0/(0+0)-1/(1+0))
            elif score == 'BA':
                self.assertTrue(math.isnan(result[score]))
                # Balanced Accuracy = (1/2)*(h/(h+m)+c/(f+c) = 1/2 * (0/(0+0)+ 0/(1+0))
            elif score == 'FM':
                self.assertTrue(math.isnan(result[score]))
                # Fowlkes-Mallows Index = Sqrt((h/(h+f))*(h/(h+m))) = sqrt((0/(0+1))*((0/(0+0)))
            elif score == 'FAER':
                self.assertAlmostEqual(result[score], np.inf)
                # FAER = f / (h + m) = 1 / (0 + 0)
            elif score == 'Tau':
                self.assertTrue(math.isnan(result[score]))
                # Tau = 1 - (np.sqrt((f/(c + f))**2 + (m/(h + m))**2)/np.sqrt(2)) = 1 - sqrt((1 / 1)^2 + (0/(0 + 0))^2 )


    def test_cont_only_correct_negatives(self):
        y_true = [1]
        y_pred = [1]
        thresh = 10
        result = metrics.calc_contingency(y_true, y_pred, thresh)
        # contigency table 0 0
        #                  0 1
        for score in result:
            # print(score)
            if score == 'TP':
                self.assertAlmostEqual(result[score], 0)
                # 0 hits
            elif score == 'FN':
                self.assertAlmostEqual(result[score], 0)
                # 0 miss
            elif score == 'FP':
                self.assertAlmostEqual(result[score], 0)
                # 0 false alarms
            elif score == 'TN':
                self.assertAlmostEqual(result[score], 1)
                # 1 correct negatives
            elif score == 'PC':
                self.assertAlmostEqual(result[score], 1)
                # PC = h+c / n = 0+1 / 1
            elif score == 'B':
                self.assertTrue(math.isnan(result[score]))
                # B = h+f / h+m = 0+0 / 0+0
            elif score == 'H':
                self.assertTrue(math.isnan(result[score]))
                # H = h / h+m = 0 / 0+0
            elif score == 'FAR':
                self.assertTrue(math.isnan(result[score]))
                # FAR = f / h+f = 0 / 0+0
            elif score == 'F':
                self.assertAlmostEqual(result[score], 0)
                # f = f / f+c = 0 / 1+0
            elif score == 'FOH':
                self.assertTrue(math.isnan(result[score]))
                # FOH = h / h+f = 0/ 0+0
            elif score == 'FOM':
                self.assertTrue(math.isnan(result[score]))
                # FOM = m / h+m = 0 / 0+0
            elif score == 'POCN':
                self.assertAlmostEqual(result[score], 1)
                # POCN = c / f+c = 1 / 1+0
            elif score == 'DFR':
                self.assertAlmostEqual(result[score], 0)
                # DFR = m / m+c = 0 / 0+1
            elif score == 'FOCN':
                self.assertAlmostEqual(result[score], 1)
                # FOCN = c / m+c = 1 / 0+1
            elif score == 'TS': 
                self.assertTrue(math.isnan(result[score]))
                # TS = h / h+f+m = 0 / 0+0+0
            elif score == 'OR':
                self.assertTrue(math.isnan(result[score]))
                # OR = hc / fm = 0*1 / 0*0
            elif score == 'GSS':
                self.assertTrue(math.isnan(result[score]))
                # GSS = h-(h+f)*(h+m)/n / h+f+m-(h+f)*(h+m)/n = 0-(0+0)*(0+0)/1 / 0+0+0-(0+0)(0+0)/1
            elif score == 'TSS':
                self.assertTrue(math.isnan(result[score]))
                # TSS = h / h+m - f / f+c = 0 / 0+0 - 0 / 1+0
            elif score == 'HSS':
                self.assertTrue(math.isnan(result[score]))
                # HSS = 2.0*(h*c-f*m) / (h+m)*(m+c)+(h+f)*(f+c) = 2*(0*1 - 0*0) / (0+0)(0+1)+(0+0)(1+0)
            elif score == 'ORSS':
                self.assertTrue(math.isnan(result[score]))
                # ORSS = h*c-m*f / h*c+m*f = 0*1-0*0 / 0*1+0*0
            elif score == 'SEDS':
                self.assertTrue(math.isnan(result[score]))
                # SEDS = (log((h+f)/n)+log((h+m)/n) / log(h/n)) - 1 = (log((0+0)/1)+log((0+0)/1) / log(0/1)) - 1
                # Will throw out the following error messages (but will pass unittest)
                # RuntimeWarning: divide by zero encountered in log
                # RuntimeWarning: invalid value encountered in scalar divide
            elif score == 'FONE':
                self.assertTrue(math.isnan(result[score]))
            elif score == 'FTWO':
                self.assertTrue(math.isnan(result[score]))
            elif score == 'FHALF':
                self.assertTrue(math.isnan(result[score]))
                # F-Scores (Beta = 0.5, 1, 2) = ((1+ Beta^2)* h) / ((1+Beta^2)*h + Beta^2 * m + f)
                # F(Beta = 0.5) = ((1+0.5^2)*0) / ((1+0.5^2)*0 + 0 + 0)
                # F(Beta = 1) = ((1+1^2)*0) / ((1+1^2)*0 + 0 + 0)
                # F(Beta = 2) = ((1+2^2)*0) / ((1+2^2)*0 + 0 + 0)
            elif score == 'PREV':
                self.assertAlmostEqual(result[score], 0)
                # Prevalence = (h+m)/n = (0+0)/1
            elif score == 'MCC':
                self.assertTrue(math.isnan(result[score]))
                # Matthew Correlation Coefficient = (h*c-f*m)/Sqrt((h+f)*(h+m)*(c+f)*(f+m)) = (1*0-0*0)/sqrt((0+0)*(0+0)*(1+0)*(0+0))
            elif score == 'INFORM':
                self.assertTrue(math.isnan(result[score]))
                # Informedness = h/(h+m) + c/(f+c) - 1 = 0/(0+0) + 1/(0+1) - 1
            elif score == 'MARK':
                self.assertTrue(math.isnan(result[score]))
                # Markedness = h/(h+f) + c/(f+c) - 1 = 0/(0+0) + 1/(0+1) - 1
            elif score == 'PT':
                self.assertTrue(math.isnan(result[score]))
                # Prevalence Threshold = (Sqrt(h/(h+m)*f/(f+c))-(f/(f+c))) / (h/(h+m)-f/(f+c)) = (Sqrt(0/(0+0)*0/(0+1))-(0/(0+1))) / (0/(0+0)-0/(0+1))
            elif score == 'BA':
                self.assertTrue(math.isnan(result[score]))
                # Balanced Accuracy = (1/2)*(h/(h+m)+c/(f+c) = 1/2 * (0/(0+0)+ 1/(0+1))
            elif score == 'FM':
                self.assertTrue(math.isnan(result[score]))
                # Fowlkes-Mallows Index = Sqrt((h/(h+f))*(h/(h+m))) = sqrt((0/(0+0))*((0/(0+0)))
            elif score == 'FAER':
                self.assertTrue(math.isnan(result[score]))
                # FAER = f / (h + m) = 0 / (0 + 0)
            elif score == 'Tau':
                self.assertTrue(math.isnan(result[score]))
                # Tau = 1 - (np.sqrt((f/(c + f))**2 + (m/(h + m))**2)/np.sqrt(2)) = 1 - sqrt((0 / 1)^2 + (0/(0 + 0))^2 )
    
    
    def test_cont_mixed_table(self):
        y_true = [10, 10, 1, 1]
        y_pred = [10, 1, 10, 1]
        thresh = 10
        result = metrics.calc_contingency(y_true, y_pred, thresh)
        # contigency table 1 1
        #                  1 1
        for score in result:
            # print(score)
            if score == 'TP':
                self.assertAlmostEqual(result[score], 1)
                # 1 hits
            elif score == 'FN':
                self.assertAlmostEqual(result[score], 1)
                # 1 miss
            elif score == 'FP':
                self.assertAlmostEqual(result[score], 1)
                # 1 false alarms
            elif score == 'TN':
                self.assertAlmostEqual(result[score], 1)
                # 1 correct negatives
            elif score == 'PC':
                self.assertAlmostEqual(result[score], 1/2)
                # PC = h+c / n = 1+1 / 4
            elif score == 'B':
                self.assertAlmostEqual(result[score], 1)
                # B = h+f / h+m = 1+1 / 1+1
            elif score == 'H':
                self.assertAlmostEqual(result[score], 1/2)
                # H = h / h+m = 1 / 1+1
            elif score == 'FAR':
                self.assertAlmostEqual(result[score], 1/2)
                # FAR = f / h+f = 1 / 1+1
            elif score == 'F':
                self.assertTrue(result[score], 1/2)
                # f = f / f+c = 1 / 1+1
            elif score == 'FOH':
                self.assertTrue(result[score], 1/2)
                # FOH = h / h+f = 1 / 1+1
            elif score == 'FOM':
                self.assertTrue(result[score], 1/2)
                # FOM = m / h+m = 1 / 1+1
            elif score == 'POCN':
                self.assertAlmostEqual(result[score], 1/2)
                # POCN = c / f+c = 1 / 1+1
            elif score == 'DFR':
                self.assertTrue(result[score], 1/2)
                # DFR = m / m+c = 1 / 1+1
            elif score == 'FOCN':
                self.assertAlmostEqual(result[score], 1/2)
                # FOCN = c / m+c = 1 / 1+1
            elif score == 'TS': 
                self.assertAlmostEqual(result[score], 1/3)
                # TS = h / h+f+m = 1 / 1+1+1
            elif score == 'OR':
                self.assertAlmostEqual(result[score], 1)
                # OR = hc / fm = 1*1 / 1*1
            elif score == 'GSS':
                self.assertAlmostEqual(result[score], 0)
                # GSS = h-(h+f)*(h+m)/n / h+f+m-(h+f)*(h+m)/n = 1-(1+1)*(1+1)/4 / 1+1+1-(1+1)(1+1)/4   = 
            elif score == 'TSS':
                self.assertAlmostEqual(result[score], 0)
                # TSS = h / h+m - f / f+c = 1 / 1+1 - 1 / 1+1
            elif score == 'HSS':
                self.assertAlmostEqual(result[score], 0)
                # HSS = 2.0*(h*c-f*m) / (h+m)*(m+c)+(h+f)*(f+c) = 2*(1*1 - 1*1) / (1+1)(1+1)+(1+1)(1+1)
            elif score == 'ORSS':
                self.assertAlmostEqual(result[score], 0)
                # ORSS = h*c-m*f / h*c+m*f = 1*1-1*1 / 1*1+1*1
            elif score == 'SEDS':
                self.assertAlmostEqual(result[score], 0)
                # SEDS = (log((h+f)/n)+log((h+m)/n) / log(h/n)) - 1 = (log((1+1)/1)+log((1+1)/1) / log(1/1)) - 1
            elif score == 'FONE':
                self.assertAlmostEqual(result[score], 0.5)
            elif score == 'FTWO':
                self.assertAlmostEqual(result[score], 0.5)
            elif score == 'FHALF':
                self.assertAlmostEqual(result[score], 0.5)
                # F-Scores (Beta = 0.5, 1, 2) = ((1+ Beta^2)* h) / ((1+Beta^2)*h + Beta^2 * m + f)
                # F(Beta = 0.5) = ((1+0.5^2)*1) / ((1+0.5^2)*1 + 1 + 1)
                # F(Beta = 1) = ((1+1^2)*1) / ((1+1^2)*1 + 1 + 1)
                # F(Beta = 2) = ((1+2^2)*1) / ((1+2^2)*1 + 1 + 1)
            elif score == 'PREV':
                self.assertAlmostEqual(result[score], 1/2)
                # Prevalence = (h+m)/n = (1+1)/4
            elif score == 'MCC':
                self.assertAlmostEqual(result[score], 0)
                # Matthew Correlation Coefficient = (h*c-f*m)/Sqrt((h+f)*(h+m)*(c+f)*(f+m)) = (1*1-1*1)/sqrt((1+1)*(1+1)*(1+1)*(1+1))
            elif score == 'INFORM':
                self.assertAlmostEqual(result[score], 0)
                # Informedness = h/(h+m) + c/(f+c) - 1 = 1/(1+1) + 1/(1+1) - 1
            elif score == 'MARK':
                self.assertAlmostEqual(result[score], 0)
                # Markedness = h/(h+f) + c/(f+c) - 1 = 1/(1+1) + 1/(1+1) - 1
            elif score == 'PT':
                self.assertTrue(math.isnan(result[score]))
                # Prevalence Threshold = (Sqrt(h/(h+m)*f/(f+c))-(f/(f+c))) / (h/(h+m)-f/(f+c)) = (Sqrt(1/(1+1)*1/(1+1))-(1/(1+1))) / (1/(1+1)-1/(1+1))
            elif score == 'BA':
                self.assertAlmostEqual(result[score], 1/2)
                # Balanced Accuracy = (1/2)*(h/(h+m)+c/(f+c) = 1/2 * (1/(1+1)+ 1/(1+1))
            elif score == 'FM':
                self.assertAlmostEqual(result[score],1/2)
                # Fowlkes-Mallows Index = Sqrt((h/(h+f))*(h/(h+m))) = sqrt((1/(1+1))*((1/(1+1)))
            elif score == 'FAER':
                self.assertAlmostEqual(result[score],1/2)
                # FAER = f / (h + m) = 1 / (1 + 1)
            elif score == 'Tau':
                self.assertAlmostEqual(result[score],1/2)
                # Tau = 1 - (np.sqrt((f/(c + f))**2 + (m/(h + m))**2)/np.sqrt(2)) = 1 - sqrt((1 / (1+1))^2 + (1/(1 + 1))^2 )
            



    def test_cont_complex_table(self):
        y_true = [10, 10, 10, 1, 1, 1, 1, 1, 1, 1]
        y_pred = [10, 1, 1, 10, 10, 10, 1, 1, 1, 1]
        thresh = 10
        result = metrics.calc_contingency(y_true, y_pred, thresh)
        # builds contigency table 1 3 
        #                         2 4
        for score in result:
            # print(score)
            if score == 'TP':
                self.assertAlmostEqual(result[score], 1)
                # 1 hits
            elif score == 'FN':
                self.assertAlmostEqual(result[score], 2)
                # 2 miss
            elif score == 'FP':
                self.assertAlmostEqual(result[score], 3)
                # 3 false alarms
            elif score == 'TN':
                self.assertAlmostEqual(result[score], 4)
                # 4 correct negatives
            elif score == 'PC':
                self.assertAlmostEqual(result[score], 1/2)
                # PC = h+c / n = 1+4 / 10
            elif score == 'B':
                self.assertAlmostEqual(result[score], 4/3)
                # B = h+f / h+m = 1+3 / 1+2
            elif score == 'H':
                self.assertAlmostEqual(result[score], 1/3)
                # H = h / h+m = 1 / 1+2
            elif score == 'FAR':
                self.assertAlmostEqual(result[score], 3/4)
                # FAR = f / h+f = 3 / 1+3
            elif score == 'F':
                self.assertTrue(result[score], 3/7)
                # f = f / f+c = 3 / 3+4
            elif score == 'FOH':
                self.assertTrue(result[score], 1/4)
                # FOH = h / h+f = 1 / 1+3
            elif score == 'FOM':
                self.assertTrue(result[score], 2/3)
                # FOM = m / h+m = 2 / 1+2
            elif score == 'POCN':
                self.assertAlmostEqual(result[score], 4/7)
                # POCN = c / f+c = 4 / 4+3
            elif score == 'DFR':
                self.assertTrue(result[score], 1/3)
                # DFR = m / m+c = 2 / 2+4
            elif score == 'FOCN':
                self.assertAlmostEqual(result[score], 2/3)
                # FOCN = c / m+c = 4 / 2+4
            elif score == 'TS': 
                self.assertAlmostEqual(result[score], 1/6)
                # TS = h / h+f+m = 1 / 1+2+3
            elif score == 'OR':
                self.assertAlmostEqual(result[score], 2/3)
                # OR = hc / fm = 1*4 / 2*3
            elif score == 'GSS':
                self.assertAlmostEqual(result[score], -1/24)
                # GSS = h-(h+f)*(h+m)/n / h+f+m-(h+f)*(h+m)/n = 1-(1+3)*(1+2)/10 / 1+3+2-(1+3)(1+2)/10    
            elif score == 'TSS':
                self.assertAlmostEqual(result[score], -2/21)
                # TSS = h / h+m - f / f+c = 1 / 1+2 - 3 / 3+4
            elif score == 'HSS':
                self.assertAlmostEqual(result[score], -2/23)
                # HSS = 2.0*(h*c-f*m) / (h+m)*(m+c)+(h+f)*(f+c) = 2*(1*4 - 2*3) / (1+2)(2+4)+(1+3)(3+4)
            elif score == 'ORSS':
                self.assertAlmostEqual(result[score], -1/5)
                # ORSS = h*c-m*f / h*c+m*f = 1*4-2*3 / 1*4+2*3
            elif score == 'SEDS':
                self.assertAlmostEqual(result[score], ((np.log(4/10)+np.log(3/10))/np.log(1/10)) - 1)
                # SEDS = (log((h+f)/n)+log((h+m)/n) / log(h/n)) - 1 = (log((1+3)/10)+log((1+2)/10) / log(1/10)) - 1
            elif score == 'FONE':
                self.assertAlmostEqual(result[score], ((1+1**2)*1) / ((1+1**2)*1 + 2*(1**2) + 3))
            elif score == 'FTWO':
                self.assertAlmostEqual(result[score], ((1+2**2)*1) / ((1+2**2)*1 + 2*(2**2) + 3))         
            elif score == 'FHALF':
                self.assertAlmostEqual(result[score], ((1+0.5**2)*1) / ((1+0.5**2)*1 + 2*(0.5**2) + 3))   
                # F-Scores (Beta = 0.5, 1, 2) = ((1+ Beta^2)* h) / ((1+Beta^2)*h + Beta^2 * m + f)
                # F(Beta = 0.5) = ((1+0.5^2)*1) / ((1+0.5^2)*1 + 2 + 3)
                # F(Beta = 1) = ((1+1^2)*1) / ((1+1^2)*1 + 2 + 3)
                # F(Beta = 2) = ((1+2^2)*1) / ((1+2^2)*1 + 2 + 3)
            elif score == 'PREV':
                self.assertAlmostEqual(result[score], 3/10)
                # Prevalence = (h+m)/n = (1+2)/10
            elif score == 'MCC':
                self.assertAlmostEqual(result[score], -2/np.sqrt(504))
                # Matthew Correlation Coefficient = (h*c-f*m)/Sqrt((h+f)*(h+m)*(c+f)*(c+m)) = (1*4-3*2)/sqrt((1+3)*(1+2)*(4+3)*(4+2))
            elif score == 'INFORM':
                self.assertAlmostEqual(result[score], -2/21)
                # Informedness = h/(h+m) + c/(f+c) - 1 = 1/(1+2) + 4/(3+4) - 1= 1/3 + 4/7 - 1 = 7/21 + 12/21 - 1 = 19/21 - 1
            elif score == 'MARK':
                self.assertAlmostEqual(result[score], -2/24)
                # Markedness = h/(h+f) + c/(m+c) - 1 = 1/(1+3) + 4/(2+4) - 1 = 1/4 + 4/6 - 1 = 6/24 + 16/24 - 1 = 22/24 - 1
            elif score == 'PT':
                self.assertAlmostEqual(result[score], (np.sqrt(3/21)- 3/7)/(-2/21))
                # Prevalence Threshold = (Sqrt(h/(h+m)*f/(f+c))-(f/(f+c))) / (h/(h+m)-f/(f+c)) = (Sqrt(1/(1+2)*3/(3+4))-(3/(3+4))) / (1/(1+2)-3/(3+4))
            elif score == 'BA':
                self.assertAlmostEqual(result[score], 19/42)
                # Balanced Accuracy = (1/2)*(h/(h+m)+c/(f+c) = 1/2 * (1/(1+2)+ 4/(3+4))
            elif score == 'FM':
                self.assertAlmostEqual(result[score], np.sqrt(1/12))
                # Fowlkes-Mallows Index = Sqrt((h/(h+f))*(h/(h+m))) = sqrt((1/(1+3))*((1/(1+2)))
            elif score == 'FAER':
                self.assertAlmostEqual(result[score],1)
                # FAER = f / (h + m) = 3 / (1 + 2)
            elif score == 'Tau':
                self.assertAlmostEqual(result[score],0.43959036686000863)
                # Tau = 1 - (np.sqrt((f/(c + f))**2 + (m/(h + m))**2)/np.sqrt(2)) = 1 - sqrt((3 / (4+3))^2 + (2/(1 + 2))^2 )
            
             
    
    def test_cont_garbage(self):
        test_GSS_garbage = metrics.check_GSS(0,0,0,0)
        self.assertTrue(math.isnan(test_GSS_garbage))
        
        y_true = ['String']
        y_pred = ['String']
        thresh = 10
        with self.assertRaises(ValueError):
            result = metrics.calc_contingency(y_true, y_pred, thresh)
       
       
class TimeMetricsTestCase(unittest.TestCase):
    def test_time_mean_error(self):
        y_true = [datetime.datetime.utcnow()]
        y_pred = [datetime.datetime.utcnow() - datetime.timedelta(days=1)]
        result = metrics.calc_E(y_true, y_pred)
        hand_calc =  y_pred[0] - y_true[0]
        self.assertAlmostEqual(result[0], hand_calc)


        y_true = [datetime.datetime.utcnow(), datetime.datetime.utcnow(), datetime.datetime.utcnow(), datetime.datetime.utcnow()]
        y_pred = [datetime.datetime.utcnow() + datetime.timedelta(days=1), datetime.datetime.utcnow() - datetime.timedelta(days=1), datetime.datetime.utcnow() + datetime.timedelta(days=1), datetime.datetime.utcnow() - datetime.timedelta(days=1)]
        zipped = zip(y_pred, y_true)
        temp = []
        for elements in zipped:
            temp.append(elements[0] - elements[1])
        hand_calc = temp
        result = metrics.calc_E(y_true, y_pred)
        result_list = [x.days for x in result]
        hand_calc_list = [x.days for x in hand_calc]
        self.assertAlmostEqual(result_list, hand_calc_list)

        y_true = [datetime.datetime.utcnow(), datetime.datetime.utcnow(), datetime.datetime.utcnow(), datetime.datetime.utcnow()]
        y_pred = [datetime.datetime.utcnow() - datetime.timedelta(days=1), datetime.datetime.utcnow() - datetime.timedelta(days=1), datetime.datetime.utcnow() - datetime.timedelta(days=1), datetime.datetime.utcnow() - datetime.timedelta(days=1)]
        zipped = zip(y_pred, y_true)
        result = np.mean(metrics.calc_E(y_true, y_pred))
        self.assertTrue(result < datetime.timedelta(days = 0.0))
        # Testing the sign

        y_true = [datetime.datetime.utcnow(), datetime.datetime.utcnow(), datetime.datetime.utcnow(), datetime.datetime.utcnow()]
        y_pred = [datetime.datetime.utcnow() + datetime.timedelta(days=1), datetime.datetime.utcnow() + datetime.timedelta(days=1), datetime.datetime.utcnow() + datetime.timedelta(days=1), datetime.datetime.utcnow() + datetime.timedelta(days=1)]
        zipped = zip(y_pred, y_true)
        result = np.mean(metrics.calc_E(y_true, y_pred))
        self.assertTrue(result > datetime.timedelta(days = 0.0))
        # Testing the sign


    def test_time_median_error(self):
        y_true = [datetime.datetime.utcnow()]
        y_pred = [datetime.datetime.utcnow() - datetime.timedelta(days=1)]
        result = metrics.calc_E(y_true, y_pred)
        hand_calc =  y_pred[0] - y_true[0]
        self.assertAlmostEqual(result[0], hand_calc)


        y_true = [datetime.datetime.utcnow(), datetime.datetime.utcnow(), datetime.datetime.utcnow(), datetime.datetime.utcnow()]
        y_pred = [datetime.datetime.utcnow() + datetime.timedelta(days=1), datetime.datetime.utcnow() - datetime.timedelta(days=1), datetime.datetime.utcnow() + datetime.timedelta(days=1), datetime.datetime.utcnow() - datetime.timedelta(days=1)]
        zipped = zip(y_pred, y_true)
        temp = []
        for elements in zipped:
            temp.append(elements[0] - elements[1])
        hand_calc = np.median(temp)
        result = np.median(metrics.calc_E(y_true, y_pred))
        self.assertAlmostEqual(result, hand_calc)

        y_true = [datetime.datetime.utcnow(), datetime.datetime.utcnow(), datetime.datetime.utcnow(), datetime.datetime.utcnow()]
        y_pred = [datetime.datetime.utcnow() - datetime.timedelta(days=1), datetime.datetime.utcnow() - datetime.timedelta(days=1), datetime.datetime.utcnow() - datetime.timedelta(days=1), datetime.datetime.utcnow() - datetime.timedelta(days=1)]
        zipped = zip(y_pred, y_true)
        result = np.median(metrics.calc_E(y_true, y_pred))
        self.assertTrue(result < datetime.timedelta(days = 0.0))

        y_true = [datetime.datetime.utcnow(), datetime.datetime.utcnow(), datetime.datetime.utcnow(), datetime.datetime.utcnow()]
        y_pred = [datetime.datetime.utcnow() + datetime.timedelta(days=1), datetime.datetime.utcnow() + datetime.timedelta(days=1), datetime.datetime.utcnow() + datetime.timedelta(days=1), datetime.datetime.utcnow() + datetime.timedelta(days=1)]
        zipped = zip(y_pred, y_true)
        result = np.median(metrics.calc_E(y_true, y_pred))
        self.assertTrue(result > datetime.timedelta(0.0))


    def test_time_mean_absolue_error(self):
        y_true = [datetime.datetime.utcnow()]
        y_pred = [datetime.datetime.utcnow() - datetime.timedelta(days=1)]
        result = metrics.calc_AE(y_true, y_pred)
        hand_calc =  np.abs(y_pred[0] - y_true[0])
        self.assertAlmostEqual(result[0], hand_calc)


        y_true = [datetime.datetime.utcnow(), datetime.datetime.utcnow(), datetime.datetime.utcnow(), datetime.datetime.utcnow()]
        y_pred = [datetime.datetime.utcnow() + datetime.timedelta(days=1), datetime.datetime.utcnow() - datetime.timedelta(days=1), datetime.datetime.utcnow() + datetime.timedelta(days=1), datetime.datetime.utcnow() - datetime.timedelta(days=1)]
        zipped = zip(y_pred, y_true)
        temp = []
        for elements in zipped:
            temp.append(np.abs(elements[0] - elements[1]))
        hand_calc = temp
        result = metrics.calc_AE(y_true, y_pred)
        result_list = [x.days for x in result]
        hand_calc_list = [x.days for x in hand_calc]
        self.assertAlmostEqual(result_list, hand_calc_list)
      
      

        y_true = [datetime.datetime.utcnow(), datetime.datetime.utcnow(), datetime.datetime.utcnow(), datetime.datetime.utcnow()]
        y_pred = [datetime.datetime.utcnow() - datetime.timedelta(days=1), datetime.datetime.utcnow() - datetime.timedelta(days=1), datetime.datetime.utcnow() - datetime.timedelta(days=1), datetime.datetime.utcnow() - datetime.timedelta(days=1)]
        zipped = zip(y_pred, y_true)
        result = np.mean(metrics.calc_AE(y_true, y_pred))
        self.assertTrue(result > datetime.timedelta(0.0))
        # Testing sign of result

        y_true = [datetime.datetime.utcnow(), datetime.datetime.utcnow(), datetime.datetime.utcnow(), datetime.datetime.utcnow()]
        y_pred = [datetime.datetime.utcnow() + datetime.timedelta(days=1), datetime.datetime.utcnow() + datetime.timedelta(days=1), datetime.datetime.utcnow() + datetime.timedelta(days=1), datetime.datetime.utcnow() + datetime.timedelta(days=1)]
        zipped = zip(y_pred, y_true)
        result = np.mean(metrics.calc_AE(y_true, y_pred))
        self.assertTrue(result > datetime.timedelta(0.0))
        # Testing sign of result


    def test_time_median_absolute_error(self):
        y_true = [datetime.datetime.utcnow()]
        y_pred = [datetime.datetime.utcnow() - datetime.timedelta(days=1)]
        result = metrics.calc_AE(y_true, y_pred)
        hand_calc =  np.abs(y_pred[0] - y_true[0])
        self.assertAlmostEqual(result[0], hand_calc)


        y_true = [datetime.datetime.utcnow(), datetime.datetime.utcnow(), datetime.datetime.utcnow(), datetime.datetime.utcnow()]
        y_pred = [datetime.datetime.utcnow() + datetime.timedelta(days=1), datetime.datetime.utcnow() - datetime.timedelta(days=1), datetime.datetime.utcnow() + datetime.timedelta(days=1), datetime.datetime.utcnow() - datetime.timedelta(days=1)]
        zipped = zip(y_pred, y_true)
        temp = []
        for elements in zipped:
            temp.append(np.abs(elements[0] - elements[1]))
        hand_calc = np.median(temp)
        result = np.median(metrics.calc_AE(y_true, y_pred))
        self.assertAlmostEqual(result, hand_calc)

        y_true = [datetime.datetime.utcnow(), datetime.datetime.utcnow(), datetime.datetime.utcnow(), datetime.datetime.utcnow()]
        y_pred = [datetime.datetime.utcnow() - datetime.timedelta(days=1), datetime.datetime.utcnow() - datetime.timedelta(days=1), datetime.datetime.utcnow() - datetime.timedelta(days=1), datetime.datetime.utcnow() - datetime.timedelta(days=1)]
        zipped = zip(y_pred, y_true)
        result = np.median(metrics.calc_AE(y_true, y_pred))
        self.assertTrue(result > datetime.timedelta(days = 0.0))

        y_true = [datetime.datetime.utcnow(), datetime.datetime.utcnow(), datetime.datetime.utcnow(), datetime.datetime.utcnow()]
        y_pred = [datetime.datetime.utcnow() + datetime.timedelta(days=1), datetime.datetime.utcnow() + datetime.timedelta(days=1), datetime.datetime.utcnow() + datetime.timedelta(days=1), datetime.datetime.utcnow() + datetime.timedelta(days=1)]
        zipped = zip(y_pred, y_true)
        result = np.median(metrics.calc_AE(y_true, y_pred))
        self.assertTrue(result > datetime.timedelta(0.0))


class MiscTestCases(unittest.TestCase):
    def test_calc_mean(self):
        test_array = [0, 1, 2, 3]
        result = metrics.calc_mean(test_array)
        self.assertAlmostEqual(result, 3/2)

        test_array = 0
        result = metrics.calc_mean(test_array)
        self.assertAlmostEqual(result, 0)

        test_array = -1
        result = metrics.calc_mean(test_array)
        self.assertAlmostEqual(result, -1)

    def test_arr_to_df_fails(self):
        arr = [[1]]
        keys = ['key_1', 'key_2']
        with self.assertRaises(SystemExit):
            result = metrics.arr_to_df(arr, keys)
        # sys.exit("metrics.py: arr_to_df: input arrays must be the same length. arr (column values) and keys (column names) must match.") 
    
    def test_check_div_outputs(self):
        n = 1
        d = 1
        result = metrics.check_div(n, d)
        self.assertAlmostEqual(result, 1)

        n = 0
        d = 1
        result = metrics.check_div(n, d)
        self.assertAlmostEqual(result, 0)

        n = 1
        d = 0
        result = metrics.check_div(n, d)
        self.assertAlmostEqual(result, np.inf)

        n = -1
        d = 0
        result = metrics.check_div(n, d)
        self.assertAlmostEqual(result, -np.inf)

        n = 0
        d = 0
        result = metrics.check_div(n, d)
        self.assertTrue(math.isnan(result))

    def test_remove_none(self):
        arr_1 = [0]
        arr_2 = [0]
        result_1, result_2 = metrics.remove_none(arr_1, arr_2)
        self.assertAlmostEqual(arr_1, result_1)
        self.assertAlmostEqual(arr_2, result_2)

        arr_1 = []
        arr_2 = [0]
        with self.assertRaises(SystemExit):
            result = metrics.remove_none(arr_1, arr_2)

        arr_1 = [None, 0]
        arr_2 = [0, 0]
        result_1, result_2 = metrics.remove_none(arr_1, arr_2)
        self.assertAlmostEqual(result_1, [0])
        self.assertAlmostEqual(result_2, [0])

        arr_1 = [0, 0]
        arr_2 = [None, 0]
        result_1, result_2 = metrics.remove_none(arr_1, arr_2)
        self.assertAlmostEqual(result_1, [0])
        self.assertAlmostEqual(result_2, [0])

        arr_1 = [0, None]
        arr_2 = [None, 0]
        result_1, result_2 = metrics.remove_none(arr_1, arr_2)
        self.assertAlmostEqual(result_1, [])
        self.assertAlmostEqual(result_2, [])

    def test_remove_zero(self):
        arr_1 = [1]
        arr_2 = [1]
        result_1, result_2 = metrics.remove_zero(arr_1, arr_2)
        self.assertAlmostEqual(arr_1, result_1)
        self.assertAlmostEqual(arr_2, result_2)

        arr_1 = []
        arr_2 = [1]
        with self.assertRaises(SystemExit):
            result = metrics.remove_zero(arr_1, arr_2)

        arr_1 = [0., 1]
        arr_2 = [1, 1]
        result_1, result_2 = metrics.remove_zero(arr_1, arr_2)
        self.assertAlmostEqual(result_1, [1])
        self.assertAlmostEqual(result_2, [1])

        arr_1 = [1, 1]
        arr_2 = [0., 1]
        result_1, result_2 = metrics.remove_zero(arr_1, arr_2)
        self.assertAlmostEqual(result_1, [1])
        self.assertAlmostEqual(result_2, [1])

        arr_1 = [1, 0.]
        arr_2 = [0., 1]
        result_1, result_2 = metrics.remove_zero(arr_1, arr_2)
        self.assertAlmostEqual(result_1, [])
        self.assertAlmostEqual(result_2, [])
