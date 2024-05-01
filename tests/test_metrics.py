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

'''
unittest framework - built in python function
directory called tests - make discoverable
test a few contigency tables - a couple per metric
~100 small tests
find the documentation (https://docs.python.org/3/library/unittest.html)
'''

class FluxMetricsTestCase(unittest.TestCase):
     
    # Error ###############################################################################################
    def test_flux_metric_error_calcs(self):        
        y_true = [10]
        y_pred = [1]
        result = metrics.calc_E(y_true, y_pred)
        hand_calc =  y_pred[0] - y_true[0]
        self.assertEqual(result[0], hand_calc)


        y_true = [10, 10, 10, 10]
        y_pred = [11, 1, 11, 1]
        zipped = zip(y_pred, y_true)
        temp = []
        for elements in zipped:
            temp.append(elements[0] - elements[1])
        hand_calc = np.mean(temp)
        result = np.mean(metrics.calc_E(y_true, y_pred))
        self.assertEqual(result, hand_calc)

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
        self.assertEqual(result[0], hand_calc)


        y_true = [10, 10, 10, 10]
        y_pred = [11, 1, 11, 1]
        zipped = zip(y_pred, y_true)
        temp = []
        for elements in zipped:
            temp.append(np.abs(elements[0] - elements[1]))
        hand_calc = np.mean(temp)
        result = np.mean(metrics.calc_AE(y_true, y_pred))
        self.assertEqual(result, hand_calc)

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
        self.assertEqual(result[0], -1)


        y_true = [10, 10, 10, 10]
        y_pred = [11, 1, 11, 1]
        zipped = zip(y_pred, y_true)
        temp = []
        for elements in zipped:
            temp.append(np.log10(elements[0]) - np.log10(elements[1]))
        hand_calc = np.mean(temp)
        result = np.mean(metrics.calc_LE(y_true, y_pred))
        self.assertEqual(result, hand_calc)

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


#     # # Abs Log Error ################################################################################
    def test_flux_metric_absolute_log_error_calcs(self):
        y_true = [10]
        y_pred = [1]
        result = metrics.calc_ALE(y_true, y_pred)
        hand_calc = np.abs(np.log10(y_pred[0]) - np.log10(y_true[0]))
        self.assertEqual(result[0], hand_calc)


        y_true = [10, 10, 10, 10]
        y_pred = [11, 1, 11, 1]
        zipped = zip(y_pred, y_true)
        temp = []
        for elements in zipped:
            temp.append(np.abs(np.log10(elements[0]) - np.log10(elements[1])))
        hand_calc = np.mean(temp)
        result = np.mean(metrics.calc_ALE(y_true, y_pred))
        self.assertEqual(result, hand_calc)

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



#     # # Squared Error ################################################################################
    def test_flux_metric_squared_error_calcs(self):
        y_true = [10]
        y_pred = [1]
        result = metrics.calc_SE(y_true, y_pred)
        hand_calc = (y_pred[0] - y_true[0])**2
        self.assertEqual(result[0], hand_calc)


        y_true = [10, 10, 10, 10]
        y_pred = [11, 1, 11, 1]
        zipped = zip(y_pred, y_true)
        temp = []
        for elements in zipped:
            temp.append((elements[0] - elements[1])**2)
        hand_calc = np.mean(temp)
        result = np.mean(metrics.calc_SE(y_true, y_pred))
        self.assertEqual(result, hand_calc)

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
        self.assertEqual(result[0], hand_calc)


        y_true = [10, 10, 10, 10]
        y_pred = [11, 1, 11, 1]
        zipped = zip(y_pred, y_true)
        temp = []
        for elements in zipped:
            temp.append((np.log10(elements[0]) - np.log10(elements[1]))**2)
        hand_calc = np.mean(temp)
        result = np.mean(metrics.calc_SLE(y_true, y_pred))
        self.assertEqual(result, hand_calc)

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
    
    
    
#     # # Root Mean Squared Error ################################################################################
    def test_flux_metric_root_mean_squared_error_calcs(self):
        y_true = [10]
        y_pred = [1]
        result = metrics.calc_RMSE(y_true, y_pred)
        hand_calc = math.sqrt((y_pred[0] - y_true[0])**2)
        # hand_calc = math.sqrt(sum(error)/len(error))
        
        self.assertEqual(result, hand_calc)


        y_true = [10, 10, 10, 10]
        y_pred = [11, 1, 11, 1]
        zipped = zip(y_pred, y_true)
        temp = []
        for elements in zipped:
            temp.append((elements[0] - elements[1])**2)
        hand_calc = math.sqrt(sum(temp)/len(temp))
        result = np.mean(metrics.calc_RMSE(y_true, y_pred))
        self.assertEqual(result, hand_calc)

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
        self.assertEqual(result, hand_calc)


        y_true = [10, 10, 10, 10]
        y_pred = [11, 1, 11, 1]
        zipped = zip(y_pred, y_true)
        temp = []
        for elements in zipped:
            temp.append((np.log10(elements[0]) - np.log10(elements[1]))**2)
        hand_calc = math.sqrt(sum(temp)/len(temp))
        result = np.mean(metrics.calc_RMSLE(y_true, y_pred))
        self.assertEqual(result, hand_calc)

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
  

    

#     # # Percent Error ################################################################################
    def test_flux_metric_percent_error_calcs(self):
        y_true = [10]
        y_pred = [1]
        result = metrics.calc_PE(y_true, y_pred)
        hand_calc = (y_pred[0] - y_true[0]) / y_true[0]
        self.assertEqual(result[0], hand_calc)


        y_true = [10, 10, 10, 10]
        y_pred = [11, 1, 11, 1]
        zipped = zip(y_pred, y_true)
        temp = []
        for elements in zipped:
            temp.append((elements[0] - elements[1]) / elements[1])
        hand_calc = np.mean(temp)
        result = np.mean(metrics.calc_PE(y_true, y_pred))
        self.assertEqual(result, hand_calc)

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

    

#     # # Abs Percent Error ################################################################################
    def test_flux_metric_absolute_percent_error_calcs(self):
        y_true = [10]
        y_pred = [1]
        result = metrics.calc_APE(y_true, y_pred)
        hand_calc = np.abs(y_pred[0] - y_true[0]) / y_true[0]
        self.assertEqual(result[0], hand_calc)


        y_true = [10, 10, 10, 10]
        y_pred = [11, 1, 11, 1]
        zipped = zip(y_pred, y_true)
        temp = []
        for elements in zipped:
            temp.append(np.abs(elements[0] - elements[1]) / elements[1])
        hand_calc = np.mean(temp)
        result = np.mean(metrics.calc_APE(y_true, y_pred))
        self.assertEqual(result, hand_calc)

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



#     # # Symmetric Percent Error ################################################################################
    def test_flux_metric_symmetric_percent_error_calcs(self):
        y_true = [10]
        y_pred = [1]
        result = metrics.calc_SPE(y_true, y_pred)
        hand_calc =  2.0 * (y_pred[0] - y_true[0]) / (y_pred[0] + y_true[0])
        self.assertEqual(result[0], hand_calc)


        y_true = [10, 10, 10, 10]
        y_pred = [11, 1, 11, 1]
        zipped = zip(y_pred, y_true)
        temp = []
        for elements in zipped:
            temp.append(2.0* (elements[0] - elements[1]) / (elements[0] + elements[1]))
        hand_calc = np.mean(temp)
        result = np.mean(metrics.calc_SPE(y_true, y_pred))
        self.assertEqual(result, hand_calc)

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
    



#     # # Symmetric Abs Percent Error ################################################################################
    def test_flux_metric_symmetric_absolute_percent_error_calcs(self):
        y_true = [10]
        y_pred = [1]
        result = metrics.calc_SAPE(y_true, y_pred)
        hand_calc =  2.0 * np.abs(y_pred[0] - y_true[0]) / (y_pred[0] + y_true[0])
        self.assertEqual(result[0], hand_calc)


        y_true = [10, 10, 10, 10]
        y_pred = [11, 1, 11, 1]
        zipped = zip(y_pred, y_true)
        temp = []
        for elements in zipped:
            temp.append(2.0* np.abs(elements[0] - elements[1]) / (elements[0] + elements[1]))
        hand_calc = np.mean(temp)
        result = np.mean(metrics.calc_SAPE(y_true, y_pred))
        self.assertEqual(result, hand_calc)

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
        self.assertEqual(result[0], 0)

        y_true = [10, 11, 12, 13]
        y_pred = [1, 2, 3, 4]
        zipped = zip(y_pred, y_true)
        result = metrics.calc_pearson(y_true, y_pred)
        self.assertEqual(result[0], 1)

        y_true = [10, 11, 12, 13]
        y_pred = [4, 3, 2, 1]
        result = metrics.calc_pearson(y_true, y_pred)
        self.assertEqual(result[0], -1.0)

    
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
        self.assertEqual(result[1], 0)

        y_true = [1, 10, 100, 1000]
        y_pred = [1, 10, 100, 1000]
        zipped = zip(y_pred, y_true)
        result = metrics.calc_pearson(y_true, y_pred)
        self.assertEqual(result[1], 1)
       

        y_true = [1, 10, 100, 1000]
        y_pred = [1000, 100, 10, 1]
        result = metrics.calc_pearson(y_true, y_pred)
        self.assertEqual(result[1], -1.0)




#     # # Mean Accuracy Ratio ################################################################################
    def test_flux_metric_mean_acc_ratio_calcs(self):
        y_true = [10]
        y_pred = [1]
        result = metrics.calc_MAR(y_true, y_pred)
        hand_calc = y_pred[0] / y_true[0]
        self.assertEqual(result, hand_calc)
        

        y_true = [10, 1, 1, 10]
        y_pred = [11, 1, 11, 1]
        zipped = zip(y_pred, y_true)
        temp = []
        for elements in zipped:
            temp.append(elements[0] / elements[1])
        hand_calc = np.mean(temp)
        result = metrics.calc_MAR(y_true, y_pred)
        self.assertEqual(result, hand_calc)

        y_true = [10, 11, 12, 13]
        y_pred = [0, 1, 2, 3]
        zipped = zip(y_pred, y_true)
        temp = []
        for elements in zipped:
            temp.append(elements[0] / elements[1])
        hand_calc = np.mean(temp)
        result = metrics.calc_MAR(y_true, y_pred)
        self.assertEqual(result, hand_calc)

        y_true = [10, 11, 12, 13]
        y_pred = [3, 2, 1, 0]
        zipped = zip(y_pred, y_true)
        temp = []
        for elements in zipped:
            temp.append(elements[0] / elements[1])
        hand_calc = np.mean(temp)
        result = metrics.calc_MAR(y_true, y_pred)
        self.assertEqual(result, hand_calc)
    


#      # # Median Symmetric Accuracy ################################################################################
    def test_flux_metric_med_symm_acc_calcs(self):
        y_true = [10]
        y_pred = [1]
        result = metrics.calc_MdSA(y_true, y_pred)
        hand_calc = (np.exp(np.median(np.abs(np.log(y_true[0] / y_pred[0])))) - 1.0)
        self.assertEqual(result, hand_calc)
        

        y_true = np.array([10, 1, 1, 10])
        y_pred = np.array([11, 1, 11, 1])
        zipped = zip(y_pred, y_true)
        temp = []
        hand_calc = (np.exp(np.median(np.abs(np.log(y_true/ y_pred)))) - 1.0)
        result = metrics.calc_MdSA(y_true, y_pred)
        self.assertEqual(result, hand_calc)

        y_true = np.array([10, 11, 12, 13])
        y_pred = np.array([1, 2, 3, 4])
        hand_calc = (np.exp(np.median(np.abs(np.log(y_true/ y_pred)))) - 1.0)
        result = metrics.calc_MdSA(y_true, y_pred)
        self.assertEqual(result, hand_calc)

        y_true = np.array([10, 11, 12, 13])
        y_pred = np.array([4, 3, 2, 1])
        hand_calc = (np.exp(np.median(np.abs(np.log(y_true/ y_pred)))) - 1.0)
        result = metrics.calc_MdSA(y_true, y_pred)
        self.assertEqual(result, hand_calc)




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
        self.assertEqual(result, 0)




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
                self.assertEqual(np.mean(func_call), hand_calc)
            elif metric == 'AE':
                hand_calc = np.mean(np.abs(y_pred - y_true))
                self.assertEqual(np.mean(func_call), hand_calc)
            elif metric == 'LE':
                hand_calc = np.mean(np.log10(y_pred) - np.log10(y_true))
                self.assertEqual(np.mean(func_call), hand_calc)
            elif metric == 'ALE':
                hand_calc = np.mean(np.abs(np.log10(y_pred) - np.log10(y_true)))
                self.assertEqual(np.mean(func_call), hand_calc)
            elif metric == 'SE':
                hand_calc = np.mean((y_pred - y_true)**2)
                self.assertEqual(np.mean(func_call), hand_calc)
            elif metric == 'SLE':
                hand_calc = np.mean((np.log10(y_pred) - np.log10(y_true))**2)
                self.assertEqual(np.mean(func_call), hand_calc)
            elif metric == 'RMSE':
                error = (y_pred - y_true)**2
                hand_calc = math.sqrt(sum(error)/ len(error))
                self.assertEqual(np.mean(func_call), hand_calc)
            elif metric == 'RMSLE':
                error = (np.log10(y_pred) - np.log10(y_true))**2
                hand_calc = math.sqrt(sum(error)/len(error))
                self.assertEqual(np.mean(func_call), hand_calc)
            elif metric == 'PE':
                hand_calc = np.mean((y_pred - y_true) / y_true)
                self.assertEqual(np.mean(func_call), hand_calc)
            elif metric == 'APE':
                hand_calc = np.mean(np.abs(y_pred - y_true) / y_true)
                self.assertEqual(np.mean(func_call), hand_calc)
            elif metric == 'SPE':
                hand_calc = np.mean(2.0 * (y_pred - y_true) / (y_pred + y_true))
                self.assertEqual(np.mean(func_call), hand_calc)
            elif metric == 'SAPE':
                hand_calc = np.mean(2.0 * np.abs(y_pred - y_true) / (y_pred + y_true))
                self.assertEqual(np.mean(func_call), hand_calc)
            elif metric == 'r_lin':
                self.assertEqual(func_call, 0)
            elif metric == 'r_log':
                self.assertEqual(func_call, 0)
            elif metric == 'MAR':
                hand_calc = np.mean(y_pred / y_true)
                self.assertEqual(func_call, hand_calc)
            elif metric == 'MdSA':
                hand_calc = (np.exp(np.median(np.abs(np.log(y_true / y_pred)))) - 1.0)
                self.assertEqual(func_call, hand_calc)
            elif metric == 'spearman':
                self.assertEqual(func_call, 0)
            


class ProbabilityMetricsTestCase(unittest.TestCase):
#   # Brier Score
    def test_prob_brier(self):
        y_true = [1]
        y_pred = [0.1]
        result = metrics.calc_brier(y_true, y_pred)
        hand_calc =  (y_pred[0] - y_true[0])**2
        self.assertEqual(result, hand_calc)


        y_true = [1., 1., 0., 0.]
        y_pred = [1., 0.1, 1., 0.1]
        zipped = zip(y_pred, y_true)
        temp = []
        for elements in zipped:
            temp.append((elements[0] - elements[1])**2)
        hand_calc = np.mean(temp)
        result = np.mean(metrics.calc_brier(y_true, y_pred))
        self.assertEqual(result, hand_calc)

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
        self.assertEqual(result, hand_calc)


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
        self.assertEqual(result, hand_calc)

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
        self.assertEqual(result[0], 0)

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
        self.assertEqual(result, 1)

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
                self.assertEqual(result[score], 1)
                # 1 hit
            elif score == 'FN':
                self.assertEqual(result[score], 0)
                # 0 Misses
            elif score == 'FP':
                self.assertEqual(result[score], 0)
                # 0 False Alarms
            elif score == 'TN':
                self.assertEqual(result[score], 0)
                # 0 True Negatives
            elif score == 'PC':
                self.assertEqual(result[score], 1)
                # PC = (h+c)/1 = 1+0/1
            elif score == 'B':
                self.assertEqual(result[score], 1)
                # B = h+f / h+m = 1+0/1+0
            elif score == 'H':
                self.assertEqual(result[score], 1)
                # H = h/h+m = 1 / 1+0 
            elif score == 'FAR':
                self.assertEqual(result[score], 0)
                # FAR = f / h+f = 0 / 1+0
            elif score == 'F':
                self.assertTrue(math.isnan(result[score]))
                # F = f / f+c = 0/0
            elif score == 'FOH':
                self.assertEqual(result[score], 1)
                # FOH = h / h+f = 1 / 1+0
            elif score == 'FOM':
                self.assertEqual(result[score], 0)
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
                self.assertEqual(result[score], 1)
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
                self.assertEqual(result[score], 0)
                # 0 hits
            elif score == 'FN':
                self.assertEqual(result[score], 1)
                # 1 miss
            elif score == 'FP':
                self.assertEqual(result[score], 0)
                # 0 false alarms
            elif score == 'TN':
                self.assertEqual(result[score], 0)
                # 0 correct negatives
            elif score == 'PC':
                self.assertEqual(result[score], 0)
                # PC = h+c / n = 0+0 / 1
            elif score == 'B':
                self.assertEqual(result[score], 0)
                # B = h+f / h+m = 0+0 / 0+1
            elif score == 'H':
                self.assertEqual(result[score], 0)
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
                self.assertEqual(result[score], 1)
                # FOM = m / h+m = 1 / 0+1
            elif score == 'POCN':
                self.assertTrue(math.isnan(result[score]))
                # POCN = c / f+c = 0 / 0+0
            elif score == 'DFR':
                self.assertEqual(result[score], 1)
                # DFR = m / m+c = 1 / 1+0
            elif score == 'FOCN':
                self.assertEqual(result[score], 0)
                # FOCN = c / m+c = 0 / 1+0
            elif score == 'TS': 
                self.assertEqual(result[score], 0)
                # TS = h / h+f+m = 0 / 1+0+0
            elif score == 'OR':
                self.assertTrue(math.isnan(result[score]))
                # OR = hc / fm = 0*0 / 0*1
            elif score == 'GSS':
                self.assertEqual(result[score], 0)
                # GSS = h-(h+f)*(h+m)/n / h+f+m-(h+f)*(h+m)/n = 0-(0+0)*(0+1)/1 / 0+0+1-(0+0)(0+1)/1
            elif score == 'TSS':
                self.assertTrue(math.isnan(result[score]))
                # TSS = h / h+m - f / f+c = 0 / 1+0 - 0 / 0
            elif score == 'HSS':
                self.assertEqual(result[score], 0)
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
                self.assertEqual(result[score], 0)
                # 0 hits
            elif score == 'FN':
                self.assertEqual(result[score], 0)
                # 0 miss
            elif score == 'FP':
                self.assertEqual(result[score], 1)
                # 1 false alarms
            elif score == 'TN':
                self.assertEqual(result[score], 0)
                # 0 correct negatives
            elif score == 'PC':
                self.assertEqual(result[score], 0)
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
                self.assertEqual(result[score], 0)
                # FOH = h / h+f = 0/ 0+1
            elif score == 'FOM':
                self.assertTrue(math.isnan(result[score]))
                # FOM = m / h+m = 0 / 0+0
            elif score == 'POCN':
                self.assertEqual(result[score], 0)
                # POCN = c / f+c = 0 / 1+0
            elif score == 'DFR':
                self.assertTrue(math.isnan(result[score]))
                # DFR = m / m+c = 0 / 0+0
            elif score == 'FOCN':
                self.assertTrue(math.isnan(result[score]))
                # FOCN = c / m+c = 0 / 0+0
            elif score == 'TS': 
                self.assertEqual(result[score], 0)
                # TS = h / h+f+m = 0 / 0+1+0
            elif score == 'OR':
                self.assertTrue(math.isnan(result[score]))
                # OR = hc / fm = 0*0 / 0*1
            elif score == 'GSS':
                self.assertEqual(result[score], 0)
                # GSS = h-(h+f)*(h+m)/n / h+f+m-(h+f)*(h+m)/n = 0-(0+1)*(0+0)/1 / 0+1+0-(0+1)(0+0)/1
            elif score == 'TSS':
                self.assertTrue(math.isnan(result[score]))
                # TSS = h / h+m - f / f+c = 0 / 0+0 - 1 / 1+0
            elif score == 'HSS':
                self.assertEqual(result[score], 0)
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
                self.assertEqual(result[score], 0)
                # 0 hits
            elif score == 'FN':
                self.assertEqual(result[score], 0)
                # 0 miss
            elif score == 'FP':
                self.assertEqual(result[score], 0)
                # 0 false alarms
            elif score == 'TN':
                self.assertEqual(result[score], 1)
                # 1 correct negatives
            elif score == 'PC':
                self.assertEqual(result[score], 1)
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
                self.assertEqual(result[score], 0)
                # f = f / f+c = 0 / 1+0
            elif score == 'FOH':
                self.assertTrue(math.isnan(result[score]))
                # FOH = h / h+f = 0/ 0+0
            elif score == 'FOM':
                self.assertTrue(math.isnan(result[score]))
                # FOM = m / h+m = 0 / 0+0
            elif score == 'POCN':
                self.assertEqual(result[score], 1)
                # POCN = c / f+c = 1 / 1+0
            elif score == 'DFR':
                self.assertEqual(result[score], 0)
                # DFR = m / m+c = 0 / 0+1
            elif score == 'FOCN':
                self.assertEqual(result[score], 1)
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
                self.assertEqual(result[score], 1)
                # 1 hits
            elif score == 'FN':
                self.assertEqual(result[score], 1)
                # 1 miss
            elif score == 'FP':
                self.assertEqual(result[score], 1)
                # 1 false alarms
            elif score == 'TN':
                self.assertEqual(result[score], 1)
                # 1 correct negatives
            elif score == 'PC':
                self.assertEqual(result[score], 1/2)
                # PC = h+c / n = 1+1 / 4
            elif score == 'B':
                self.assertEqual(result[score], 1)
                # B = h+f / h+m = 1+1 / 1+1
            elif score == 'H':
                self.assertEqual(result[score], 1/2)
                # H = h / h+m = 1 / 1+1
            elif score == 'FAR':
                self.assertEqual(result[score], 1/2)
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
                self.assertEqual(result[score], 1/2)
                # POCN = c / f+c = 1 / 1+1
            elif score == 'DFR':
                self.assertTrue(result[score], 1/2)
                # DFR = m / m+c = 1 / 1+1
            elif score == 'FOCN':
                self.assertEqual(result[score], 1/2)
                # FOCN = c / m+c = 1 / 1+1
            elif score == 'TS': 
                self.assertEqual(result[score], 1/3)
                # TS = h / h+f+m = 1 / 1+1+1
            elif score == 'OR':
                self.assertEqual(result[score], 1)
                # OR = hc / fm = 1*1 / 1*1
            elif score == 'GSS':
                self.assertEqual(result[score], 0)
                # GSS = h-(h+f)*(h+m)/n / h+f+m-(h+f)*(h+m)/n = 1-(1+1)*(1+1)/4 / 1+1+1-(1+1)(1+1)/4   = 
            elif score == 'TSS':
                self.assertEqual(result[score], 0)
                # TSS = h / h+m - f / f+c = 1 / 1+1 - 1 / 1+1
            elif score == 'HSS':
                self.assertEqual(result[score], 0)
                # HSS = 2.0*(h*c-f*m) / (h+m)*(m+c)+(h+f)*(f+c) = 2*(1*1 - 1*1) / (1+1)(1+1)+(1+1)(1+1)
            elif score == 'ORSS':
                self.assertEqual(result[score], 0)
                # ORSS = h*c-m*f / h*c+m*f = 1*1-1*1 / 1*1+1*1
            elif score == 'SEDS':
                self.assertEqual(result[score], 0)
                # SEDS = (log((h+f)/n)+log((h+m)/n) / log(h/n)) - 1 = (log((1+1)/1)+log((1+1)/1) / log(1/1)) - 1



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
                self.assertEqual(result[score], 1)
                # 1 hits
            elif score == 'FN':
                self.assertEqual(result[score], 2)
                # 2 miss
            elif score == 'FP':
                self.assertEqual(result[score], 3)
                # 3 false alarms
            elif score == 'TN':
                self.assertEqual(result[score], 4)
                # 4 correct negatives
            elif score == 'PC':
                self.assertEqual(result[score], 1/2)
                # PC = h+c / n = 1+4 / 10
            elif score == 'B':
                self.assertEqual(result[score], 4/3)
                # B = h+f / h+m = 1+3 / 1+2
            elif score == 'H':
                self.assertEqual(result[score], 1/3)
                # H = h / h+m = 1 / 1+2
            elif score == 'FAR':
                self.assertEqual(result[score], 3/4)
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
                self.assertEqual(result[score], 4/7)
                # POCN = c / f+c = 4 / 4+3
            elif score == 'DFR':
                self.assertTrue(result[score], 1/3)
                # DFR = m / m+c = 2 / 2+4
            elif score == 'FOCN':
                self.assertEqual(result[score], 2/3)
                # FOCN = c / m+c = 4 / 2+4
            elif score == 'TS': 
                self.assertEqual(result[score], 1/6)
                # TS = h / h+f+m = 1 / 1+2+3
            elif score == 'OR':
                self.assertEqual(result[score], 2/3)
                # OR = hc / fm = 1*4 / 2*3
            elif score == 'GSS':
                self.assertAlmostEqual(result[score], -1/24)
                # GSS = h-(h+f)*(h+m)/n / h+f+m-(h+f)*(h+m)/n = 1-(1+3)*(1+2)/10 / 1+3+2-(1+3)(1+2)/10    
            elif score == 'TSS':
                self.assertEqual(result[score], -2/21)
                # TSS = h / h+m - f / f+c = 1 / 1+2 - 3 / 3+4
            elif score == 'HSS':
                self.assertEqual(result[score], -2/23)
                # HSS = 2.0*(h*c-f*m) / (h+m)*(m+c)+(h+f)*(f+c) = 2*(1*4 - 2*3) / (1+2)(2+4)+(1+3)(3+4)
            elif score == 'ORSS':
                self.assertEqual(result[score], -1/5)
                # ORSS = h*c-m*f / h*c+m*f = 1*4-2*3 / 1*4+2*3
            elif score == 'SEDS':
                self.assertEqual(result[score], ((np.log(4/10)+np.log(3/10))/np.log(1/10)) - 1)
                # SEDS = (log((h+f)/n)+log((h+m)/n) / log(h/n)) - 1 = (log((1+3)/10)+log((1+2)/10) / log(1/10)) - 1

class TimeMetricsTestCase(unittest.TestCase):
    def test_time_mean_error(self):
        y_true = [datetime.datetime.utcnow()]
        y_pred = [datetime.datetime.utcnow() - datetime.timedelta(days=1)]
        result = metrics.calc_E(y_true, y_pred)
        hand_calc =  y_pred[0] - y_true[0]
        self.assertEqual(result[0], hand_calc)


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
        self.assertEqual(result_list, hand_calc_list)

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
        self.assertEqual(result[0], hand_calc)


        y_true = [datetime.datetime.utcnow(), datetime.datetime.utcnow(), datetime.datetime.utcnow(), datetime.datetime.utcnow()]
        y_pred = [datetime.datetime.utcnow() + datetime.timedelta(days=1), datetime.datetime.utcnow() - datetime.timedelta(days=1), datetime.datetime.utcnow() + datetime.timedelta(days=1), datetime.datetime.utcnow() - datetime.timedelta(days=1)]
        zipped = zip(y_pred, y_true)
        temp = []
        for elements in zipped:
            temp.append(elements[0] - elements[1])
        hand_calc = np.median(temp)
        result = np.median(metrics.calc_E(y_true, y_pred))
        self.assertEqual(result, hand_calc)

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
        self.assertEqual(result[0], hand_calc)


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
        self.assertEqual(result_list, hand_calc_list)
      
      

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
        self.assertEqual(result[0], hand_calc)


        y_true = [datetime.datetime.utcnow(), datetime.datetime.utcnow(), datetime.datetime.utcnow(), datetime.datetime.utcnow()]
        y_pred = [datetime.datetime.utcnow() + datetime.timedelta(days=1), datetime.datetime.utcnow() - datetime.timedelta(days=1), datetime.datetime.utcnow() + datetime.timedelta(days=1), datetime.datetime.utcnow() - datetime.timedelta(days=1)]
        zipped = zip(y_pred, y_true)
        temp = []
        for elements in zipped:
            temp.append(np.abs(elements[0] - elements[1]))
        hand_calc = np.median(temp)
        result = np.median(metrics.calc_AE(y_true, y_pred))
        self.assertEqual(result, hand_calc)

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
