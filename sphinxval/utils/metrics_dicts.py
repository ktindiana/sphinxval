def initialize_flux_dict():
    """ Metrics used for fluxes.
    
    """
    dict = {"Model": [],
            "Energy Channel": [],
            "Threshold": [],
            "Prediction Energy Channel": [],
            "Prediction Threshold": [],
            "Scatter Plot": [],
            "Linear Regression Slope": [],
            "Linear Regression y-intercept": [],
            "Pearson Correlation Coefficient (Linear)": [],
            "Pearson Correlation Coefficient (Linear) Uncertainty": [],
            "Pearson Correlation Coefficient (Log)": [],
            "Pearson Correlation Coefficient (Log) Uncertainty": [],
            "Spearman Correlation Coefficient (Linear)": [],
            "Spearman Correlation Coefficient (Linear) Uncertainty": [],
            "Mean Ratio": [],
            "Mean Ratio Uncertainty": [],
            "Median Ratio": [],
            "Median Ratio Uncertainty": [],
            "Mean Error (ME)": [],
            "Mean Error (ME) Uncertainty": [],
            "Median Error (MedE)": [],
            "Median Error (MedE) Uncertainty": [],
            "Mean Log Error (MLE)": [],
            "Mean Log Error (MLE) Uncertainty": [],
            "Median Log Error (MedLE)": [],
            "Median Log Error (MedLE) Uncertainty": [],
            "Mean Absolute Error (MAE)": [],
            "Mean Absolute Error (MAE) Uncertainty": [],
            "Median Absolute Error (MedAE)": [],
            "Median Absolute Error (MedAE) Uncertainty": [],
            "Mean Absolute Log Error (MALE)": [],
            "Mean Absolute Log Error (MALE) Uncertainty": [],
            "Median Absolute Log Error (MedALE)": [],
            "Median Absolute Log Error (MedALE) Uncertainty": [],
            "Mean Percent Error (MPE)": [],
            "Mean Percent Error (MPE) Uncertainty": [],
            "Mean Absolute Percent Error (MAPE)": [],
            "Mean Absolute Percent Error (MAPE) Uncertainty": [],
            "Mean Symmetric Percent Error (MSPE)": [],
            "Mean Symmetric Percent Error (MSPE) Uncertainty": [],
            "Mean Symmetric Absolute Percent Error (SMAPE)": [],
            "Mean Symmetric Absolute Percent Error (SMAPE) Uncertainty": [],
            "Mean Accuracy Ratio (MAR)": [],
            "Mean Accuracy Ratio (MAR) Uncertainty": [],
            "Root Mean Square Error (RMSE)": [],
            "Root Mean Square Error (RMSE) Uncertainty": [],
            "Root Mean Square Log Error (RMSLE)": [],
            "Root Mean Square Log Error (RMSLE) Uncertainty": [],
            "Median Symmetric Accuracy (MdSA)": [],
            "Median Symmetric Accuracy (MdSA) Uncertainty": [],
            "Percentage within an Order of Magnitude (%)": [],
            "Percentage within an Order of Magnitude (%) Uncertainty": [],
            "Percentage within a factor of 2 (%)": [],
            "Percentage within a factor of 2 (%) Uncertainty": []
            }
    
    return dict


def initialize_time_dict():
    """ Metrics for predictions related to time.
    
    """
    dict = {"Model": [],
            "Energy Channel": [],
            "Threshold": [],
            "Prediction Energy Channel": [],
            "Prediction Threshold": [],
            "Mean Error (pred - obs)": [],
            "Mean Error (pred - obs) Uncertainty": [],
            "Median Error (pred - obs)": [],
            "Median Error (pred - obs) Uncertainty": [],
            "Mean Absolute Error (|pred - obs|)": [],
            "Mean Absolute Error (|pred - obs|) Uncertainty": [],
            "Median Absolute Error (|pred - obs|)": [],
            "Median Absolute Error (|pred - obs|) Uncertainty": []
            }
            
    return dict
    
    
def initialize_awt_dict():
    """ Metrics for Adanced Warning Time to SEP start, SEP peak, SEP end.
        The "Forecasted Value" field indicates which forecasted quantity
        was used to calculate the AWT.
    """
    dict = {"Model": [],
            "Energy Channel": [],
            "Threshold": [],
            "Prediction Energy Channel": [],
            "Prediction Threshold": [],
            
            #All Clear Forecasts
            #Commenting out redundant comparisons to Observed SEP Threshold Crossing Time and
            #observed SEP Start Time. If there is ever a case where those fields are different
            #then should uncomment them.
            "Mean AWT for Predicted SEP All Clear to Observed SEP Threshold Crossing Time": [],
            "Median AWT for Predicted SEP All Clear to Observed SEP Threshold Crossing Time": [],
            "Mean AWT Efficiency for Predicted SEP All Clear to Observed SEP Threshold Crossing Time": [],
            #Threshold Crossing Time Forecasts
            "Mean AWT for Predicted SEP Threshold Crossing Time to Observed SEP Threshold Crossing Time": [],
            "Median AWT for Predicted SEP Threshold Crossing Time to Observed SEP Threshold Crossing Time": [],
            "Mean AWT Efficiency for Predicted SEP Threshold Crossing Time to Observed SEP Threshold Crossing Time": [], 
            #Start Time Forecasts
            "Mean AWT for Predicted SEP Start Time to Observed SEP Threshold Crossing Time": [],
            "Median AWT for Predicted SEP Start Time to Observed SEP Threshold Crossing Time": [],
            "Mean AWT Efficiency for Predicted SEP Start Time to Observed SEP Threshold Crossing Time": [],
 
#             #Point Intensity Forecasts
#            "Mean AWT for Predicted Point Intensity to Observed SEP Threshold Crossing Time": [],
#            "Median AWT for Predicted Point Intensity to Observed SEP Threshold Crossing Time": [],
#            "Mean AWT for Predicted Point Intensity to Observed SEP Start Time": [],
#            "Median AWT for Predicted Point Intensity to Observed SEP Start Time": [],
 
 
            #Peak Intensity Forecasts
            "Mean AWT for Predicted SEP Peak Intensity (Onset Peak) to Observed SEP Threshold Crossing Time": [],
            "Median AWT for Predicted SEP Peak Intensity (Onset Peak) to Observed SEP Threshold Crossing Time": [],
#            "Mean AWT for Predicted SEP Peak Intensity (Onset Peak) to Observed SEP Start Time": [],
#            "Median AWT for Predicted SEP Peak Intensity (Onset Peak) to Observed SEP Start Time": [],
            "Mean AWT for Predicted SEP Peak Intensity (Onset Peak) to Observed SEP Peak Intensity (Onset Peak) Time": [],
            "Median AWT for Predicted SEP Peak Intensity (Onset Peak) to Observed SEP Peak Intensity (Onset Peak) Time": [],
 #           "Mean AWT for Predicted SEP Peak Intensity (Onset Peak) to Observed SEP Peak Intensity Max (Max Flux) Time": [],
 #           "Median AWT for Predicted SEP Peak Intensity (Onset Peak) to Observed SEP Peak Intensity Max (Max Flux) Time": [],

            #Peak Intensity Max Forecasts
            "Mean AWT for Predicted SEP Peak Intensity Max (Max Flux) to Observed SEP Threshold Crossing Time": [],
            "Median AWT for Predicted SEP Peak Intensity Max (Max Flux) to Observed SEP Threshold Crossing Time": [],
#            "Mean AWT for Predicted SEP Peak Intensity Max (Max Flux) to Observed SEP Start Time": [],
#            "Median AWT for Predicted SEP Peak Intensity Max (Max Flux) to Observed SEP Start Time": [],
            "Mean AWT for Predicted SEP Peak Intensity Max (Max Flux) to Observed SEP Peak Intensity Max (Max Flux) Time": [],
            "Median AWT for Predicted SEP Peak Intensity Max (Max Flux) to Observed SEP Peak Intensity Max (Max Flux) Time": [],

            #End Time Forecasts
            "Mean AWT for Predicted SEP End Time to Observed SEP Threshold Crossing Time": [],
            "Median AWT for Predicted SEP End Time to Observed SEP Threshold Crossing Time": [],
#            "Mean AWT for Predicted SEP End Time to Observed SEP Start Time": [],
#            "Median AWT for Predicted SEP End Time to Observed SEP Start Time": [],
            "Mean AWT for Predicted SEP End Time to Observed SEP End Time": [],
            "Median AWT for Predicted SEP End Time to Observed SEP End Time": []
            }
            
    return dict


# def initialize_all_clear_dict():
#     """ Metrics for all clear predictions.
    
#     """
#     uncert_keys = ["All Clear 'True Positives' (Hits)", "All Clear 'False Positives' (False Alarms)",
#             "All Clear 'True Negatives' (Correct Negatives)", "All Clear 'False Negatives' (Misses)", "Percent Correct",
#             "Bias", "Hit Rate", "False Alarm Rate", 'False Negative Rate', "Frequency of Misses", "Frequency of Hits",
#             "Probability of Correct Negatives", "Frequency of Correct Negatives", "False Alarm Ratio", "Detection Failure Ratio", "Threat Score",
#             "Odds Ratio", "Gilbert Skill Score", "True Skill Statistic", "Heidke Skill Score", "Odds Ratio Skill Score",
#             "Symmetric Extreme Dependency Score", "F1 Score", "F2 Score", "Fhalf Score", 'Prevalence', 'Matthew Correlation Coefficient',
#             'Informedness', 'Markedness', 'Prevalence Threshold', 'Balanced Accuracy', 'Fowlkes-Mallows Index']
        
#     other_keys = ['Model', 'Energy Channel', 'Threshold', 'Prediction Energy Channel', 'Prediction Threshold', 
#                     "N (Total Number of Forecasts)", "Number SEP Events Correctly Predicted","Number SEP Events Missed", 
#                     "Predicted SEP Events" , "Missed SEP Events"]
#     all_clear_keys = []

#     for keys in other_keys:
#         all_clear_keys.append(keys)
#     for keys in uncert_keys:
#         all_clear_keys.append(keys)
#         all_clear_keys.append(keys + ' Uncertainty')

    
#     all_clear_dict = dict.fromkeys(all_clear_keys, [])
#     return all_clear_dict



def initialize_all_clear_dict():
    """ Metrics for all clear predictions.
    
    """
    dict = {"Model": [],
            "Energy Channel": [],
            "Threshold": [],
            "Prediction Energy Channel": [],
            "Prediction Threshold": [],
            "All Clear 'True Positives' (Hits)": [],
            "All Clear 'True Positives' (Hits) Uncertainty": [],
            "All Clear 'False Positives' (False Alarms)": [],
            "All Clear 'False Positives' (False Alarms) Uncertainty": [],
            "All Clear 'True Negatives' (Correct Negatives)": [],  
            "All Clear 'True Negatives' (Correct Negatives) Uncertainty": [],  
            "All Clear 'False Negatives' (Misses)": [],
            "All Clear 'False Negatives' (Misses) Uncertainty": [],
            "N (Total Number of Forecasts)": [],
            "Percent Correct": [],
            "Percent Correct Uncertainty": [],
            "Bias": [],
            "Bias Uncertainty": [],
            "Hit Rate": [],
            "Hit Rate Uncertainty": [],
            "False Alarm Rate": [],
            "False Alarm Rate Uncertainty": [],
            "False Negative Rate": [],
            "False Negative Rate Uncertainty": [],
            "Frequency of Misses": [],
            "Frequency of Misses Uncertainty": [],
            "Frequency of Hits": [],
            "Frequency of Hits Uncertainty": [],
            "Probability of Correct Negatives": [],
            "Probability of Correct Negatives Uncertainty": [],
            "Frequency of Correct Negatives": [],
            "Frequency of Correct Negatives Uncertainty": [],
            "False Alarm Ratio": [],
            "False Alarm Ratio Uncertainty": [],
            "Detection Failure Ratio": [],
            "Detection Failure Ratio Uncertainty": [],
            "Threat Score": [],
            "Threat Score Uncertainty": [],
            "Odds Ratio": [],
            "Odds Ratio Uncertainty": [],
            "Gilbert Skill Score": [],
            "Gilbert Skill Score Uncertainty": [],
            "True Skill Statistic": [],
            "True Skill Statistic Uncertainty": [],
            "Heidke Skill Score": [],
            "Heidke Skill Score Uncertainty": [],
            "Odds Ratio Skill Score": [],
            "Odds Ratio Skill Score Uncertainty": [],
            "Symmetric Extreme Dependency Score": [],
            "Symmetric Extreme Dependency Score Uncertainty": [],
            "F1 Score": [],
            "F1 Score Uncertainty": [],
            "F2 Score": [],
            "F2 Score Uncertainty": [],
            "Fhalf Score": [],
            "Fhalf Score Uncertainty": [],
            'Prevalence': [],
            'Prevalence Uncertainty': [],
            'Matthew Correlation Coefficient': [],
            'Matthew Correlation Coefficient Uncertainty': [],
            'Informedness': [],
            'Informedness Uncertainty': [],
            'Markedness': [],
            'Markedness Uncertainty': [],
            'Prevalence Threshold': [],
            'Prevalence Threshold Uncertainty': [],
            'Balanced Accuracy': [],
            'Balanced Accuracy Uncertainty': [],
            'Fowlkes-Mallows Index': [],
            'Fowlkes-Mallows Index Uncertainty': [],
            "Number SEP Events Correctly Predicted": [],
            "Number SEP Events Missed": [],
            "Predicted SEP Events": [], #date string
            "Missed SEP Events": [] #date string
            }
            
    return dict

            
def initialize_probability_dict():
    """ Metrics for probability predictions.
    
    """
    dict = {"Model": [],
            "Energy Channel": [],
            "Threshold": [],
            "Prediction Energy Channel": [],
            "Prediction Threshold": [],
            "ROC Curve Plot": [],
            "Brier Score": [],
            "Brier Score Uncertainty": [],
            "Brier Skill Score": [],
            "Brier Skill Score Uncertainty": [],
            "Spearman Correlation Coefficient": [],
            "Spearman Correlation Coefficient Uncertainty": [],
            "Area Under ROC Curve": [],
            "Area Under ROC Curve Uncertainty": []
            }
            
    return dict

