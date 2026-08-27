def initialize_flux_dict():
    """ Metrics used for fluxes.
    
    """
    uncert_keys = ["Linear Regression Slope", "Linear Regression y-intercept", "Pearson Correlation Coefficient (Linear)",
            "Pearson Correlation Coefficient (Log)", "Spearman Correlation Coefficient (Linear)", "Mean Ratio",
            "Median Ratio", "Mean Error (ME)", "Median Error (MedE)", "Mean Log Error (MLE)", "Median Log Error (MedLE)",
            "Mean Absolute Error (MAE)", "Median Absolute Error (MedAE)", "Mean Absolute Log Error (MALE)",
            "Median Absolute Log Error (MedALE)", "Mean Percent Error (MPE)", "Mean Absolute Percent Error (MAPE)",
            "Mean Symmetric Percent Error (MSPE)", "Mean Symmetric Absolute Percent Error (SMAPE)", "Mean Accuracy Ratio (MAR)",
            "Root Mean Square Error (RMSE)", "Root Mean Square Log Error (RMSLE)","Median Symmetric Accuracy (MdSA)", 
            "Percentage within an Order of Magnitude (%)", "Percentage within a factor of 2 (%)"]
        
    other_keys = ['Model', 'Energy Channel', 'Threshold', 'Prediction Energy Channel', 'Prediction Threshold', 
                    "Scatter Plot"]
    flux_keys = []

    for keys in other_keys:
        flux_keys.append(keys)
    for keys in uncert_keys:
        flux_keys.append(keys)
        flux_keys.append(keys + ' Uncertainty')

    flux_dict = {key: [] for key in flux_keys}
    

    
    return flux_dict

def initialize_time_dict():
    """ Metrics for predictions related to time.
    
    """
    uncert_keys = ["Mean Error (pred - obs)", "Median Error (pred - obs)", "Mean Absolute Error (|pred - obs|)",
                    "Median Absolute Error (|pred - obs|)"]
    other_keys = ['Model', 'Energy Channel', 'Threshold', 'Prediction Energy Channel', 'Prediction Threshold']
    time_keys = []

    for keys in other_keys:
        time_keys.append(keys)
    for keys in uncert_keys:
        time_keys.append(keys)
        time_keys.append(keys + ' Uncertainty')

    time_dict = {key: [] for key in time_keys}
 
            
    return time_dict
    
    
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


def initialize_all_clear_dict():
    """ Metrics for all clear predictions.
    
    """
    uncert_keys = ["All Clear 'True Positives' (Hits)", "All Clear 'False Positives' (False Alarms)",
            "All Clear 'True Negatives' (Correct Negatives)", "All Clear 'False Negatives' (Misses)", "Percent Correct",
            "Bias", "Hit Rate", "False Alarm Rate", 'False Negative Rate', "Frequency of Misses", "Frequency of Hits",
            "Probability of Correct Negatives", "Frequency of Correct Negatives", "False Alarm Ratio", "Detection Failure Ratio", "Threat Score",
            'False Alarm Event Ratio', 'Tau',
            "Odds Ratio", "Gilbert Skill Score", "True Skill Statistic", "Heidke Skill Score", "Odds Ratio Skill Score",
            "Symmetric Extreme Dependency Score", "F1 Score", "F2 Score", "Fhalf Score", 'Prevalence', 'Matthew Correlation Coefficient',
            'Informedness', 'Markedness', 'Prevalence Threshold', 'Balanced Accuracy', 'Fowlkes-Mallows Index']
        
    other_keys = ['Model', 'Energy Channel', 'Threshold', 'Prediction Energy Channel', 'Prediction Threshold', 
                    "N (Total Number of Forecasts)", "Number SEP Events Correctly Predicted","Number SEP Events Missed", 
                    "Predicted SEP Events" , "Missed SEP Events"]
    all_clear_keys = []

    for keys in other_keys:
        all_clear_keys.append(keys)
    for keys in uncert_keys:
        all_clear_keys.append(keys)
        all_clear_keys.append(keys + ' Uncertainty')

    all_clear_dict = {key: [] for key in all_clear_keys}
    return all_clear_dict


            
def initialize_probability_dict():
    """ Metrics for probability predictions.
    
    """
    prob_keys = []
    uncert_keys = ["Brier Score", "Brier Skill Score", "Spearman Correlation Coefficient", "Area Under ROC Curve"]
    other_keys = ['Model', 'Energy Channel', 'Threshold', 'Prediction Energy Channel', 'Prediction Threshold', "ROC Curve Plot"]
    for keys in other_keys:
        prob_keys.append(keys)
    for keys in uncert_keys:
        prob_keys.append(keys)
        prob_keys.append(keys + ' Uncertainty')

    prob_dict = {key: [] for key in prob_keys}            
    return prob_dict

