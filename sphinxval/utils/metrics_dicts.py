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
            "Pearson Correlation Coefficient (Log)": [],
            "Spearman Correlation Coefficient (Linear)": [],
            "Mean Ratio": [],
            "Median Ratio": [],
            "Mean Error (ME)": [],
            "Median Error (MedE)": [],
            "Mean Log Error (MLE)": [],
            "Median Log Error (MedLE)": [],
            "Mean Absolute Error (MAE)": [],
            "Median Absolute Error (MedAE)": [],
            "Mean Absolute Log Error (MALE)": [],
            "Median Absolute Log Error (MedALE)": [],
            "Mean Percent Error (MPE)": [],
            "Mean Absolute Percent Error (MAPE)": [],
            "Mean Symmetric Percent Error (MSPE)": [],
            "Mean Symmetric Absolute Percent Error (SMAPE)": [],
            "Mean Accuracy Ratio (MAR)": [],
            "Root Mean Square Error (RMSE)": [],
            "Root Mean Square Log Error (RMSLE)": [],
            "Median Symmetric Accuracy (MdSA)": [],
            "Percentage within an Order of Magnitude (%)": [],
            "Percentage within a factor of 2 (%)": []
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
            "Median Error (pred - obs)": [],
            "Mean Absolute Error (|pred - obs|)": [],
            "Median Absolute Error (|pred - obs|)": [],
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


def initialize_all_clear_dict():
    """ Metrics for all clear predictions.
    
    """
    dict = {"Model": [],
            "Energy Channel": [],
            "Threshold": [],
            "Prediction Energy Channel": [],
            "Prediction Threshold": [],
            "All Clear 'True Positives' (Hits)": [], #Hits
            "All Clear 'False Positives' (False Alarms)": [], #False Alarms
            "All Clear 'True Negatives' (Correct Negatives)": [],  #Correct negatives
            "All Clear 'False Negatives' (Misses)": [], #Misses
            "N (Total Number of Forecasts)": [],
            "Percent Correct": [],
            "Bias": [],
            "Hit Rate": [],
            "False Alarm Rate": [],
            'False Negative Rate': [],
            "Frequency of Misses": [],
            "Frequency of Hits": [],
            "Probability of Correct Negatives": [],
            "Frequency of Correct Negatives": [],
            "False Alarm Ratio": [],
            "Detection Failure Ratio": [],
            "Threat Score": [],
            "Odds Ratio": [],
            "Gilbert Skill Score": [],
            "True Skill Statistic": [],
            "Heidke Skill Score": [],
            "Odds Ratio Skill Score": [],
            "Symmetric Extreme Dependency Score": [],
            "F1 Score": [],
            "F2 Score": [],
            "Fhalf Score": [],
            'Prevalence': [],
            'Matthew Correlation Coefficient': [],
            'Informedness': [],
            'Markedness': [],
            'Prevalence Threshold': [],
            'Balanced Accuracy': [],
            'Fowlkes-Mallows Index': [],
            "Number SEP Events Correctly Predicted": [],
            "Number SEP Events Missed": [],
            "Predicted SEP Events": [], #date string
            "Missed SEP Events": [] #date string
#            "Mean Percentage Error": [],
#            "Mean Absolute Percentage Error": []
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
            "Brier Skill Score": [],
            "Spearman Correlation Coefficient": [],
            "Area Under ROC Curve": []
            }
            
    return dict

