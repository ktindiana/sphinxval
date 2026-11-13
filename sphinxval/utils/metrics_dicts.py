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


def initialize_fluence_spectrum_selections_dict():
    selection_file_keys = []
    selection_keys = ['Band Function Low Energy Spectral Slope', 'Band Function High Energy Spectral Slope', 'Band Function Rollover Energy',
    'Band Function Normalization Factor', 'Band Function ChiSquared', 'Band Function Reduced ChiSquared', 'Ellison Ramaty Spectral Slope', 'Ellison Ramaty Rollover Energy', 
    'Ellison Ramaty Normalization Factor', 'Ellison Ramaty ChiSquared', 'Ellison Ramaty Reduced ChiSquared']
    type_keys = ['Observed ', 'Model ']
    other_selection_keys = ['Model', 'Energy Channel', 'Threshold', 'Prediction Energy Channel', 'Prediction Threshold', 'Predicted SEP Fluence Spectrum',
    'Predicted SEP Fluence Spectrum Units', 'Observed SEP Fluence Spectrum', 'Observed SEP Fluence Spectrum Units']
    for keys in other_selection_keys:
        selection_file_keys.append(keys)
    for pair in type_keys:
        for keys in selection_keys:
            if 'ChiSquared' not in keys:
                selection_file_keys.append(pair + keys)
                selection_file_keys.append(pair + keys + ' Uncertainty')
            else:
                selection_file_keys.append(pair + keys)
    selection_file_keys.append('Band Function Fit Plot')
    selection_file_keys.append('Ellison Ramaty Fit Plot')
    selection_file_dict = {key: [] for key in selection_file_keys}


    return selection_file_dict


def initialize_fluence_spectrum_metrics_dict():
    metrics_keys = []
    # selection_file_keys = []
    # selection_keys = ['Band Function Low Energy Spectral Slope', 'Band Function High Energy Spectral Slope', 'Band Function Rollover Energy',
    # 'Band Function Normalization Factor', 'Ellison Ramaty Spectral Slope', 'Ellison Ramaty Rollover Energy', 
    # 'Ellison Ramaty Normalization Factor']
    # type_keys = ['Observed ', 'Model ']
    generic_keys = ['Model', 'Energy Channel', 'Threshold']
    for keys in generic_keys:
        metrics_keys.append(keys)

    
    plot_keys = ['Band Function Low Energy Spectral Slope Correlation Plot', 'Band Function High Energy Spectral Slope Correlation Plot',
    'Band Function Rollover Energy Correlation Plot', 'Ellison Ramaty Spectral Slope Correlation Plot',
    'Ellison Ramaty Rollover Energy Correlation Plot', 'Histogram of Log Errors Plot']

    metric_median_keys = ['Mean Band Function Low Energy Spectral Slope Error', 'Median Band Function Low Energy Spectral Slope Error',
    'Mean Band Function High Energy Spectral Slope Error', 'Median Band Function High Energy Spectral Slope Error',
    'Mean ']

    fit_parameter_keys = ['Band Function Low Energy Spectral Slope', 'Band Function High Energy Spectral Slope', 'Band Function Rollover Energy',
    'Ellison Ramaty Spectral Slope', 'Ellison Ramaty Rollover Energy']
    for keys in fit_parameter_keys:
        metrics_keys.append('Median ' + keys + ' Error')
        metrics_keys.append('Mean ' + keys + ' Error')
        metrics_keys.append(keys + ' Correlation Plot')
    metrics_keys.append('Log Errors Dictionary')
    metrics_keys.append('Histogram of Log Errors Plot')



    fluence_spectrum_metrics_dict = {key: [] for key in metrics_keys}
    return fluence_spectrum_metrics_dict