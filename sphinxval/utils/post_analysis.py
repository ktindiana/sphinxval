''' Functions for post analysis of the validation results
	produced by SPHINX.

    Make plots and provides selected information from
    the output pkl and csv
'''
import sys
from . import plotting_tools as plt_tools
from . import resume
import pickle
import pandas as pd

#Columns to exclude from box plots
exclude = ["N (Total Number of Forecasts)", "Predicted SEP Events",
        "Missed SEP Events", "Scatter Plot", "Linear Regression y-intercept",
        "ROC Curve Plot"]



def export_all_clear_false_alarms(filename, doplot=False):
    """ Provide the filename of an all_clear_selections_*.pkl
        file.
        
        Select cases where observed All Clear is True and
        predicted All Clear is False.
        
        Output as a csv file.
        Plot all forecasts with time with the False Alarms highlighted.
        
        INPUT:
        
        :filename: (string) name of all_clear_selections_*.pkl
            file. Full path.
            
        :doplot: (bool) set to True to plot false alarms with time
            
        OUTPUT:
        
        Write out csv file with false alarms.
        Create plot with distribution of times between false alarms.
        
    """
    
    df = resume.read_in_df(filename)
    
    if df.empty:
        return

    sub = df.loc[(df["Observed SEP All Clear"] == True) & (df["Predicted SEP All Clear"] == False)]
    
    if sub.empty:
        print("post_analysis: export_all_clear_false_alarms: No false alarms identified. Returning.")
        return
    
    
    fname = filename.replace(".pkl","_false_alarms.csv")
    figname = filename.replace(".pkl","_false_alarms.png")
    fname = fname.replace("pkl","csv")
    figname = figname.replace("pkl","plots")
    
    #Write false alarms out to csv file
    sub.to_csv(fname)
    

    all_dates = df["Prediction Window Start"].to_list()
    fa_dates = sub["Prediction Window Start"].to_list()

    if doplot:
        model = sub["Model"].iloc[0]
        energy_channel = sub["Energy Channel Key"].iloc[0]
        thresh_key = sub["Threshold Key"].iloc[0]
        
        title = model + " False Alarms (" + energy_channel + ", " + thresh_key +")"
        
        mismatch = sub["Mismatch Allowed"].iloc[0]
        if mismatch:
            pred_energy_channel = sub["Prediction Energy Channel Key"].iloc[0]
            pred_thresh_key = sub["Prediction Threshold Key"].iloc[0]
            title = model + " False Alarms (Observations: " + energy_channel \
                    + ", " + thresh_key +" and "  + " Predictions: " \
                    + pred_energy_channel + ", " + pred_thresh_key +")"
        
        labels = ["All Forecasts", "False Alarms"]
        fig, _ = plt_tools.plot_false_alarms(all_dates, fa_dates, labels,
            x_label="Date", y_label="", date_format="year", title=title,
            figname=figname, saveplot=True, showplot=True)
        

def get_file_prefix(quantity):
    """ File prefix for various forecasted quantities.
    
    """
    dict = {"All Clear": "all_clear",
            "Advanced Warning Time": "awt",
            "Probability": "probability",
            "Threshold Crossing Time": "threshold_crossing_time",
            "Start Time": "start_time",
            "End Time": "end_time",
            "Onset Peak Time": "peak_intensity_time",
            "Onset Peak": "peak_intensity",
            "Max Flux Time": "peak_intensity_max_time",
            "Max Flux": "peak_intensity_max",
            "Max Flux in Prediction Window": "max_flux_in_pred_win",
            "Duration": "duration",
            "Fluence": "fluence",
            "Time Profile": "time_profile"
            }

    if quantity not in dict.keys():
        sys.exit("post_analysis: " + quantity + "not valid. Choose one "
            + str(dict.keys()))

    return dict[quantity]
    


def read_in_metrics(path, quantity):
    """ Read in metrics files related to specfied quantity.
    
    INPUT:
    
    :path: (string) location of the output/ folder
    :quantity: (string) Forecasted quantity of interest.
    
    OUTPUT:
    
    :df: (pandas DataFrame) dataframe containing all the metrics
    
    """
    
    prefix = get_file_prefix(quantity)
    fname = path + "/output/pkl/" + prefix + "_metrics.pkl"
    
    df = resume.read_in_df(fname)
    
    return df


def plot_groups(quantity):
    """ Return metrics that should be plotted together according
        to forecasted quantity.
        
        INPUT:
        
            :quantity: (string) forecasted quantity
            
        OUTPUT:
        
            :groups: (arr of strings) arrays containing metric names to be
                be plotted together
                
    """
    #ALL CLEAR
    if quantity == "All Clear":
        groups = [  ["All Clear 'True Positives' (Hits)",
                    "All Clear 'False Positives' (False Alarms)",
                    "All Clear 'True Negatives' (Correct Negatives)",
                    "All Clear 'False Negatives' (Misses)"],
                    ["Percent Correct", "Bias", "Hit Rate", "False Alarm Rate",
                    "Frequency of Misses", "Frequency of Hits"],
                    ["Probability of Correct Negatives",
                    "Frequency of Correct Negatives", "False Alarm Ratio",
                    "Detection Failure Ratio", "Threat Score"],
                    ["Odds Ratio"],
                    ["Gilbert Skill Score", "True Skill Statistic",
                    "Heidke Skill Score", "Odds Ratio Skill Score",
                    "Symmetric Extreme Dependency Score"],
                    ["Number SEP Events Correctly Predicted",
                    "Number SEP Events Missed"]
                ]

    #PROBABILITY
    if quantity == "Probability":
        groups = [ ["Brier Score", "Brier Skill Score",
                    "Spearman Correlation Coefficient", "Area Under ROC Curve"]
                ]

    #FLUX METRICS
    flux_types = ["Onset Peak", "Max Flux", "Fluence",
                "Max Flux in Prediction Window", "Time Profile"]
    if quantity in flux_types:
        groups = [ ["Linear Regression Slope",
                    "Pearson Correlation Coefficient (Linear)",
                    "Pearson Correlation Coefficient (Log)",
                    "Spearman Correlation Coefficient (Linear)",
                    "Spearman Correlation Coefficient (Log)"],
                    ["Mean Error (ME)", "Median Error (MedE)"],
                    ["Mean Absolute Error (MAE)",
                    "Median Absolute Error (MedAE)"],
                    ["Root Mean Square Error (RMSE)"],
                    ["Mean Log Error (MLE)", "Median Log Error (MedLE)"],
                    ["Mean Absolute Log Error (MALE)",
                    "Median Absolute Log Error (MedALE)",
                    "Root Mean Square Log Error (RMSLE)"],
                    ["Mean Percent Error (MPE)",
                    "Mean Absolute Percent Error (MAPE)",
                    "Mean Accuracy Ratio (MAR)"],
                    ["Mean Symmetric Percent Error (MSPE)",
                    "Mean Symmetric Absolute Percent Error (SMAPE)"],
                    ["Median Symmetric Accuracy (MdSA)"]
                ]

    #TIME METRICS
    time_types = ["Threshold Crossing Time", "Start Time", "End Time",
                "Onset Peak Time", "Max Flux Time"]
    if quantity in time_types:
        groups = []

    return groups



def make_box_plots(df, path, quantity, anonymous, highlight):
    """ Take a dataframe of metrics and generate box plots
        of each of the metrics.
        
        If anonymous = True, then will generate a generic lengend, i.e.
            Model 1, Model 2
            
        If a value is specified for highlight, will use that model
            name in the legend and set data points to red.
            
        INPUT:
        
        :df: (pandas DataFrame) contains metrics
        :anonymous: (bool) False uses model names in legend.
            True uses generic names in legend.
        :highlight: (string) model name to highlight on the plot.
            If anonymous True, then this model name will be shown.
            Points corresponding to this model will be in red.
            
        OUTPUT:
        
        Figure(s) with box plots will be written to the
        path/output/plots/. directory
    
    """

    energy_channels = resume.identify_unique(df,'Energy Channel')
    thresholds = resume.identify_thresholds_per_energy_channel(df,
            ek_name='Energy Channel', tk_name='Threshold')

    groups = plot_groups(quantity)

    #Make plots according to energy channel and threshold combinations
    for ek in energy_channels:
        thresh = thresholds[ek]
        for tk in thresh:
            print(ek + ", " + tk)
            sub = df.loc[(df['Energy Channel'] == ek) &
                    (df['Threshold'] == tk)]

            
            
            grp = 0
            for group in groups:
                grp += 1
                values = []
                metric_names = []
                model_names = []
                for metric_col in group:
                    vals = sub[metric_col].to_list()
                    values.extend(vals)
                    metric_names.extend([metric_col]*len(vals))
                    model_list = sub['Model'].to_list()
                    
                    if anonymous and highlight == '':
                        for j in range(len(model_list)):
                            model_list[j] = "Model " + str(j)
                    
                    if highlight != '':
                        for j in range(len(model_list)):
                            if highlight in model_list[j]:
                                continue
                            else:
                                model_list[j] = "Models"
                    
                    model_names.extend(model_list)
 
                
                dict = {"Metrics": metric_names, "Models":model_names,
                        "Values":values}
                metrics_df = pd.DataFrame(dict)
                
          
                title = quantity + " Group " + str(grp) + " (" + ek + ", " + tk + ")"
                figname = path + "/output/plots/" + quantity + "_" + ek + "_" + tk \
                        + "_boxes_Group" + str(grp)
                plt_tools.box_plot_metrics(metrics_df, group, highlight,
                    x_label="Metric", y_label="Value", title=title,
                    save=figname, uselog=False, showplot=True, \
                    closeplot=False)

