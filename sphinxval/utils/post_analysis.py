''' Functions for post analysis of the validation results
	produced by SPHINX.

    Make plots and provides selected information from
    the output pkl and csv
'''
import sys
from . import plotting_tools as plt_tools
from . import time_profile as profile
from . import resume
from . import validation
from . import metrics
import pickle
import pandas as pd
import matplotlib as plt
from . import config as cfg
from . import metrics
from . import validation
import datetime
import os.path
import numpy as np
import sklearn.metrics as skl
import matplotlib.pylab as plt
from scipy import stats 
from statsmodels.distributions.empirical_distribution import ECDF
import math

scoreboard_models = ["ASPECS", "iPATH", "MagPy", "SEPMOD",
                    "SEPSTER", "SPRINTS", "UMASEP", "GSU",
                    "MAG4", "REleASE"]


#If not empty, add metrics to the contingency metrics analysis
add_contingency = {}
#e.g.
#add_contingency = {
#"Model": ['UMASEP-10'],
#"Energy Channel": ['min.10.0.max.-1.0.units.MeV'],
#"Threshold": ['threshold.10.0.units.1 / (cm2 s sr)'],
#"Prediction Energy Channel": ['min.10.0.max.-1.0.units.MeV'],
#"Prediction Threshold": ['threshold.10.0.units.1 / (cm2 s sr)'],
#"Hits": [30], #Hits
#"False Alarms": [1], #False Alarms
#"Correct Negatives": [29],  #Correct negatives
#"Misses": [2] #Misses
#}

add_probability = {}

non_event_duration = datetime.timedelta(hours=14)
non_event_start = [
        '2011-05-09 20:42:00',
        '2012-03-04 10:29:00',
        '2012-03-05 03:30:00',
        '2012-06-13 11:29:00',
        '2012-06-29 09:13:00',
        '2013-06-07 22:32:00',
        '2013-06-28 01:36:00',
        '2014-08-01 18:00:00',
        '2014-10-24 07:37:00',
        '2014-11-06 03:32:00',
        '2014-11-07 16:53:00',
        '2014-12-17 04:25:00',
        '2014-12-18 21:41:00',
        '2015-03-09 23:29:00',
        '2016-07-23 05:00:00',
        '2021-11-01 00:57:00',
        '2021-11-02 02:03:00',
        '2022-01-18 17:01:00',
        '2022-04-17 03:17:00',
        '2022-04-20 03:41:00',
        '2022-04-29 07:15:00',
        '2022-05-25 18:12:00',
        '2022-08-17 13:26:00',
        '2022-08-18 10:37:00',
        '2022-08-19 04:14:00',
        '2022-08-29 16:15:00',
        '2022-08-30 18:05:00',
        '2022-12-01 07:04:00',
        '2023-03-04 15:19:00',
        '2023-03-06 02:08:00'
        ]

def read_observed_flux_files(path, energy_key, thresh_key):
    """ Read in all observed flux time profiles that were associated
        with a forecast prediction window from the SPHINX_dataframe.pkl
        file.
        
        INPUT:
        
        :path: (string) path to the output directory with trailing /
            (not including) output/
        :energy_key: (string) energy channel key
        :thresh_key: (string) threshold key
            
        OUTPUT:
        
        :dates: (1xn datetime array) dates
        :fluxes: (1xn floar array) fluxes associated with dates
        
    """

    spx_fname = path + "output/pkl/SPHINX_dataframe.pkl"
    sphinx_df = resume.read_in_df(spx_fname)
    sphinx_df = sphinx_df[(sphinx_df["Energy Channel Key"] == energy_key) & (sphinx_df["Threshold Key"] == thresh_key)]
    
    
    observations = sphinx_df['Observed Time Profile'].to_list()
    #Create list of unique observed time profile filenames
    #(may be repeates in the sphinx dataframe
    tprof = []
    for obsfile in observations:
        obsfile = obsfile.strip().split(",")
        for tp in obsfile:
            if tp not in tprof:
                tprof.append(tp)

    dates = []
    fluxes = []
    for fnm in tprof:
        dt, flx = profile.read_single_time_profile(fnm)
        if dt == []:
            continue
        dates.extend(dt)
        fluxes.extend(flx)

    return dates, fluxes


def export_all_clear_incorrect(filename, threshold, doplot=False):
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
        
    model = df["Model"].iloc[0]
    energy_key = df["Energy Channel Key"].iloc[0]
    thresh_key = df["Threshold Key"].iloc[0]

    #Correct Predictions
    cn_dates = []
    cn_fluxes = []
    sub = df.loc[(df["Observed SEP All Clear"] == True) & (df["Predicted SEP All Clear"] == True)]
    if sub.empty:
        print("post_analysis: export_all_clear_incorrect: No correct negatives identified.")
    else:
        cn_dates = sub["Prediction Window Start"].to_list()
        cn_fluxes = [threshold]*len(cn_dates)

    #Hits
    hits_dates = []
    hits_fluxes = []
    sub = df.loc[(df["Observed SEP All Clear"] == False) & (df["Predicted SEP All Clear"] == False)]
    if sub.empty:
        print("post_analysis: export_all_clear_incorrect: No hits.")
    else:
        hits_dates = sub["Prediction Window Start"].to_list()
        hits_fluxes = [threshold]*len(hits_dates)



    #False Alarms
    fa_dates = []
    fa_fluxes = []
    fa_sub = df.loc[(df["Observed SEP All Clear"] == True) & (df["Predicted SEP All Clear"] == False)]
    
    if fa_sub.empty:
        print("post_analysis: export_all_clear_incorrect: No false alarms identified.")
    else:
        fa_dates = fa_sub["Prediction Window Start"].to_list()
        fa_fluxes = [threshold+2]*len(fa_dates)

        fname = filename.replace(".pkl","_false_alarms.csv")
        fname = fname.replace("pkl","csv")
        
        #Write false alarms out to csv file
        fa_sub.to_csv(fname)


    #Misses
    miss_dates = []
    miss_fluxes = []
    miss_sub = df.loc[(df["Observed SEP All Clear"] == False) & (df["Predicted SEP All Clear"] == True)]
    
    if miss_sub.empty:
        print("post_analysis: export_all_clear_incorrect: No misses identified.")
    else:
        miss_dates = miss_sub["Prediction Window Start"].to_list()
        miss_fluxes = [threshold-2]*len(miss_dates)

        fname = filename.replace(".pkl","_misses.csv")
        fname = fname.replace("pkl","csv")
        
        #Write false alarms out to csv file
        miss_sub.to_csv(fname)



    if doplot:
        #Read in observed time profiles to plot with the forecasts
        path = filename.strip().split("output")[0]
        obs_dates, obs_fluxes = read_observed_flux_files(path, energy_key, thresh_key)
        
        figname = filename.replace(".pkl","_incorrect.png")
        figname = figname.replace("pkl","plots")
        
        title = "All Clear " + model + " (" + energy_key + ", " + thresh_key +")"
        
        mismatch = df["Mismatch Allowed"].iloc[0]
        if mismatch:
            pred_energy_channel = df["Prediction Energy Channel Key"].iloc[0]
            pred_thresh_key = df["Prediction Threshold Key"].iloc[0]
            title = "All Clear " + model + " (Observations: " + energy_key \
                    + ", " + thresh_key +" and "  + " Predictions: " \
                    + pred_energy_channel + ", " + pred_thresh_key +")"
        
        labels = ["Observed Flux", "Hits", "Correct Negatives", "False Alarms", "Misses"]
        fig, _ = plt_tools.plot_flux_false_alarms(obs_dates, obs_fluxes,
            hits_dates, hits_fluxes, cn_dates, cn_fluxes, fa_dates, fa_fluxes,
            miss_dates, miss_fluxes, labels, threshold,
            x_label="Date", y_label="", date_format="Year", title=title,
            figname=figname, saveplot=True, showplot=True)
        


def export_max_flux_incorrect(filename, threshold, doplot=False):
    """ Provide the filename of an max_flux_in_pred_win_selections_*.pkl
        file.
        
        Select cases where observed max flux in the prediction window is below
        threshold and predicted max flux is above threshold.
        
        Output as a csv file.
        Plot all forecasts with time with the False Alarms highlighted.
        
        INPUT:
        
        :filename: (string) name of all_clear_selections_*.pkl
            file. Full path.
        :threshold: (float) flux threshold
            
        :doplot: (bool) set to True to plot false alarms with time
            
        OUTPUT:
        
        Write out csv file with false alarms.
        Create plot with distribution of times between false alarms.
        
    """
    
    df = resume.read_in_df(filename)
    
    energy_key = resume.identify_unique(df, "Energy Channel Key")[0]
    thresh_key = resume.identify_unique(df, "Threshold Key")[0]
    
    if df.empty:
        print("post_analysis: export_max_flux_incorrect: Dataframe empty. Returning.")
        return

    #Could have a column with "Predicted SEP Peak Intensity (Onset Peak)" or
    #"Predicted SEP Peak Intensity Max (Max Flux)"
    pred_col = None
    columns = df.columns.to_list()
    for col in columns:
        if "Units" in col:
            continue
        if "Predicted SEP Peak Intensity" in col:
            pred_col = col
            print("Predicted column is " + pred_col)
    

    #Correct Predictions
    cn_dates = []
    cn_fluxes = []
    #Correct negatives
    sub = df[(df["Observed Max Flux in Prediction Window"] < threshold) & (df[pred_col] < threshold)]
    
    if sub.empty:
        print("post_analysis: export_max_flux_incorrect: No correct negatives identified.")
    else:
        cn_dates = sub["Prediction Window Start"].to_list()
        cn_fluxes = sub[pred_col].to_list()


    #Hits
    hits_dates = []
    hits_fluxes = []
    sub = df[(df["Observed Max Flux in Prediction Window"] >= threshold) & (df[pred_col] >= threshold)]
    
    if sub.empty:
        print("post_analysis: export_max_flux_incorrect: No hits identified.")
    else:
        hits_dates= sub["Prediction Window Start"].to_list()
        hits_fluxes = sub[pred_col].to_list()


    #False Alarms
    fa_dates = []
    fa_fluxes = []
    fa_sub = df[(df["Observed Max Flux in Prediction Window"] < threshold) & (df[pred_col] >= threshold)]
    
    if fa_sub.empty:
        print("post_analysis: export_max_flux_incorrect: No false alarms identified.")
    else:
        fa_dates = fa_sub["Prediction Window Start"].to_list()
        fa_fluxes = fa_sub[pred_col].to_list()
        
        fafname = filename.replace(".pkl","_false_alarms.csv")
        fafname = fafname.replace("pkl","csv")
        
        #Write false alarms out to csv file
        fa_sub.to_csv(fafname)
    
    
    #Misses
    miss_dates = []
    miss_fluxes = []
    miss_sub = df[(df["Observed Max Flux in Prediction Window"] >= threshold) & (df[pred_col] < threshold)]
 
    if miss_sub.empty:
        print("post_analysis: export_max_flux_incorrect: No misses identified.")
    else:
        miss_dates = miss_sub["Prediction Window Start"].to_list()
        miss_fluxes = miss_sub[pred_col].to_list()
        
        mfname = filename.replace(".pkl","_misses.csv")
        mfname = mfname.replace("pkl","csv")

        #Write misses out to csv file
        miss_sub.to_csv(mfname)



    if doplot:
        figname = filename.replace(".pkl","_Outcomes.png")
        figname = figname.replace("pkl","plots")
                

        #Read in observed time profiles to plot with the forecasts
        path = filename.strip().split("output")[0]
        obs_dates, obs_fluxes = read_observed_flux_files(path, energy_key, thresh_key)
        
        model = df["Model"].iloc[0]
        
        title = "Max Flux " + model + " (" + energy_key + ", " + thresh_key +")"
        
        mismatch = df["Mismatch Allowed"].iloc[0]
        if mismatch:
            pred_energy_channel = df["Prediction Energy Channel Key"].iloc[0]
            pred_thresh_key = df["Prediction Threshold Key"].iloc[0]
            title = model + " False Alarms (Observations: " + energy_key \
                    + ", " + thresh_key +" and "  + " Predictions: " \
                    + pred_energy_channel + ", " + pred_thresh_key +")"
        
        labels = ["Observed Flux", "Hits", "Correct Negatives", "False Alarms", "Misses"]
        fig, _ = plt_tools.plot_flux_false_alarms(obs_dates, obs_fluxes,
            hits_dates, hits_fluxes, cn_dates, cn_fluxes, fa_dates, fa_fluxes,
            miss_dates, miss_fluxes, labels, threshold,
            x_label="Date", y_label="", date_format="Year", title=title,
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
    


def read_in_metrics(path, quantity, include, exclude):
    """ Read in metrics files related to specfied quantity.
    
    INPUT:
    
    :path: (string) location of the output/ folder
    :quantity: (string) Forecasted quantity of interest.
    :exclude: (array of strings) names or partial names of models
        to exclude from the metrics post analysis
    
    OUTPUT:
    
    :df: (pandas DataFrame) dataframe containing all the metrics
    
    """
    
    prefix = get_file_prefix(quantity)
    fname = path + "output/pkl/" + prefix + "_metrics.pkl"
    print("read_in_metrics: Reading in " + fname)
    
    df = resume.read_in_df(fname)
    
    #This is a little tricky because a part of a model
    #short_name might be in include. For example, to
    #include all 30 of SAWS-ASPECS flavors, the user would
    #simply have to put "ASPECS" in include.
    #So need to check if the substring is in any of the
    #model names. If not, then will append the model name
    #to the exclude array and remove from the data frame.
    if include[0] != 'All':
        models = resume.identify_unique(df,'Model')
        for model in models:
            included = False
            for incl_model in include:
                if incl_model in model:
                    included = True
            if not included:
                exclude.append(model)

    #Remove model results that should be excluded from the plots
    for model in exclude:
        if model != '':
            model = model.replace('+','\+')
            model = model.replace('(','\(')
            model = model.replace(')','\)')
            
            #Avoid removing an included model that contains an excluded
            #substring
            included_model = ''
            for incl_model in include:
                if model in incl_model:
                    included_model = incl_model
            
            if included_model != '':
                df = df[(~df['Model'].str.contains(model) | df['Model'].str.contains(included_model))]
            else:
                df = df[~df['Model'].str.contains(model)]
            print("read_in_metrics: Removed model metrics for " + model)

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
                    ["Gilbert Skill Score", "True Skill Statistic",
                    "Heidke Skill Score", "Odds Ratio Skill Score",
                    "Symmetric Extreme Dependency Score"],
                    ["Number SEP Events Correctly Predicted",
                    "Number SEP Events Missed"],
                    ["Odds Ratio"],
                    ["Hit Rate", "False Alarm Ratio", "Bias",
                    "True Skill Statistic", "Heidke Skill Score"] #RE
                ]

    #PROBABILITY
    if quantity == "Probability":
        groups = [["Brier Score", "Brier Skill Score", "Area Under ROC Curve"]]

    #FLUX METRICS
    flux_types = ["Onset Peak", "Max Flux", "Fluence",
                "Max Flux in Prediction Window", "Time Profile"]
    if quantity in flux_types:
        groups = [ ["Linear Regression Slope",
                    "Pearson Correlation Coefficient (Linear)",
                    "Pearson Correlation Coefficient (Log)",
                    "Spearman Correlation Coefficient (Linear)"],
                    ["Mean Error (ME)", "Median Error (MedE)"],
                    ["Mean Absolute Error (MAE)",
                    "Median Absolute Error (MedAE)",
                    "Root Mean Square Error (RMSE)"],
                    ["Mean Log Error (MLE)", "Median Log Error (MedLE)"],
                    ["Mean Absolute Log Error (MALE)",
                    "Median Absolute Log Error (MedALE)",
                    "Root Mean Square Log Error (RMSLE)"],
                    ["Mean Percent Error (MPE)",
                    "Mean Symmetric Percent Error (MSPE)",
                    "Mean Symmetric Absolute Percent Error (SMAPE)"],
                    ["Mean Absolute Percent Error (MAPE)",
                    "Median Symmetric Accuracy (MdSA)",
                    "Mean Accuracy Ratio (MAR)"]
                ]

    #TIME METRICS
    time_types = ["Threshold Crossing Time", "Start Time", "End Time",
                "Onset Peak Time", "Max Flux Time"]
    if quantity in time_types:
        groups = []

    return groups


def add_to_all_clear(df):
    """ Add more lines to the all clear metrics dataframe if the
        add_contingency dict above is populated.
        
        INPUT:
        
            :df: (pandas DataFrame) contains all clear metrics
            
        Output:
        
            None but df is updated with more rows
        
    """
    if add_contingency == {}:
        return df
        
    dict = validation.initialize_all_clear_dict()
        
    n = len(add_contingency['Model'])
    for i in range(n):
        model = add_contingency['Model'][i]
        energy_key = add_contingency['Energy Channel'][i]
        thresh_key = add_contingency['Threshold'][i]
        pred_energy_key = add_contingency['Prediction Energy Channel'][i]
        pred_thresh_key = add_contingency['Prediction Threshold'][i]
        h = add_contingency['Hits'][i]
        m = add_contingency['Misses'][i]
        f = add_contingency['False Alarms'][i]
        c = add_contingency['Correct Negatives'][i]
    
        scores = metrics.contingency_scores(h,m,f,c)
    
        print("post_analysis: add_to_all_clear: Adding " + model + " for " + energy_key + " and " + thresh_key + " to all clear plots:")
        print(scores)
    
        validation.fill_all_clear_dict(dict, model, energy_key, thresh_key, pred_energy_key,
        pred_thresh_key, scores, h, 'Not provided', m, 'Not provided')
    
    
    add_df = pd.DataFrame(dict)
    df = pd.concat([df,add_df], ignore_index=True)
    
    
    return df


def max_prob_per_time_period(df, model, energy_key, thresh_key, starttime, endtime):
    """ Given a probability_selections df, find the maximum
        probability issued by the model between the
        starttime and endtime.
        
    """
    sub  = df[(df['Model'] == model) & (df['Energy Channel Key'] == energy_key)
            & (df['Threshold Key'] == thresh_key)]
    
    sub = sub[(sub['Prediction Window End'] >= starttime) & (sub['Prediction Window Start'] < endtime)]
    
    if sub.empty:
        return None
    
    row = validation.identify_max_forecast(sub,'Predicted SEP Probability')
    
    return row
    

def calculate_probability_metrics(df, dict, path, model, energy_key, thresh_fnm):

    obs = df['Observed SEP Probability'].to_list()
    pred = df['Predicted SEP Probability'].to_list()

    #Calculate metrics
    brier_score = metrics.calc_brier(obs, pred)
    brier_skill = metrics.calc_brier_skill(obs, pred)
    rank_corr_coeff = metrics.calc_spearman(obs, pred)

    roc_auc, roc_curve_plt = metrics.receiver_operator_characteristic(obs, pred, model)
    
    roc_curve_plt.plot()
    skill_line = np.linspace(0.0, 1.0, num=10) # Constructing a diagonal line that represents no skill/random guess
    plt.plot(skill_line, skill_line, '--', label = 'Random Guess')
    figname = path + '/summary/ROC_curve_' \
            + model + "_" + energy_key.strip() + "_" + thresh_fnm
#    if mismatch:
#            figname = figname + "_mm"
#    if validation_type != "" and validation_type != "All":
#            figname = figname + "_" + validation_type
    figname += "_Max_All.pdf"
    plt.legend(loc="lower right")
    roc_curve_plt.figure_.savefig(figname, dpi=300, bbox_inches='tight')
    plt.close(roc_curve_plt.figure_)


    #Save to dict (ultimately dataframe)
    dict['Model'].append(df['Model'].iloc[0])
    dict['Energy Channel'].append(df['Energy Channel Key'].iloc[0])
    dict['Threshold'].append(df['Threshold Key'].iloc[0])
    dict['Prediction Energy Channel'].append(df['Prediction Energy Channel Key'].iloc[0])
    dict['Prediction Threshold'].append(df['Prediction Threshold Key'].iloc[0])
    dict['ROC Curve Plot'].append(figname)
    dict['Brier Score'].append(brier_score)
    dict['Brier Skill Score'].append(brier_skill)
    dict['Spearman Correlation Coefficient'].append(rank_corr_coeff)
    dict['Area Under ROC Curve'].append(roc_auc)




def get_max_probabilities(df, path):
    """ Models that produce more than one probability forecast per
        SEP event or non-event period will have a file named
        probability_selections_*_Max.pkl or csv.
        This Max file only contains the maximum probability issued
        for each observed SEP event that fell within the prediction
        windows. Note that only models that made more than one prediction
        for an SEP event will be in the Max file. The full list of
        models should come from the full probability dataframe.
        
        SPHINX does not know about non-event periods. The array above
        called non_event_start lists the flare start time for all
        SEPVAL non-event periods. The non_event_duration indicates
        how long after the flare start should be considered to assess
        probability predictions.
        
        This code will identify the maximum probability for forecasts
        between non_event_start + non_event_duration and write out
        to the probability_selections_*_Max_non_event.csv file.
        
        The probability metrics will be recalculated and saved to file.
        
    """
    columns = df.columns.to_list()
    all_max_metrics_df = pd.DataFrame(columns=columns)
    dict = validation.initialize_probability_dict()
    
    all_models = df['Model'].to_list()
    
    #Check for a probability_selections_*_Max.pkl file. If exists,
    #then need to add max probability for non-events and recalculate
    #metrics.
    #If doesn't exist, then keep metrics as-is.
    for i in range(len(df)):
        metrics_df = pd.DataFrame(columns = columns)
        model = df['Model'].iloc[i]
        energy_key = df['Energy Channel'].iloc[i]
        thresh_key = df['Threshold'].iloc[i]
        
        thresh_fnm = validation.make_thresh_fname(thresh_key)
        fnm = "probability_selections_" + model + "_" + energy_key.strip() \
            + "_" + thresh_fnm
        
        all_fname = path + "output/pkl/" + fnm + ".pkl"
        max_fname = path + "output/pkl/" + fnm + "_Max.pkl"

 
        #If no Max file, then no need to recalculate metrics, just
        #save already calculated metrics
        if not os.path.exists(max_fname):
            all_max_metrics_df[len(all_max_metrics_df)] = df.iloc[i]
            continue
    
        #Read in the maximum values per SEP event
        max_event_df = resume.read_in_df(max_fname)

        #Need to add in the maximum values per non-event
        selections_df = resume.read_in_df(all_fname)
        
        for date in non_event_start:
            starttime = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
            endtime = starttime + non_event_duration
            max_row = max_prob_per_time_period(selections_df, model,
                    energy_key, thresh_key, starttime, endtime)
            max_event_df.loc[len(max_event_df)] = max_row
        
        
        #Write out selections file with SEP event and non-event periods.
        max_all_fname = path + "output/csv/" + fnm + "_Max_all.csv"
        max_event_df.to_csv(max_all_fname)
        
        
        #Clear the dataframe
        #Find predicted None values
        noneval = pd.isna(max_event_df['Predicted SEP Probability'])
        #Extract only indices for Nones
        #True indicates that peak intensity was a None value
        noneval = noneval.loc[noneval == True]
        noneval = noneval.index.to_list()
        if len(noneval) > 0:
            for ix in noneval:
                max_event_df = max_event_df.drop(index=ix)

        if not max_event_df.empty:
            #Find predicted None values
            noneval = pd.isna(max_event_df['Observed SEP Probability'])
            #Extract only indices for Nones
            #True indicates that peak intensity was a None value
            noneval = noneval.loc[noneval == True]
            noneval = noneval.index.to_list()
            if len(noneval) > 0:
                for ix in noneval:
                    max_event_df = max_event_df.drop(index=ix)

        
        
        #Now have maximum probability for all SEP event and non_event
        #time periods. Need to recalculate metrics.
        calculate_probability_metrics(max_event_df, dict, path,
            model, energy_key, thresh_fnm)
        
        #Add the newly calculated metrics for the max of all SEP event
        #and non-event periods for the give model, energy channel, and thresh
        metrics_df = pd.concat([metrics_df, pd.DataFrame(dict)],
            ignore_index=True)
    
    all_max_metrics_df = pd.concat([all_max_metrics_df, metrics_df], ignore_index=True)
    
    return all_max_metrics_df



def make_box_plots(df, path, quantity, anonymous, highlight, scoreboard,
    saveplot, showplot):
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
        path/output/output/plots/. directory
    
    """
    if quantity == 'All Clear':
        #Add metrics hard coded at the top of this code
        df = add_to_all_clear(df)
        fname = path + ""
        prefix = get_file_prefix(quantity)
        fname = path + "output/pkl/" + prefix + "_metrics_plotted.pkl"
        fname = fname.replace("pkl","csv")
        print("make_box_plots: Writing " + fname)
        df.to_csv(fname)

    if quantity == "Probability":
        #Get maximum probability for non-event time periods for models
        #that produce high cadence forecasts.
        #If models produce only a single forecast for an event period,
        #these will also be included
        df = get_max_probabilities(df, path)
        fname = path + ""
        prefix = get_file_prefix(quantity)
        fname = path + "output/pkl/" + prefix + "_metrics_plotted.pkl"
        fname = fname.replace("pkl","csv")
        print("make_box_plots: Writing " + fname)
        df.to_csv(fname)


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
                hghlt = ''
                for metric_col in group:
                    vals = sub[metric_col].to_list()
                    if metric_col in cfg.in_percent:
                        vals = [x*100. for x in vals]


                    model_list = sub['Model'].to_list()
                    nfcasts = []
                    if 'N (Total Number of Forecasts)' in sub.columns.to_list():
                        nfcasts = sub['N (Total Number of Forecasts)'].to_list()


                    #--- Adjust names of models in lists as needed --------
                    #ANONYMOUS
                    if anonymous and highlight == '':
                        for j in range(len(model_list)):
                            model_list[j] = "Model " + str(j)

                    for jj in range(len(nfcasts)):
                        model_list[jj] += " (" + str(nfcasts[jj]) + ")"


                    #HIGHLIGHTED MODEL
                    if highlight != '':
                        in_list = False
                        for j in range(len(model_list)):
                            if highlight in model_list[j]:
                                in_list = True
                                continue
                            else:
                                model_list[j] = "Models"
                    
                        #Only include the plots where the highlighted model
                        #is in the model list
                        if in_list:
                            values.extend(vals)
                            metric_names.extend([metric_col]*len(vals))
                            model_names.extend(model_list)


                    #SCOREBOARD MODELS
                    #Highlight only the models on the SEP Scoreboards
                    if scoreboard:
                         for j in range(len(model_list)):
                            in_list = False
                            for mod in scoreboard_models:
                                if mod in model_list[j] and 'electrons' not in model_list[j]:
                                    in_list = True

                            if in_list:
                                model_list[j] = "SEP Scoreboards"
                            else:
                                model_list[j] = "Models"
                    
                    #If no model is highlighted, then make all the plots
                    if highlight == '':
                        values.extend(vals)
                        metric_names.extend([metric_col]*len(vals))
                        model_names.extend(model_list)




                dict = {"Metrics": metric_names, "Models":model_names,
                        "Values":values}
                metrics_df = pd.DataFrame(dict)
                
          
                title = quantity + " Group " + str(grp) + " (" + ek + ", " + tk + ")"
                figname = path + "/summary/" + quantity + "_" + ek  \
                        + "_boxes_Group" + str(grp)
                if highlight != '':
                    figname += "_" + highlight
                if scoreboard:
                    figname += "_Scoreboards"
                if anonymous:
                    figname += "_anon"
                plt_tools.box_plot_metrics(metrics_df, group, highlight,
                    x_label="Metric", y_label="Value", title=title,
                    save=figname, uselog=False, showplot=showplot, \
                    closeplot=False, saveplot=saveplot)


############# SPHINX OUTPUT STATS ############
def forecast_coverage(df, start_date, end_date):
    """ Daily coverage of forecasts. 
        Counts the number of days in the total time range where forecasts
        were issued. Returns the number of days no forecasts were issued
        
        INPUT: 
        
            :df: (pandas DataFrame) SPHINX dataframe for only one model, 
                (an probabily energy channel, and threshold combination)
            :start_date: (datetime) Start of date range checking for coverage
            :end_date: (datetime) End of date range checking for coverage
                
        OUTPUT:
        
            :N_no_data: (int) number of days with no forecasts
            :N_data: (int) number of days with forecasts
        
    """

    sub = df.loc[(df['Prediction Window Start'] >= start_date) & (df['Prediction Window End'] <= end_date)]
    td24 = datetime.timedelta(hours=24)
    
    start_day = datetime.datetime(start_date.year,start_date.month,start_date.day)
    end_day = datetime.datetime(end_date.year,end_date.month,end_date.day)
    Ndays = int((end_day - start_day)/td24) + 1 #+1 to include the end_day
    
    N_no_data = 0
    N_data = 0

    for i in range(Ndays):
        check_day = start_day + i*td24
        check_day_end = check_day + td24
        sub_range = sub.loc[(sub['Prediction Window Start'] >= check_day) & (sub['Prediction Window Start'] < check_day_end)]
        
        if sub_range.empty:
            N_no_data += 1
        else:
            N_data += 1
        
    return N_no_data, N_data



def forecast_stats(fname):
    """ Read in a SPHINX dataframe (SPHINX_evaluated, SPHINX_removed) and 
        calculate stats related to the forecasts within.
        
        INPUT:
        
            :fname: csv file for a SPHINX dataframe
            
        OUTPUT:
        
            Stats file written out to same directory of the dataframe.
        
    """
    
    dict = {'Model': [],
            'Energy Channel': [],
            'Threshold': [],
            'Prediction Energy Channel': [],
            'Prediction Threshold': [],
            'First Issue Time': [],
            'Last Issue Time': [],
            'N Issue Time Days': [],
            'First Prediction Window Start': [],
            'Last Prediction Window End': [],
            'N Prediction Days': [],
            'N Days with Forecasts': [],
            'N Days without Forecasts': [],
            'N Days with SEP Event Onsets': [],
            'Imbalance (Days)': [],
            'Total Number of Forecasts': [],
            'Total Number of Forecasts with SEP Events': [],
            'Imbalance (Forecasts)': [],
            }
    
    df = pd.read_csv(fname, parse_dates=['Forecast Issue Time', 'Prediction Window Start',
                            'Prediction Window End', 'Observed SEP Threshold Crossing Time',
                            'Observed SEP End Time'])
                            
    models = resume.identify_unique(df, 'Model')
    
    #Extract all forecasts per model
    for model in models:
        df_model = df.loc[df['Model'] == model]
        energy_keys = resume.identify_unique(df_model, 'Energy Channel Key')
 
        #Extract all forecasts per energy channel
        for ek in energy_keys:
            sub = df_model.loc[(df_model['Energy Channel Key'] == ek)]
            if sub.empty: continue
            thresholds = resume.identify_unique(sub, 'Threshold Key')

            for tk in thresholds:
                sub = df_model.loc[(df_model['Energy Channel Key'] == ek) & (df_model['Threshold Key'] == tk)]
                if sub.empty: continue
                pred_energy_keys = resume.identify_unique(sub, 'Prediction Energy Channel Key')
                
                for pek in pred_energy_keys:
                    sub = df_model.loc[(df_model['Energy Channel Key'] == ek) & (df_model['Threshold Key'] == tk)
                            & (df_model['Prediction Energy Channel Key'] == pek)]
                    if sub.empty: continue
                    pred_thresholds = resume.identify_unique(sub, 'Prediction Threshold Key')
                    
                    for ptk in pred_thresholds:
                        sub = df_model.loc[(df_model['Energy Channel Key'] == ek) & (df_model['Threshold Key'] == tk)
                            & (df_model['Prediction Energy Channel Key'] == pek) & (df_model['Prediction Threshold Key'] == ptk)]
                        if sub.empty: continue
                        
                        #Calculate stats for unique model configuration
                        td24 = datetime.timedelta(hours=24)
                        
                        #Total number of days producing forecasts assuming real time forecasting
                        #starts with first timestamp and finishes at last timestamp
                        first_pred_win_st = sub['Prediction Window Start'].min()
                        last_pred_win_end = sub['Prediction Window End'].max()
                        Ndays = math.ceil((last_pred_win_end - first_pred_win_st)/td24)
                        
                        #Total number of days using issue time
                        issue_st = sub['Forecast Issue Time'].min()
                        issue_end = sub['Forecast Issue Time'].max()
                        Ndays_issue = math.ceil((issue_end - issue_st)/td24)
                        
                        #Total number of Forecasts
                        Nforecasts = len(sub)
                        
                        #Number of days with Forecasts issued and without Forecasts issued
                        N_no_data, N_data = forecast_coverage(sub, first_pred_win_st, last_pred_win_end)
                        
                        #Number of SEP events
                        sub_sep = sub.dropna(subset=['Observed SEP Threshold Crossing Time'])
                        Nsep_forecasts = len(sub_sep)
                        sep = resume.identify_unique(sub, 'Observed SEP Threshold Crossing Time')
                        Nsep = len(sep)
                        
                        #Imbalance
                        if Nsep_forecasts != 0:
                            fcast_imbalance = (Nforecasts - Nsep_forecasts)/Nsep_forecasts
                        else:
                            fcast_imbalance = 0
                            
                        if Nsep != 0:
                            days_imbalance = (N_data - Nsep)/Nsep
                        else:
                            days_imbalance = 0
                
#                        print(f"Model: {model} \n"
#                              f"Energy Channel: {ek} \n"
#                              f"Threshold: {tk} \n"
#                              f"Prediction Energy Channel: {pek} \n"
#                              f"Prediction Threshold: {ptk} \n"
#                              f"First Issue Time: {issue_st} \n"
#                              f"Last Issue Time: {issue_end} \n"
#                              f"N Issue Time Days: {Ndays_issue} \n"
#                              f"First Prediction Window Start: {first_pred_win_st} \n"
#                              f"Last Prediction Window End: {last_pred_win_end} \n"
#                              f"N Prediction Days: {Ndays} \n"
#                              f"N Days with Forecasts: {N_data} \n"
#                              f"N Days without Forecasts: {N_no_data} \n"
#                              f"N Days with SEP Event Onsets: {Nsep} \n"
#                              f"Imbalance (Days): {days_imbalance} \n"
#                              f"Total Number of Forecasts: {Nforecasts} \n"
#                              f"Total Number of Forecasts with SEP Events: {Nsep_forecasts} \n"
#                              f"Imbalance (Forecasts): {fcast_imbalance} \n" )
                              
                
                
                        dict['Model'].append(model)
                        dict['Energy Channel'].append(ek)
                        dict['Threshold'].append(tk)
                        dict['Prediction Energy Channel'].append(pek)
                        dict['Prediction Threshold'].append(ptk)
                        dict['First Prediction Window Start'].append(first_pred_win_st)
                        dict['Last Prediction Window End'].append(last_pred_win_end)
                        dict['N Prediction Days'].append(Ndays)
                        dict['First Issue Time'].append(issue_st)
                        dict['Last Issue Time'].append(issue_end)
                        dict['N Issue Time Days'].append(Ndays_issue)
                        dict['N Days with Forecasts'].append(N_data)
                        dict['N Days without Forecasts'].append(N_no_data)
                        dict['N Days with SEP Event Onsets'].append(Nsep)
                        dict['Imbalance (Days)'].append(days_imbalance)
                        dict['Total Number of Forecasts'].append(Nforecasts)
                        dict['Total Number of Forecasts with SEP Events'].append(Nsep_forecasts)
                        dict['Imbalance (Forecasts)'].append(fcast_imbalance)
                        

    df_stats = pd.DataFrame(dict)
    df_stats = df_stats.sort_values(by=['Model', 'Energy Channel'])
    outfname = fname.split('.csv')
    outfname = outfname[0] + '_stats.csv'
    df_stats.to_csv(outfname, index=False)


############ DEOVERLAPPING ###################

def create_date_range(first_date, last_date, td = datetime.timedelta(hours=24)):
    """ Create a date range between first_date and last_date with timedelta
        of td, default 24 hours.
        
        INPUTS:
        
            :first_date: (datetime) starting time for date range
            :last_date: (datetime) ending time for date range
            :td: (datetime timedelta) duration of each time range
            
        OUTPUT:
        
            :date_range_st: (datetime list) starting date of each time period
                in the range
            :date_range_end: (datetime list) ending date of each time period 
                in the range. e.g. the first period is:
                date_range_st[0] to date_range_end[0] 
                Also date_range_end[0] == date_range_st[1], but two lists are
                output for convenience

    """

    #Specify date range covered by prediction windows
    start_date = pd.Timestamp(year=first_date.year, month=first_date.month, day=first_date.day)
    end_date = pd.Timestamp(year=last_date.year, month=last_date.month, day=last_date.day)
        
    #Create a range of daily time stamps from the start date to the end date
    date_range_st = pd.date_range(start=start_date, end=end_date-td, freq=td)
    date_range_end = date_range_st + td


    date_range_st = date_range_st.to_list()
    date_range_end = date_range_end.to_list()
    
    return date_range_st, date_range_end


def create_date_range_df(dates_file, seps, nonevent_st, nonevent_end):
    """ Contains dates of SEP events and specific non-event time periods.
        SEP Event, Non-Event Start, Non-Event End
        
    """
    df_dates = pd.DataFrame()
    if dates_file != '':
        df_dates = pd.read_csv(dates_file, parse_dates=['SEP Events','Non-Event Start','Non-Event End'])
        print(f"all_clear_grid: Read in dates from file {dates_file}.")
    else:
        print(f"all_clear_grid: Using input date range for seps and nonevents.")
        #seps and nonevent arrays have to be same length to create a dataframe
        if len(seps) > len(nonevent_st):
            nmore = len(seps) - len(nonevent_st)
            nonevent_st = nonevent_st + [pd.NaT]*nmore
            nonevent_end = nonevent_end + [pd.NaT]*nmore
        if len(seps) < len(nonevent_st):
            nmore = len(nonevent_st) - len(seps)
            seps = seps + [pd.NaT]*nmore
        temp = {'SEP Events': seps, 'Non-Event Start': nonevent_st, 'Non-Event End': nonevent_end}
        df_dates = pd.DataFrame(temp)
        df_dates['SEP Events'] = pd.to_datetime(df_dates['SEP Events'])
        df_dates['Non-Event Start'] = pd.to_datetime(df_dates['Non-Event Start'])
        df_dates['Non-Event End'] = pd.to_datetime(df_dates['Non-Event End'])

    return df_dates



def associated_forecasts(df, date_st, date_end, split):
    """ Extract a sub-dictionary of only the forecasts associated with
        date_st to date_end.

            :split: (bool) 
                If True, for forecasts with prediction windows that cross dates, 
                    they will be associated with the date with the most overlap 
                    with their prediction window.
                If False, then forecasts with the prediction_window_start inside of 
                    the non-event time period will be evaluated.
    
    """
    sub = pd.DataFrame()

    if not split:
        sub = df[(df['Prediction Window Start'] >= date_st) & (df['Prediction Window Start'] < date_end)]
    
    if split:
        #Prediction window within the date range
        check_in = (df['Prediction Window Start'] >= date_st) & (df['Prediction Window End'] <= date_end)
        
        #Forecasts that start before the date range of interest and overlap
        #more than the previous date are included
        check_pre = ((df['Prediction Window End'] - date_st) > (date_st - df['Prediction Window Start'])) & (df['Prediction Window Start'] < date_st)

        #Forecasts that end after the date of interest and overlap more
        #than the next date are included
        check_post = ((df['Prediction Window End'] - date_end) < (date_end - df['Prediction Window Start'])) & (df['Prediction Window End'] > date_end)
        
        #Forecasts with prediction window extending beyond both sides of
        #the date range; if prediction window much larger than td,
        #then won't get any matches
#        check_overlap = ((sub['Prediction Window Start'] < start) & (sub['Prediction Window End'] > end)) & ((end-start) > (start - sub['Prediction Window Start'])) & ((end-start) > (sub['Prediction Window End']-end))
        
        keep = check_in | check_pre | check_post
        
        sub = df.loc[keep]

    return sub


def all_clear_any(df, obs, pred):
    """ Returns True if any of the entries where the observed
        condition is obs have a prediction condition pred.
        
        e.g. Hits
        obs = False
        pred = False
        
        if any entries in df that have observed all clear = False and 
        predicted all clear = False, then will return True that this
        set of forecasts was a "Hit"

        e.g. False Alarms
        obs = True
        pred = False
        
        if all entries in df that have observed all clear = True and 
        predicted all clear = False, then will return True that this
        set of forecasts was a "False Alarm"        
        
        
        INPUT:
        
            :df: (dataframe) contains all clear observed and predicted outcomes
                for each forecast
            :obs: (bool) desired observational condition
            :pred: (bool) desired predicted condition
            
        Output:
        
            :condition: (bool) True if condition is met, False is condition not met
        
    """

    sub = df.loc[(df['Observed SEP All Clear'] == obs)]
    
    #If no observations meet the condition
    if sub.empty:
        return None
        
    sub = sub.loc[(sub['Predicted SEP All Clear'] == pred)]
    
    #If no predictions meet the condition
    if sub.empty:
        return False
    #if any of the predictions meet the condition
    else:
        return True
            


def all_clear_all(df, obs, pred):
    """ Returns True if all of the entries where the observed
        condition is obs have a prediction condition pred.
        
        e.g. Misses
        obs = False
        pred = True
        
        if all entries in df that have observed all clear = False and 
        predicted all clear = True, then will return True that this
        set of forecasts was a "Miss"

        e.g. Correct Negatives
        obs = True
        pred = True
        
        if all entries in df have observed all clear = True and 
        predicted all clear = True, then will return True that this
        set of forecasts was a "Correct Negative"        
        
        
        INPUT:
        
            :df: (dataframe) contains all clear observed and predicted outcomes
                for each forecast
            :obs: (bool) desired observational condition
            :pred: (bool) desired predicted condition
            
        Output:
        
            :condition: (bool) True if condition is met, False is condition not met
        
    """

    sub = df.loc[(df['Observed SEP All Clear'] == obs)]
    
    #If no observations meet the condition
    if sub.empty:
        return None
        
    sub_pred = sub.loc[(sub['Predicted SEP All Clear'] == pred)]
    
    if len(sub) == len(sub_pred):
        return True
    else:
        return False



def all_clear_deoverlap(csv_path, models, energy_min, energy_max, threshold,
    dates_file='',seps=[], nonevent_st=[], nonevent_end=[], split=False,
    write_grid=True):
    """ For a list of models, reads in all_clear_selections files output
        by sphinx and checks whether the model predicted a hit/miss or
        false alarm/correct negative for a set of dates, stored in df_dates.
        
        User may enter a file containing SEP and non-event dates.
        df_dates is expected to consist of two columns, labeled 
        "SEP Events" and "Non-Events".
        
        Or user may enter a list of SEP dates, a list of non-event start times, 
        and a list of non-event end times.
        
        This subroutine uses the original all_clear_selections files.
        
        INPUT:
        
            :csv_path: (string) path the all_clear_selections csv files.
            :models: (list) list of models to be included in the grid; 
                model names must exactly match the short_name field in 
                the forecast jsons or SPHINX output files.
            :dates_file: (string) csv file containing list of dates that want to use to 
                generate grids with a column titled "SEP Events" with the 
                SEP start times and columns titled "Non-Event Start" 
                and "Non-Event End". 
            :energy_min: (float) low edge of energy channel of interest
            :energy_max: (float) high edge of energy channel of interest (-1)
                for infinity (>10 MeV -> energy_min = 10, energy_max = -1)
            :threshold: (float) threshold applied for SEP event definition
                (e.g. >10 MeV exceeds 10 pfu -> threshold = 10)
            :split: (bool) 
                If True, for forecasts with prediction windows that cross dates, 
                    they will be associated with the date with the most overlap 
                    with their prediction window.
                If False, then forecasts with the prediction_window_start inside of 
                    the non-event time period will be evaluated.
            :write_grid: (bool) if True, will write out visual grid csv file
                
        OUTPUT:
        
            Write out two csv files containing desired grid with 
            dates on one axis, models on other axis, and outcomes
            as the entries. "No Data" entries indicate a forecast
            was not provided for a given date.
         
    """
    
    #Find date range that fully spans SEP event and non-event periods for
    #models that need to be deoverlapped.
    print(f"all_clear_grid: Reading in dates file {dates_file}.")
    
    df_dates = create_date_range_df(dates_file, seps, nonevent_st, nonevent_end)


    #Dictionaries needed to create visual grids
    sep_results = {"SEP Events": df_dates["SEP Events"].to_list()}
    nonsep_results = {"Non-Event Start": df_dates["Non-Event Start"].to_list(),
        "Non-Event End": df_dates["Non-Event End"].to_list()}

    #All Clear metrics
    all_clear_dict = validation.initialize_all_clear_dict()

    #Find first and last date out of all the dates
    max_date = df_dates.max().max()
    min_date = df_dates.min().min()
    
    print(f"all_clear_grid: Models will be deoverlapped between {min_date} and {max_date}.")

    energy_key = f"min.{float(energy_min):.1f}.max.{float(energy_max):.1f}.units.MeV"
    print(f"Energy channel: {energy_key}")
    thresh_key = f"threshold_{float(threshold):.1f}"
    print(f"Threshold: {thresh_key}")

    for model in models:
        #Counts of forecasts associated with each date range output to file
        dict = {'Start Date': [], 'End Date': [],
                'Observed SEP Threshold Crossing Time': [],
                'Observed SEP All Clear': [],
                'Predicted SEP All Clear': [],
                'Total Forecasts': [],
                'Total Hits': [],
                'Total Misses': [],
                'Total False Alarms': [],
                'Total Correct Negatives': [],
                'First Prediction Window': [],
                'Last Prediction Window': []}


        #Store Hit/Miss/No Data
        sep_outcomes = []
        #Store False Alarm/Correct Negative/No Data
        nonsep_outcomes = []
        #Calculate skill scores for chosen time periods
        hit = 0
        miss = 0
        fa = 0
        cn = 0
        sep_caught_str = ''
        sep_miss_str = ''
        
        fname = os.path.join(csv_path,
            f"all_clear_selections_{model}_{energy_key}_{thresh_key}.csv")

        if 'UNSPELL' in model: #should be revised to generally account for mismatch
            if energy_min == 10:
                fname = os.path.join(csv_path,
                "all_clear_selections_UNSPELL flare_min.10.0.max.-1.0.units.MeV_min.5.0.max.-1.0.units.MeV_threshold_10.0_mm.csv")
        
        if not os.path.isfile(fname):
            print(f"all_clear_grid: File does not exist. Check model and csv_path. {fname}. Skipping model.")
            continue

        print(f"all_clear_grid: Reading in file {fname}.")
        
        df_ac = pd.read_csv(fname, parse_dates=['Observed SEP Threshold Crossing Time','Prediction Window Start','Prediction Window End'])

        #Date columns indicating SEP range
        key_st = 'Prediction Window Start'
        key_end = 'Prediction Window End'

        #Go through SEP Events and Non-Events and record results
        for sep, non_st, non_end in df_dates.itertuples(index=False):

            #SEP EVENTS
            #For each SEP event, get Hit/Miss/No Data
            sep_outcome = None
            sep_outcome_bool = None
            nsepcasts = np.nan
            nhits = np.nan
            nmiss = np.nan
            nnoncasts = np.nan
            nfa = np.nan
            ncn = np.nan
            if not pd.isnull(sep):
                #Get results for a single SEP event
                sub = df_ac[df_ac['Observed SEP Threshold Crossing Time'] == sep]
                
                if sub.empty:
                    sep_outcome = 'No Data'
                    #Record info for deoverlapping
                    dict['Start Date'].append(sep)
                    dict['End Date'].append(pd.NaT)
                    dict['Observed SEP Threshold Crossing Time'].append(sep)
                    dict['Observed SEP All Clear'].append(False)
                    dict['Predicted SEP All Clear'].append(sep_outcome)
                    dict['Total Forecasts'].append(np.nan)
                    dict['Total Hits'].append(np.nan)
                    dict['Total Misses'].append(np.nan)
                    dict['Total False Alarms'].append(np.nan)
                    dict['Total Correct Negatives'].append(np.nan)
                    dict['First Prediction Window'].append(pd.NaT)
                    dict['Last Prediction Window'].append(pd.NaT)
                else:
                    sub.sort_values(by='Prediction Window Start', inplace=True)
                    pred_win_first = sub['Prediction Window Start'].iloc[0]
                    pred_win_last = sub['Prediction Window Start'].iloc[len(sub['Prediction Window End'])-1]

                    nsepcasts = len(sub) #total forecasts associated with the SEP event
                    print(f"{nsepcasts} {model} forecasts associated with {sep}")
                    #Desire only one forecast per event, but if there are
                    #multiple, then can use Hit = any Hit, Miss = all Miss
                    #"Observed SEP All Clear" will be False (because an observed
                    #SEP event means False all clear)
                    #If "Predicted SEP All Clear" is equal to the "observed SEP All Clear"
                    #field, then the model got it right -> Hit, and vice versa
                    compare = sub['Observed SEP All Clear'] == sub['Predicted SEP All Clear']
                    #Hits, any False, False columns which give a true in compare
                    if compare.any():
                        sep_outcome = 'Hit'
                        sep_outcome_bool = False
                        hit += 1
                        sep_caught_str += str(sep) + ';'
                    else:
                        sep_outcome = 'Miss'
                        sep_outcome_bool = True
                        miss += 1
                        sep_miss_str += str(sep) + ';'

                    nhits = compare.sum() #total number of hit forecasts
                    nmiss = (~compare).sum() #total numer of miss forecasts

                    #Record info for deoverlapping
                    dict['Start Date'].append(sep)
                    dict['End Date'].append(pd.NaT)
                    dict['Observed SEP Threshold Crossing Time'].append(sep)
                    dict['Observed SEP All Clear'].append(False)
                    dict['Predicted SEP All Clear'].append(sep_outcome_bool)
                    dict['Total Forecasts'].append(nsepcasts)
                    dict['Total Hits'].append(nhits)
                    dict['Total Misses'].append(nmiss)
                    dict['Total False Alarms'].append(np.nan)
                    dict['Total Correct Negatives'].append(np.nan)
                    dict['First Prediction Window'].append(pred_win_first)
                    dict['Last Prediction Window'].append(pred_win_last)

            sep_outcomes.append(sep_outcome)

            
            #NON-EVENTS
            #For each non-event, get False Alarm/Correct Negative/No Data
            #Here we need to extract the appropriate columns
            #All forecasts with prediction windows that start at the
            #Non-Event Start dates and all predictions with start timess all the
            #way through to the End date
            nonsep_outcome = None
            nonsep_outcome_bool = None
            if not pd.isnull(non_st):
                sub = associated_forecasts(df_ac, non_st, non_end, split)
                #Remove any forecasts in sub that are associated to a SEP
                #event (Observed SEP All Clear False), because those should
                #not be considered to be part of the non-event time period
                sub = sub[sub['Observed SEP All Clear']==True]
 
                if sub.empty:
                    nonsep_outcome = 'No Data'
                    #Record info for deoverlapping
                    dict['Start Date'].append(non_st)
                    dict['End Date'].append(non_end)
                    dict['Observed SEP Threshold Crossing Time'].append(pd.NaT)
                    dict['Observed SEP All Clear'].append(True)
                    dict['Predicted SEP All Clear'].append(nonsep_outcome)
                    dict['Total Forecasts'].append(np.nan)
                    dict['Total Hits'].append(np.nan)
                    dict['Total Misses'].append(np.nan)
                    dict['Total False Alarms'].append(np.nan)
                    dict['Total Correct Negatives'].append(np.nan)
                    dict['First Prediction Window'].append(pd.NaT)
                    dict['Last Prediction Window'].append(pd.NaT)
                else:
                    sub.sort_values(by='Prediction Window Start', inplace=True)
                    pred_win_first = sub['Prediction Window Start'].iloc[0]
                    pred_win_last = sub['Prediction Window Start'].iloc[len(sub['Prediction Window End'])-1]

                    nnoncasts = len(sub) #total number of forecasts
                    print(f"{nnoncasts} {model} forecasts associated with {non_st} to {non_end}")
                
                    compare = sub['Observed SEP All Clear'] == sub['Predicted SEP All Clear']
                    #Hits, any True, False columns will give a False in compare
                    if compare.all():
                        nonsep_outcome = 'CN'
                        nonsep_outcome_bool = True
                        cn += 1
                    else:
                        nonsep_outcome = 'FA'
                        nonsep_outcome_bool = False
                        fa += 1

                    nfa = (~compare).sum()
                    ncn = compare.sum()

                    #Record info for deoverlapping
                    dict['Start Date'].append(non_st)
                    dict['End Date'].append(non_end)
                    dict['Observed SEP Threshold Crossing Time'].append(pd.NaT)
                    dict['Observed SEP All Clear'].append(True)
                    dict['Predicted SEP All Clear'].append(nonsep_outcome_bool)
                    dict['Total Forecasts'].append(nnoncasts)
                    dict['Total Hits'].append(np.nan)
                    dict['Total Misses'].append(np.nan)
                    dict['Total False Alarms'].append(nfa)
                    dict['Total Correct Negatives'].append(ncn)
                    dict['First Prediction Window'].append(pred_win_first)
                    dict['Last Prediction Window'].append(pred_win_last)

            nonsep_outcomes.append(nonsep_outcome)


        scores = metrics.contingency_scores(hit, miss, fa, cn)
        validation.fill_all_clear_dict(all_clear_dict, model, energy_key, thresh_key,
            energy_key, thresh_key, scores, hit, sep_caught_str,
            miss, sep_miss_str)
        
        sep_results.update({model: sep_outcomes})
        nonsep_results.update({model: nonsep_outcomes})

        #Deoverlapped dataframe
        df_do = pd.DataFrame(dict)
        df_do = df_do.sort_values('Start Date')
        fnameout = fname.replace('.csv','_deoverlap.csv')
        df_do.to_csv(fnameout, index=False)
        print(f"Wrote out {fnameout}.")

    
    df_sep = pd.DataFrame(sep_results)
    df_sep_drop = df_sep.drop(columns=['SEP Events'], axis=1)
    df_sep['Total Hits'] = df_sep_drop.apply(lambda x: x.str.contains('Hit')).sum(axis=1)
    df_sep['Total Misses'] = df_sep_drop.apply(lambda x: x.str.contains('Miss')).sum(axis=1)
    df_sep['Total No Data'] = df_sep_drop.apply(lambda x: x.str.contains('No Data')).sum(axis=1)
    if write_grid:
        gridname = f"all_clear_deoverlap_SEP_{energy_key}_{thresh_key}.csv"
        if len(models) == 1:
            gridname = f"all_clear_deoverlap_SEP_{models[0]}_{energy_key}_{thresh_key}.csv"
        df_sep.to_csv(os.path.join(csv_path,gridname), index=False)

    df_nonsep = pd.DataFrame(nonsep_results)
    df_nonsep_drop = df_nonsep.drop(columns=['Non-Event Start', 'Non-Event End'], axis=1)
    df_nonsep['Total Correct Negatives'] = df_nonsep_drop.apply(lambda x: x.str.contains('CN')).sum(axis=1)
    df_nonsep['Total False Alarms'] = df_nonsep_drop.apply(lambda x: x.str.contains('FA')).sum(axis=1)
    df_nonsep['Total No Data'] = df_nonsep_drop.apply(lambda x: x.str.contains('No Data')).sum(axis=1)
    if write_grid:
        gridname = f"all_clear_deoverlap_NonEvent_{energy_key}_{thresh_key}.csv"
        if len(models) == 1:
            gridname = f"all_clear_deoverlap_NonEvent_{models[0]}_{energy_key}_{thresh_key}.csv"
        df_nonsep.to_csv(os.path.join(csv_path,gridname), index=False)

    df_scores = pd.DataFrame(all_clear_dict)
    gridname = f"all_clear_deoverlap_metrics_{energy_key}_{thresh_key}.csv"
    if len(models) == 1:
        gridname = f"all_clear_deoverlap_metrics_{models[0]}_{energy_key}_{thresh_key}.csv"
    df_scores.to_csv(os.path.join(csv_path,gridname), index=False)





def probability_deoverlap(csv_path, models, energy_min, energy_max, threshold,
    dates_file='',seps=[], nonevent_st=[], nonevent_end=[], split=False,
    write_grid=True):
    """ For a list of models, reads in probability_selections files output
        by sphinx and extracts the maximum probability predicted for a set of dates, 
        stored in df_dates.
        
        User may enter a file containing SEP and non-event dates.
        df_dates is expected to consist of two columns, labeled 
        "SEP Events" and "Non-Events".
        
        Or user may enter a list of SEP dates, a list of non-event start times, 
        and a list of non-event end times.
        
        This subroutine uses the original all_clear_selections files.
        
        INPUT:
        
            :csv_path: (string) path the all_clear_selections csv files.
            :models: (list) list of models to be included in the grid; 
                model names must exactly match the short_name field in 
                the forecast jsons or SPHINX output files.
            :dates_file: (string) csv file containing list of dates that want to use to 
                generate grids with a column titled "SEP Events" with the 
                SEP start times and columns titled "Non-Event Start" 
                and "Non-Event End". 
            :energy_min: (float) low edge of energy channel of interest
            :energy_max: (float) high edge of energy channel of interest (-1)
                for infinity (>10 MeV -> energy_min = 10, energy_max = -1)
            :threshold: (float) threshold applied for SEP event definition
                (e.g. >10 MeV exceeds 10 pfu -> threshold = 10)
            :split: (bool) 
                If True, for forecasts with prediction windows that cross dates, 
                    they will be associated with the date with the most overlap 
                    with their prediction window.
                If False, then forecasts with the prediction_window_start inside of 
                    the non-event time period will be evaluated.
            :write_grid: (bool) if True, will write out visual grid csv file
                
        OUTPUT:
        
            Write out two csv files containing desired grid with 
            dates on one axis, models on other axis, and outcomes
            as the entries. "No Data" entries indicate a forecast
            was not provided for a given date.
            
            Write out recalculated probability metrics and plots.
         
    """
    
    #Find date range that fully spans SEP event and non-event periods for
    #models that need to be deoverlapped.
    print(f"probability_deoverlap: Reading in dates file {dates_file}.")
    
    df_dates = create_date_range_df(dates_file, seps, nonevent_st, nonevent_end)


    #Dictionaries needed to create visual grids
    sep_results_max = {"SEP Events": df_dates["SEP Events"].to_list()}
    nonsep_results_max = {"Non-Event Start": df_dates["Non-Event Start"].to_list(),
        "Non-Event End": df_dates["Non-Event End"].to_list()}

    sep_results_mean = {"SEP Events": df_dates["SEP Events"].to_list()}
    nonsep_results_mean = {"Non-Event Start": df_dates["Non-Event Start"].to_list(),
        "Non-Event End": df_dates["Non-Event End"].to_list()}


    #Probability metrics
    probability_dict = validation.initialize_probability_dict()

    #Find first and last date out of all the dates
    max_date = df_dates.max().max()
    min_date = df_dates.min().min()
    
    print(f"probability_deoverlap: Models will be deoverlapped between {min_date} and {max_date}.")

    energy_key = f"min.{float(energy_min):.1f}.max.{float(energy_max):.1f}.units.MeV"
    print(f"Energy channel: {energy_key}")
    thresh_key = f"threshold_{float(threshold):.1f}"
    print(f"Threshold: {thresh_key}")

    for model in models:
        #Counts of forecasts associated with each date range output to file
        dict = {'Start Date': [], 'End Date': [],
                'Observed SEP Threshold Crossing Time': [],
                'Observed SEP Probability': [],
                'Predicted SEP Probability Max': [],
                'Predicted SEP Probability Mean': [],
                'Total Forecasts': [],
                'First Prediction Window': [],
                'Last Prediction Window': []}


        #Store Max, Mean probability
        sep_max = []
        sep_mean = []
        #Store False Alarm/Correct Negative/No Data
        nonsep_max = []
        nonsep_mean = []
        
        fname = os.path.join(csv_path,
            f"probability_selections_{model}_{energy_key}_{thresh_key}.csv")

        if 'UNSPELL' in model: #should be revised to generally account for mismatch
            if energy_min == 10:
                fname = os.path.join(csv_path,
                "probability_selections_UNSPELL flare_min.10.0.max.-1.0.units.MeV_min.5.0.max.-1.0.units.MeV_threshold_10.0_mm.csv")
        
        if not os.path.isfile(fname):
            print(f"probability_deoverlap: File does not exist. Check model and csv_path. {fname}. Skipping model.")
            continue

        print(f"probability_deoverlap: Reading in file {fname}.")
        
        df = pd.read_csv(fname, parse_dates=['Observed SEP Threshold Crossing Time','Prediction Window Start','Prediction Window End'])

        #Date columns indicating SEP range
        key_st = 'Prediction Window Start'
        key_end = 'Prediction Window End'

        #Go through SEP Events and Non-Events and record results
        for sep, non_st, non_end in df_dates.itertuples(index=False):

            #SEP EVENTS
            #For each SEP event, get Hit/Miss/No Data
            sep_outcome_max = np.nan
            sep_outcome_mean = np.nan
            nsepcasts = np.nan
            maxprob = np.nan
            nnoncasts = np.nan
            if not pd.isnull(sep):
                #Get results for a single SEP event
                sub = df[df['Observed SEP Threshold Crossing Time'] == sep]
                
                if sub.empty:
                    #Record info for deoverlapping
                    dict['Start Date'].append(sep)
                    dict['End Date'].append(pd.NaT)
                    dict['Observed SEP Threshold Crossing Time'].append(sep)
                    dict['Observed SEP Probability'].append(1.0)
                    dict['Predicted SEP Probability Max'].append(np.nan)
                    dict['Predicted SEP Probability Mean'].append(np.nan)
                    dict['Total Forecasts'].append(np.nan)
                    dict['First Prediction Window'].append(pd.NaT)
                    dict['Last Prediction Window'].append(pd.NaT)
                else:
                    sub.sort_values(by='Prediction Window Start', inplace=True)
                    pred_win_first = sub['Prediction Window Start'].iloc[0]
                    pred_win_last = sub['Prediction Window Start'].iloc[len(sub['Prediction Window End'])-1]

                    nsepcasts = len(sub) #total forecasts associated with the SEP event
                    print(f"{nsepcasts} {model} forecasts associated with {sep}")
                    #Desire max probability forecast for the SEP event
                    sep_outcome_max = sub['Predicted SEP Probability'].max()
                    sep_outcome_mean = sub['Predicted SEP Probability'].mean()

                    #Record info for deoverlapping
                    dict['Start Date'].append(sep)
                    dict['End Date'].append(pd.NaT)
                    dict['Observed SEP Threshold Crossing Time'].append(sep)
                    dict['Observed SEP Probability'].append(1.0)
                    dict['Predicted SEP Probability Max'].append(sep_outcome_max)
                    dict['Predicted SEP Probability Mean'].append(sep_outcome_mean)
                    dict['Total Forecasts'].append(nsepcasts)
                    dict['First Prediction Window'].append(pred_win_first)
                    dict['Last Prediction Window'].append(pred_win_last)

            sep_max.append(sep_outcome_max)
            sep_mean.append(sep_outcome_mean)
            
            #NON-EVENTS
            #For each non-event, get False Alarm/Correct Negative/No Data
            #Here we need to extract the appropriate columns
            #All forecasts with prediction windows that start at the
            #Non-Event Start dates and all predictions with start timess all the
            #way through to the End date
            nonsep_outcome_max = np.nan
            nonsep_outcome_mean = np.nan
            if not pd.isnull(non_st):
                sub = associated_forecasts(df, non_st, non_end, split)
                #Remove any forecasts in sub that are associated to a SEP
                #event (Observed SEP All Clear False), because those should
                #not be considered to be part of the non-event time period
                sub = sub[sub['Observed SEP Probability']==0.0]
 
                if sub.empty:
                    #Record info for deoverlapping
                    dict['Start Date'].append(non_st)
                    dict['End Date'].append(non_end)
                    dict['Observed SEP Threshold Crossing Time'].append(pd.NaT)
                    dict['Observed SEP Probability'].append(0.0)
                    dict['Predicted SEP Probability Max'].append(np.nan)
                    dict['Predicted SEP Probability Mean'].append(np.nan)
                    dict['Total Forecasts'].append(np.nan)
                    dict['First Prediction Window'].append(pd.NaT)
                    dict['Last Prediction Window'].append(pd.NaT)
                else:
                    sub.sort_values(by='Prediction Window Start', inplace=True)
                    pred_win_first = sub['Prediction Window Start'].iloc[0]
                    pred_win_last = sub['Prediction Window Start'].iloc[len(sub['Prediction Window End'])-1]

                    nnoncasts = len(sub) #total number of forecasts
                    print(f"{nnoncasts} {model} forecasts associated with {non_st} to {non_end}")
                
                    nonsep_outcome_max = sub['Predicted SEP Probability'].max()
                    nonsep_outcome_mean = sub['Predicted SEP Probability'].mean()
                    
                    #Record info for deoverlapping
                    dict['Start Date'].append(non_st)
                    dict['End Date'].append(non_end)
                    dict['Observed SEP Threshold Crossing Time'].append(pd.NaT)
                    dict['Observed SEP Probability'].append(0.0)
                    dict['Predicted SEP Probability Max'].append(nonsep_outcome_max)
                    dict['Predicted SEP Probability Mean'].append(nonsep_outcome_mean)
                    dict['Total Forecasts'].append(nnoncasts)
                    dict['First Prediction Window'].append(pred_win_first)
                    dict['Last Prediction Window'].append(pred_win_last)

            nonsep_max.append(nonsep_outcome_max)
            nonsep_mean.append(nonsep_outcome_mean)

        #Save aggregated max, mean probabilities per model
        sep_results_max.update({model: sep_max})
        nonsep_results_max.update({model: nonsep_max})

        sep_results_mean.update({model: sep_mean})
        nonsep_results_mean.update({model: nonsep_mean})

        #Write out deoverlapped dataframe per model
        df_do_all = pd.DataFrame(dict)
        df_do_all = df_do_all.sort_values('Start Date')
        fnameout = fname.replace('.csv','_deoverlap.csv')
        df_do_all.to_csv(fnameout, index=False)
        print(f"Wrote out {fnameout}.")


        for type in ['Max','Mean']:
            key = 'Predicted SEP Probability ' + type

            #Recalculate probability metrics and plots using the deoverlapped max probability
            #Drop time periods with No Data
            df_do = df_do_all.dropna(subset=['Observed SEP Probability', key])
            obs = df_do['Observed SEP Probability'].to_list()
            pred = df_do[key].to_list()

            #Calculate metrics
            brier_score = metrics.calc_brier(obs, pred)
            brier_skill = metrics.calc_brier_skill(obs, pred)
            rank_corr_coeff = metrics.calc_spearman(obs, pred)

            roc_auc, roc_curve_plt = metrics.receiver_operator_characteristic(obs, pred, model)
            
            roc_curve_plt.plot()
            skill_line = np.linspace(0.0, 1.0, num=10) # Constructing a diagonal line that represents no skill/random guess
            plt.plot(skill_line, skill_line, '--', label = 'Random Guess')
            figname = csv_path + '/../output/plots/ROC_curve_' \
                    + model + "_" + energy_key.strip() + "_" + thresh_key

            figname += "_deoverlap_" + type + ".pdf"
            plt.legend(loc="lower right")
            roc_curve_plt.figure_.savefig(figname, dpi=300, bbox_inches='tight')
            plt.close(roc_curve_plt.figure_)
            
            #Save to dict (ultimately dataframe)
            probability_dict['Model'].append(model + ' ' + type)
            probability_dict['Energy Channel'].append(energy_key)
            probability_dict['Threshold'].append(thresh_key)
            probability_dict['Prediction Energy Channel'].append(energy_key)
            probability_dict['Prediction Threshold'].append(thresh_key)
            probability_dict['ROC Curve Plot'].append(figname)
            probability_dict['Brier Score'].append(brier_score)
            probability_dict['Brier Skill Score'].append(brier_skill)
            probability_dict['Spearman Correlation Coefficient'].append(rank_corr_coeff)
            probability_dict['Area Under ROC Curve'].append(roc_auc)

    
    df_sep_max = pd.DataFrame(sep_results_max)
    df_sep_mean = pd.DataFrame(sep_results_mean)
    if write_grid:
        #MAX results
        gridname = f"probability_grid_SEP_{energy_key}_{thresh_key}_Max.csv"
        if len(models) == 1:
            gridname = f"probability_grid_SEP_{models[0]}_{energy_key}_{thresh_key}_Max.csv"
        df_sep_max.to_csv(os.path.join(csv_path,gridname), index=False)
        
        #MEAN results
        gridname = f"probability_grid_SEP_{energy_key}_{thresh_key}_Mean.csv"
        if len(models) == 1:
            gridname = f"probability_grid_SEP_{models[0]}_{energy_key}_{thresh_key}_Mean.csv"
        df_sep_mean.to_csv(os.path.join(csv_path,gridname), index=False)


    df_nonsep_max = pd.DataFrame(nonsep_max)
    df_nonsep_mean = pd.DataFrame(nonsep_mean)
    if write_grid:
        #MAX results
        gridname = f"probability_grid_NonEvent_{energy_key}_{thresh_key}_Max.csv"
        if len(models) == 1:
            gridname = f"probability_grid_NonEvent_{models[0]}_{energy_key}_{thresh_key}_Max.csv"
        df_nonsep_max.to_csv(os.path.join(csv_path,gridname), index=False)

        #MEAN results
        gridname = f"probability_grid_NonEvent_{energy_key}_{thresh_key}_Mean.csv"
        if len(models) == 1:
            gridname = f"probability_grid_NonEvent_{models[0]}_{energy_key}_{thresh_key}_Mean.csv"
        df_nonsep_mean.to_csv(os.path.join(csv_path,gridname), index=False)



    df_scores = pd.DataFrame(probability_dict)
    gridname = f"probability_metrics_deoverlap_{energy_key}_{thresh_key}.csv"
    if len(models) == 1:
        gridname = f"probability_metrics_deoverlap_{models[0]}_{energy_key}_{thresh_key}.csv"
    df_scores.to_csv(os.path.join(csv_path,gridname), index=False)




def deoverlap_forecasts(quantity, csv_path, model, energy_min, energy_max, threshold,
    date_range_st = None, date_range_end=None, td=pd.NaT, split=True):
    """ For models that produce continuous forecasts on a set cadence 
        with overlapping prediction windows, deoverlap by getting a 
        single answer for a given period of time.
        
        For example, for each 24 hour period, assign a single hit, 
        miss, correct negative, or false alarm according to the 
        forecasts within that 24 hour period.
        
        All forecasts associated with SEP events will be associated to those
        events. The remaining forecasts will be used to get deoverlapped 
        answers for each date in the date range.
        
        INPUT:
        
            :quantity: (string) "All Clear", "all clear", "probability, "Probability"
            :csv_path: (string) path to the csv directory containing all_clear_selections_*.csv
                or probability_selections_*.csv files output by SPHINX.
            :model: (string) one model name to deoverlap; must exactly match model
                short_name used in the all_clear_selections files
            :date_range_st: (pd date_range series) start of time periods of interest. May include a continuous time period or a set of specific time
                periods, e.g. a challenge set
                If not specified, will default to continuous time periods between
                start and end time of forecasts in input file.
            :date_range_end: (pd date_range series) end of time periods of interest. 
            :td: (datetime timedelta) if date_range_st not specified, will create
                continuous time periods with td. If date_range_st specified, but 
                date_range_end not specified, will create time periods of
                date_range_st + td. Default td = 24 hours.
            :split: (bool) 
                If True, for forecasts with prediction windows that cross dates, 
                    they will be associated with the date with the most overlap 
                    with their prediction window.
                If False, then forecasts with the prediction_window_start inside of 
                    the non-event time period will be evaluated.

                
        OUTPUT:
        
            For All Clear: Rederived contingency table and metrics written to file
            For probability: Rederived probability metrics and ROC curve written to file
        
    """
    energy_key = f"min.{float(energy_min):.1f}.max.{float(energy_max):.1f}.units.MeV"
    thresh_key = f"threshold_{float(threshold):.1f}"

    #CREATE DATE RANGE IF NOT INPUT
    #If date cadence or window isn't provided, set to 24 hours
    if pd.isnull(td):
        td = datetime.timedelta(hours=24)

    filename = os.path.join(csv_path,
        f"all_clear_selections_{model}_{energy_key}_{thresh_key}.csv")
    if not os.path.isfile(filename):
        print(f"deoverlap_all_clear: Cannot read file {filename}. Returning.")
        return

    df = pd.read_csv(filename, parse_dates=['Prediction Window Start', 'Prediction Window End'])
    if df.empty:
        print(f"deoverlap_all_clear: Empty file {filename}. Returning.")
        return

    #Identify all the SEP events
    sep_events = resume.identify_unique(df,'Observed SEP Threshold Crossing Time')

    #Create time range for all other time periods
    df.sort_values(by='Prediction Window Start', inplace=True)
    first_date = df['Prediction Window Start'].iloc[0]
    last_date = df['Prediction Window End'].iloc[len(df['Prediction Window End'])-1]

    #If start times provided, but not end times, then apply td
    if not pd.isnull(date_range_st) and pd.isnull(date_range_end):
        date_range_end = date_range_st + td
        date_range_st = date_range_st.to_list()
        date_range_end = date_range_end.to_list()

    #If no date range provided, then create a continuous date range
    #between the first and last date in the provided forecasts
    if pd.isnull(date_range_st):
        date_range_st, date_range_end = create_date_range(first_date, last_date, td)

    if quantity == "All Clear" or quantity == "all clear":
        all_clear_deoverlap(csv_path, [model], energy_min, energy_max, threshold, seps=sep_events,
            nonevent_st=date_range_st, nonevent_end=date_range_end, split=split,
            write_grid=False)

    if quantity == "Probability" or quantity == "probability":
        probability_deoverlap(csv_path, [model], energy_min, energy_max, threshold, seps=sep_events,
            nonevent_st=date_range_st, nonevent_end=date_range_end, split=split,
            write_grid=False)
        

def make_histograms():
    from matplotlib.ticker import MultipleLocator
    import matplotlib
    
    
    plt.rcParams['font.size'] = 16
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Arial'
   
    model_names = ['SAWS-ASPECS 0-6 hrs 50%', 'SAWS-ASPECS 0-6 hrs 90%', 'SAWS-ASPECS 50%', 'SAWS-ASPECS 90%', 'SAWS-ASPECS flare 50%', 'SAWS-ASPECS flare 90%', \
    'SEPMOD', 'SEPSTER (Parker Spiral)', 'SEPSTER (WSA-ENLIL)', 'SEPSTER2D', 'UMASEP-100', 'ZEUS+iPATH_CME']
    
    # Choosing metrics here for time metrics these will be interpreted as not the log values 
    metrics_list = ['LE'] #Also sometimes ALE
    # list this out as it appears in the _selections files
    forecast_quantity = ['peak_intensity_max', 'start_time', 'peak_intensity_max_time', 'peak_intensity', 'duration']
    name_dictionary = {
        'SEPMOD': 'SEPMOD',
        'ZEUS+iPATH_CME': 'iPATH',
        'SEPSTER2D': 'SEPSTER2D',
        'SEPSTER (Parker Spiral)': 'SEPSTER (PS)',
        'SEPSTER (WSA-ENLIL)': 'SEPSTER (WE)',
        'UMASEP-100': 'UMASEP-100',
        'COMESEP flare only': 'COMESEP flare only',
        'COMESEP flare+CME ': 'COMESEP flare+CME',
        'MFLAMPA': 'MFLAMPA',
        'STAT': 'STAT',
        'SFS-Update': 'SFS Update',
        'ADEPT-AFRL 1hr' :'ADEPT 1hr',
        'ADEPT-AFRL 6hr': 'ADEPT 6hr',
        'SPREAdFAST': 'SPREAdFAST',
        'SEPSAT': 'SEPSAT',
        'SAWS-ASPECS 0-6 hrs 50%': 'SAWS-ASPECS 0-6 hrs 50%',
        'SAWS-ASPECS 0-6 hrs 90%': 'SAWS-ASPECS 0-6 hrs 90%',
        'SAWS-ASPECS 50%': 'SAWS-ASPECS 50%',
        'SAWS-ASPECS 90%': 'SAWS-ASPECS 90%',
        'SAWS-ASPECS CME (SOHO) 50%': 'ASPECS CME 50%',
        'SAWS-ASPECS CME (SOHO) electrons 50%': 'ASPECS CME electrons 50%' ,
        'SAWS-ASPECS flare + CME (SOHO) 50%': 'ASPECS CME + flare 50%',
        'SAWS-ASPECS flare 50%': 'ASPECS flare 50%', 
        'SAWS-ASPECS flare electrons 50%': 'ASPECS flare electrons 50%',
        'SAWS-ASPECS CME (SOHO) 90%': 'ASPECS CME 90%',
        'SAWS-ASPECS CME (SOHO) electrons 90%': 'ASPECS CME electrons 90%' ,
        'SAWS-ASPECS flare + CME (SOHO) 90%': 'ASPECS CME + flare 90%',
        'SAWS-ASPECS flare 90%': 'ASPECS flare 90%', 
        'SAWS-ASPECS flare electrons 90%': 'ASPECS flare electrons 90%',
        'UMASEP-10': 'UMASEP-10'
    }
    forecast_label = {
        'peak_intensity_max': 'SEP Max Peak Flux',
        'start_time': 'SEP Start Time',
        'peak_intensity_max_time': 'SEP Max Peak Flux Time',
        'peak_intensity': 'SEP Onset Peak Flux',
        'duration': 'SEP Event Duration'
    }
    observed_dictionary = {
        'peak_intensity_max': 'Observed SEP Peak Intensity Max (Max Flux)',
        'start_time': 'Observed SEP Start Time',
        'peak_intensity_max_time': 'Observed SEP Peak Intensity Max (Max Flux) Time',
        'peak_intensity': 'Observed SEP Peak Intensity (Onset Peak)',
        'duration': 'Observed SEP Duration'
    }
    forecast_dictionary = {
        'peak_intensity_max': 'Predicted SEP Peak Intensity Max (Max Flux)',
        'start_time': 'Predicted SEP Start Time',
        'peak_intensity_max_time': 'Predicted SEP Peak Intensity Max (Max Flux) Time',
        'peak_intensity': 'Predicted SEP Peak Intensity (Onset Peak)',
        'duration': 'Predicted SEP Duration'
    }
    # These 'outliers' are subject to change based on want we decide is best
    outliers_dictionary = {
        'peak_intensity_max': 2.0,
        'start_time': 10.0,
        'peak_intensity_max_time': 24.0,
        'peak_intensity': 2.0,
        'duration': 24.0
    }
    metric_units = {
        'peak_intensity': ' (pfu)',
        'peak_intensity_max': ' (pfu)',
        'start_time': ' (hours)',
        'peak_intensity_max_time': ' (hours)',
        'peak_intensity_time': ' (hours)',
        'duration': ' (hours)'
    }
    # setting up a list to be used in the outliers output file as the column names for the dataframe
    fields_outlier = ['Model', 'Dataset', 'Energy Channel Key', 'Observed SEP Threshold Crossing Time', 'Observed SEP Peak Intensity Max (Max Flux)', \
        'Observed SEP Peak Intensity Max (Max Flux) Time', 'Predicted SEP Start Time', 'Predicted SEP Peak Intensity Max (Max Flux)', \
            'Predicted SEP Peak Intensity Max (Max Flux) Time', 'Observed SEP Onset Peak Flux', 'Predicted SEP Onset Peak Flux', 'Reason for Outlier', 'Metric Name', 'Metric Calculation', 'Forecast Source']
    outliers = []
    event_list_sepval = []
    event_fields = ['Model', 'Dataset', 'Energy Channel Key', 'Observed SEP Threshold Crossing Time', 'Metric Calculation']
    energy_list = ['10', '100']
    
    # 
    # jet_cmap = plt.cm.get_cmap('jet')
    # mapping = []
    # for i in range(len(model_names)):
    #     mapping.append(jet_cmap(i*12))
    
    
    # There's probably a better way to do this but I was short on time to prepare this analysis. Histograms are made looping
    # over energy, forecast quantity and then lastly by model, which ia all based on the lists and dictionaries above

    # To do this for multiple datasets (i.e. SEPVAL and Scoreboard) on the same plot, repeat the reading in step
    # for the other dataset and recalculate the metrics for the second dataset
    for energy in energy_list:
        print('Energy Channel', energy)
        if energy == '10':
            energy_thresh = 'min.10.0.max.-1.0.units.MeV_threshold_10.0'
        else:
            energy_thresh = 'min.100.0.max.-1.0.units.MeV_threshold_1.0'
        for forecasts in forecast_quantity:
            print(forecasts)
            print(len(model_names))
            labels = []
            x_locations = []
            x_loc = 0
            big_fig, big_ax = plt.subplots(figsize=(14, 12))
            plot_iter = 0
            big_ax.set_prop_cycle(color=plt.cm.tab20.colors)
            for names in model_names:
                if 'UMASEP' in names:
                    if energy == '10':
                        names = 'UMASEP-10'
                    else:
                        names = 'UMASEP-100'
                print(names)
                if 'SEPSTER' in names or 'UMASEP' in names or 'COMESEP' in names or 'SFS' in names or 'ADEPT' in names:
                    if '2D' in names:
                        observed_label = observed_dictionary[forecasts]
                        predicted_label = forecast_dictionary[forecasts]
                        file_to_read_in = './output/csv/' + forecasts + '_selections_' + names +' CME_' + energy_thresh + '.csv'
                        if os.path.isfile(file_to_read_in):
                            if forecasts == 'peak_intensity_max':
                                predicted_label = 'Predicted SEP Peak Intensity (Onset Peak)'
                            elif forecasts == 'peak_intensity_max_time':
                                predicted_label = 'Predicted SEP Peak Intensity (Onset Peak) Time'
                            dataframe_sepval = pd.read_csv(file_to_read_in)
                            obs_sepval = dataframe_sepval[observed_label]
                            pred_sepval = dataframe_sepval[predicted_label]
   
                    elif 'UMASEP' in names:
                        observed_label = observed_dictionary[forecasts]
                        predicted_label = forecast_dictionary[forecasts]
                        file_to_read_in = './output/csv/' + forecasts + '_selections_' + names +'_' + energy_thresh + '_First.csv'
                        if os.path.isfile(file_to_read_in):
                            if forecasts == 'peak_intensity_max':
                                predicted_label = 'Predicted SEP Peak Intensity (Onset Peak)'
                            elif forecasts == 'peak_intensity_max_time':
                                predicted_label = 'Predicted SEP Peak Intensity (Onset Peak) Time'
                            dataframe_sepval = pd.read_csv(file_to_read_in)
                            obs_sepval = dataframe_sepval[observed_label]
                            pred_sepval = dataframe_sepval[predicted_label]
                
                    else:
                        observed_label = observed_dictionary[forecasts]
                        predicted_label = forecast_dictionary[forecasts]
                        file_to_read_in = './output/csv/' + forecasts + '_selections_' + names +'_' + energy_thresh + '.csv'
                        if os.path.isfile(file_to_read_in):
                            if forecasts == 'peak_intensity_max':
                                predicted_label = 'Predicted SEP Peak Intensity (Onset Peak)'
                            elif forecasts == 'peak_intensity_max_time':
                                predicted_label = 'Predicted SEP Peak Intensity (Onset Peak) Time'
                            dataframe_sepval = pd.read_csv(file_to_read_in)
                            obs_sepval = dataframe_sepval[observed_label]
                            pred_sepval = dataframe_sepval[predicted_label]
                        else:
                            pred_sepval = []
                            obs_sepval = []
                            obs_sb = []
                            pred_sb = []
                            
                    
                else:
                    observed_label = observed_dictionary[forecasts]
                    predicted_label = forecast_dictionary[forecasts]
                    file_to_read_in = './output/csv/' + forecasts + '_selections_' + names +'_' + energy_thresh + '.csv'
                    if os.path.isfile(file_to_read_in):
                        dataframe_sepval = pd.read_csv(file_to_read_in)
                        obs_sepval = dataframe_sepval[observed_label]
                        pred_sepval = dataframe_sepval[predicted_label]
                    else:
                        pred_sepval = []
                        obs_sepval = []
            
                  
                if len(pred_sepval) == 0: 
                    pass
                else:
                    
                    for scores in metrics_list:
                        
                        if 'time' in forecasts and scores == 'ALE': 
                            i = 0
                            j = 0
                            metric_label = 'Absolute Error'
                            metric_sepval = []
                            metric_sb = []
                            metric_sepval_clean = []
                            metric_sb_clean = []
                            for i in range(len(pred_sepval)):
                                foo = np.abs(datetime.fromisoformat(pred_sepval[i]) - datetime.fromisoformat(obs_sepval[i]))
                                metric_sepval.append(foo.total_seconds()/(60*60)) #convert to hours
                                metric_sepval_clean.append(foo.total_seconds()/(60*60)) #convert to hours
                           
                            n_sepval = len(metric_sepval_clean)
                            
                        elif 'time' in forecasts and scores == 'LE':
                            i = 0
                            j = 0
                            metric_label = 'Error'
                            metric_sepval = []
                            metric_sb = []
                            metric_sepval_clean = []
                            metric_sb_clean = []

                            for i in range(len(pred_sepval)):
                                foo = (datetime.datetime.fromisoformat(pred_sepval[i]) - datetime.datetime.fromisoformat(obs_sepval[i]))
                                metric_sepval.append(foo.total_seconds()/(60*60)) #convert to hours
                                metric_sepval_clean.append(foo.total_seconds()/(60*60)) #convert to hours
                            
                            n_sepval = len(metric_sepval_clean)
                            
                        elif 'duration' in forecasts:
                            obs_sepval_clean, pred_sepval_clean = metrics.remove_zero(obs_sepval, pred_sepval)
                            if scores == 'LE':
                                metric_label = "Error"
                                metric_sepval_clean = metrics.switch_error_func('E', obs_sepval_clean, pred_sepval_clean)
                                metric_sepval = metrics.switch_error_func('E', obs_sepval, pred_sepval)
                              
                            elif scores == 'ALE':
                                metric_label = 'Absolute Error'
                                metric_sepval_clean = metrics.switch_error_func('AE', obs_sepval_clean, pred_sepval_clean)
                                metric_sepval = metrics.switch_error_func('AE', obs_sepval, pred_sepval)
                            
                            if all(metric_sepval_clean) == None:
                                metric_sepval_clean = 0
                            
                            try:
                                n_sepval = len(metric_sepval_clean)
                            except:
                                n_sepval = 0
                           
                        else:
                            metric_label = scores
                            obs_sepval_clean, pred_sepval_clean = metrics.remove_zero(obs_sepval, pred_sepval)
                            metric_sepval_clean = metrics.switch_error_func(scores, obs_sepval_clean, pred_sepval_clean)
                            metric_sepval = metrics.switch_error_func(scores, obs_sepval, pred_sepval)
                            try:
                                if metric_sepval_clean == None:
                                    metric_sepval_clean = 0
                            except:
                                pass
                            try:
                                n_sepval = len(metric_sepval_clean)
                            except:
                                n_sepval = 0
                            
                        # calculating what's within an order of magnitude
                        if 'A' in scores:
                            count = 0
                            print('duration' in forecasts, 'time' not in forecasts)
                            if 'time' not in forecasts and 'duration' not in forecasts:
                                bins_hist = [0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]
                                bins_cdf = [0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]
                            else:
                                try:

                                    if len(metric_sepval_clean) != 0:
                                        
                                        bin_max = np.round(np.max(metric_sepval_clean))
                                        bin_min = np.round(np.min(metric_sepval_clean))
                                        four_hour_bins = np.arange(-200, 200, 4)
                                        bins_hist = []
                                        for x in range(len(four_hour_bins)):
                                            if four_hour_bins[x] >= 0 and four_hour_bins[x] <= bin_max+4:
                                                bins_hist.append(four_hour_bins[x])
                                            else:
                                                pass
                                        # bins_hist = np.arange(bin_min, bin_max, 4)
                                        # print(bins_hist)
                                        bins_cdf = np.arange(bin_min, bin_max, 1)
                                    else:
                                        pass
                                except:
                                    pass
                            i = 0
                            j = 0
                            # Keeping this here, it is the loops for making an outlier file which may be helpful 
                            # ['Model', 'Dataset', 'Energy Channel Key', 'Observed SEP Threshold Crossing Time', 'Observed SEP Peak Intensity Max (Max Flux)', \
                            # 'Observed SEP Peak Intensity Max (Max Flux) Time', 'Predicted SEP Start Time', 'Predicted SEP Peak Intensity Max (Max Flux)', \
                            # 'Predicted SEP Peak Intensity Max (Max Flux) Time', 'Reason for Outlier', 'Metric Name', 'Result', 'Forecast Source']
                            # for i in range(len(metric_sepval)):
                            #     if metric_sepval[i] >= outliers_dictionary[forecasts] or metric_sepval[i] <= -outliers_dictionary[forecasts]:
                            #         if 'start' in forecasts:
                            #             outliers.append([dataframe_sepval['Model'][i], 'SEPVAL', energy_thresh, dataframe_sepval['Observed SEP Start Time'][i], None, \
                            #                 None, dataframe_sepval['Predicted SEP Start Time'][i], None, \
                            #                 None, 'Start Time', metric_label, metric_sepval[i], dataframe_sepval['Forecast Source'][i]])
                            #         elif 'max_time' in forecasts:
                            #             outliers.append([dataframe_sepval['Model'][i], 'SEPVAL', energy_thresh, dataframe_sepval['Observed SEP Threshold Crossing Time'][i], None, \
                            #                 dataframe_sepval['Observed SEP Peak Intensity Max (Max Flux) Time'][i], None, None, \
                            #                 pred_sepval[i], 'Max Peak Time', metric_label, metric_sepval[i], dataframe_sepval['Forecast Source'][i]])
                            #         else:
                            #             outliers.append([dataframe_sepval['Model'][i], 'SEPVAL', energy_thresh, dataframe_sepval['Observed SEP Threshold Crossing Time'][i], dataframe_sepval['Observed SEP Peak Intensity Max (Max Flux)'][i], \
                            #                 None, None, pred_sepval[i],\
                            #                 None, 'Max Peak Flux', metric_label, metric_sepval[i], dataframe_sepval['Forecast Source'][i]])
                            
                            
                        else:
                        
                            if 'time' not in forecasts and 'duration' not in forecasts:
                                bins_hist = [-4,-3.5,-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3,3.5,4]
                                bins_cdf = [-4,-3.5,-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3,3.5,4]
                            else:            
                                if all(metric_sepval_clean) == None or metric_sepval_clean == []:
                                    metric_sepval_clean = 0
                                
                                
                                if type(metric_sepval_clean) != int:
                                    
                                    bin_max = np.round(np.max(metric_sepval_clean))
                                    bin_min = np.round(np.min(metric_sepval_clean))
                                    four_hour_bins = np.arange(-200, 200, 4)
                                    bins_hist = []
                                    for x in range(len(four_hour_bins)):
                                        if four_hour_bins[x] >= bin_min and four_hour_bins[x] <= bin_max+4:
                                            bins_hist.append(four_hour_bins[x])
                                        else:
                                            pass
                                    
                                    bins_cdf = np.arange(bin_min, bin_max, 1)
                
                                    
                            if 'peak_intensity' in forecasts and 'time' not in forecasts:
                                count = 0
                                count_over = 0
                                count_under = 0
                                count_fact_2 = 0
                                i = 0
                                for i in range(n_sepval):
                                    
                                    if metric_sepval_clean[i] >= -1 and metric_sepval_clean[i] <= 1:
                                        count += 1
                                    elif metric_sepval_clean[i] < -1:
                                        count_under +=1
                                    elif metric_sepval_clean[i] > 1:
                                        count_over += 1  
                                    if metric_sepval_clean[i] >= -np.log10(2) and metric_sepval_clean[i] <= np.log10(2):
                                        count_fact_2 += 1
                                    
                                if n_sepval == 0:
                                    m_sepval = 0
                                else:
                                    m_sepval = str(count/n_sepval)

                                    print('Within OOM SEPVAL = ', count, count/n_sepval)
                                 
                                    print('within a factor of 2', count_fact_2, count_fact_2/n_sepval)
                              
                        if 'peak_intensity' in forecasts and 'time' not in forecasts and scores == 'LE' and '100' not in energy:
                            i = 0
                            j = 0
                            
                        sepval_hist, _ = np.histogram(metric_sepval_clean, bins = 100) # Easy way to give counts in each bin
                        
                        
                        
                        
                        
                        # Histogram Plots *****************************************************************************************************************
                        fig1, ax = plt.subplots(figsize=(14, 12))
                        
                        plt.hist(metric_sepval_clean, bins = bins_hist, alpha=0.5, label = 'SEPVAL N= ' + str(n_sepval))
                        
                    
                        plt.xlabel(forecast_label[forecasts] + ' ' + metric_label + metric_units[forecasts])
                        plt.title(name_dictionary[names] + ' ' + forecast_label[forecasts] + ' ' + metric_label + ' \nDistribution for SEPVAL')
                        if 'time' in forecasts:
                            ax.xaxis.set_major_locator(MultipleLocator(8))
                            ax.xaxis.set_minor_locator(MultipleLocator(4))
                        plt.ylabel('Counts')
                        plt.legend()
                        figname = './output/plots/' + forecast_label[forecasts] + '_' +names + '_' + metric_label + '_' + energy + '_SEPVAL.png'
                        plt.savefig(figname)
                        plt.close()
                        
                       

                       
                            
                            
                        hist_range = (np.min(bins_hist), np.max(bins_hist))
                        
                        bin_edges = [-4, -3.5, -3.0, -2.5, -2.0, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
                        
                        # bin_edges = np.linspace(hist_range[0], hist_range[1], 18)
                        vert_hist = np.histogram(metric_sepval_clean, range = hist_range, bins=17)[0]/n_sepval
                        
                        binned_maximums = np.max(vert_hist)
                        heights = np.diff(bin_edges)
                        centers = bin_edges[:-1]  + heights / 2
                        # big_ax.barh(metric_sepval_clean, bins = bins_hist, alpha=0.5, label = 'SEPVAL N= ' + str(n_sepval))
                        
                        lefts = x_loc

                        big_ax.barh(centers, vert_hist, height=heights, left = lefts, label = name_dictionary[names])
                        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.tab20.colors)
                        
                        x_loc = x_loc + binned_maximums + 0.25
                            
                        
                        
                        
                        # CDF Plots *****************************************************************************************************************************************
                        # if scores == 'LE' and 'time' in forecasts:
                        #     fig3 = plt.figure()
                        
                        #     try:
                        #         plt.hist(metric_sepval_clean, bins = bins_cdf, alpha=0.5, density=True, cumulative=True, histtype="step", label = 'SEPVAL N= ' + str(n_sepval))
                        #     except:
                        #         plt.hist(metric_sepval_clean, label = 'SEPVAL N= ' + str(n_sepval))
                        #     # print(metric_sb, type(metric_sb))
                            
                        #     plt.title(name_dictionary[names] + ' ' + forecast_label[forecasts] + ' ' + metric_label + ' Cumulative \nDistribution for SEPVAL')
                        #     figname = './output/plots/' + 'ALE_CDF_' + names + '_' + forecast_label[forecasts] + '_' + energy + '.png'
                        #     plt.ylabel('Frequency')
                        #     plt.xlabel(forecast_label[forecasts] + ' ' + metric_label)
                        #     plt.legend(loc = 'lower right')
                        #     plt.savefig(figname)
                        #     plt.close()

                        
           
            big_ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
            plt.grid(visible=True, which='major', axis='y')
            plt.xticks([])
            big_ax.set_ylabel("Log Error")
            big_ax.set_xlabel("Models")
            big_ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.title('Scoreboard ' + forecast_label[forecasts] + ' ' + metric_label + ' Distribution Histograms for >' + energy + ' MeV')
            figname = './output/plots/all_scoreboard' + forecast_label[forecasts] + '_' + metric_label + '_' + energy + '.png'
            plt.savefig(figname, dpi=600, bbox_inches='tight')
            plt.close()


    # outliers_file = 'outliers_file.csv'
    # output_dataframe = pd.DataFrame(outliers, columns=fields_outlier)
    # output_dataframe.to_csv(outliers_file, sep=',', header = True)


   
    

    
   
    ##### Reliability Plot section *************************************************************************************************
    prob_models = ['MAG4_LOS_FEr', 'MAG4_LOS_r', 'MAG4_SHARP_HMI', 'MAG4_SHARP_FE', 'MAG4_SHARP', 'SWPC Day 1', 'GSU All clear', 'SAWS-ASPECS flare']
    plt.rcParams['font.size'] = 18
    for model_names in prob_models:
        fig, ax1 = plt.subplots(figsize=(14, 12))
        file_to_read_in = './output/csv/probability_selections_' + model_names + '_min.10.0.max.-1.0.units.MeV_threshold_10.0.csv'
        dataframe_sepval = pd.read_csv(file_to_read_in)
        obs_sepval = dataframe_sepval['Observed SEP Probability']
        pred_sepval = dataframe_sepval['Predicted SEP Probability']

        # from sklearn.datasets import make_classification
        # from sklearn.model_selection import train_test_split
        # from sklearn.linear_model import LogisticRegression
        bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        from sklearn.calibration import calibration_curve, CalibrationDisplay

        
        prob_true, prob_pred = calibration_curve(obs_sepval, pred_sepval, n_bins = 10)
        print(prob_true)
        print(prob_pred)
        disp = CalibrationDisplay(prob_true, prob_pred, pred_sepval)
        ax1.plot(prob_pred, prob_true, linestyle = '-', marker = 'o')
        plt.title(model_names + ' Reliability Diagram')
        plt.ylim(0, 1)
        ax1.plot([0, 1], [0, 1], label = 'Perfectly Calibrated', color = 'black', linestyle = 'dashed')
        plt.legend()
        ax1.set_ylabel('Observed Relative Frequency')
        ax1.set_xlabel('Predicted Probability')
        ax2 = ax1.twinx()
        ax2.hist(pred_sepval, bins = bins, alpha=0.35, density=False)
        ax2.set_yscale('log')
        ax2.set_ylabel('Histogram Count Numbers')
        
        plt.savefig('./output/reliability_scoreboard_' + model_names + '.png')

    return


def manual_sphinx(sphinx_file):
    '''
    Writing a sub-routine that will run SPHINX validation suite for a pre-existing SPHINX dataframe,
    since there is currently no way to do that.
    Inputs:
    sphinx_file: string, the path from the base sphinxval directory to the location of the dataframe
        that you want to run. 
    '''
    setup_logging()
    logger = logging.getLogger(__name__)
    df = pd.read_csv(sphinx_file)
    model_names = []
    all_energy_channels = []
    all_observed_thresholds = {}



    for i in range(len(df)):
        if df['Model'][i] not in model_names:
            model_names.append(df['Model'][i])
        if df['Energy Channel Key'][i] not in all_energy_channels:
            all_energy_channels.append(df['Energy Channel Key'][i])
            if df['Threshold Key'][i] not in all_observed_thresholds:
                all_observed_thresholds[df['Energy Channel Key'][i]] = [df['Threshold Key'][i]]
    # print(model_names, all_energy_channels, all_observed_thresholds)
    # print('Actually made it here', df)
    validation_type = ["All", "First", "Last", "Max", "Mean"]
    for type in validation_type:
        logger.info("-----------Starting validation of " + type +" forecasts-------------")
        validation.calculate_intuitive_metrics(df, model_names, all_energy_channels,
                all_observed_thresholds, type)
        # print('Looping')
    #Record explanatory information to the log
    validation.validation_explanation()
    
    logger.info("intuitive_validation: Validation process complete.")

