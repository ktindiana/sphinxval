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

scoreboard_models = ["ASPECS", "iPATH", "MagPy", "SEPMOD",
                    "SEPSTER", "SPRINTS", "UMASEP"]


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
        path/output/plots/. directory
    
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



def deoverlap_all_clear(filename, date_range_st = None, date_range_end=None,
    td=pd.NaT):
    """ For models that produce continuous forecasts on a set cadence 
        with overlapping prediction windows, deoverlap by getting a 
        single answer for a given period of time.
        
        For example, for each 24 hour period, assign a single hit, 
        miss, correct negative, or false alarm according to the 
        forecasts within that 24 hour period.
        
        INPUT:
        
            :filename: (string) filename of the all_clear_selections_model.pkl
                file output by SPHINX.
            :date_range_st: (pd date_range series) start of time periods of interest. May include a continuous time period or a set of specific time
                periods, e.g. a challenge set
                If not specified, will default to continuous time periods between
                start and end time of forecasts in input file.
            :date_range_end: (pd date_range series) end of time periods of interest. 
            :td: (datetime timedelta) if date_range_st not specified, will create
                continuous time periods with td. If date_range_st specified, but 
                date_range_end not specified, will create time periods of
                date_range_st + td. Default td = 24 hours.
                
        OUTPUT:
        
            Rederived contingency table and metrics
            :df_do: (dataframe) deoverlapped all clear values
            :all_clear_metrics_df: (dateframe) contains metrics derived
                from the deoverlapped results
        
    """
    
    df = pd.read_csv(filename, parse_dates=['Prediction Window Start', 'Prediction Window End'])
    if df.empty:
        return

    df['Observed SEP All Clear'] = df['Observed SEP All Clear'].astype(bool)
    df['Predicted SEP All Clear'] = df['Predicted SEP All Clear'].astype(bool)

    df = df.sort_values('Prediction Window Start')

    model = df["Model"].iloc[0]
    energy_key = df["Energy Channel Key"].iloc[0]
    thresh_key = df["Threshold Key"].iloc[0]
    pred_energy_key = df["Prediction Energy Channel Key"].iloc[0]
    pred_thresh_key = df["Prediction Threshold Key"].iloc[0]
    
    first_date = df['Prediction Window Start'].iloc[0]
    last_date = df['Prediction Window End'].iloc[len(df['Prediction Window End'])-1]

    print(f"First date in predictions: {first_date}")
    print(f"Last date in predictions: {last_date}")

    #CREATE DATE RANGE IF NOT INPUT
    #If date cadence or window isn't provided, set to 24 hours
    if pd.isnull(td):
        td = datetime.timedelta(hours=24)

    #If start times provided, but not end times, then apply td
    if not pd.isnull(date_range_st) and pd.isnull(date_range_end):
        date_range_end = date_range_st + td

    #If no date range provided, then create a continuous date range
    #between the first and last date in the provided forecasts
    if pd.isnull(date_range_st):
        #Specify date range covered by prediction windows
        start_date = pd.Timestamp(year=first_date.year, month=first_date.month, day=first_date.day)
        end_date = pd.Timestamp(year=last_date.year, month=last_date.month, day=last_date.day)
            
        #Create a range of daily time stamps from the start date to the end date
        date_range_st = pd.date_range(start=start_date, end=end_date-td, freq=td)
        date_range_end = date_range_st + td


    #Group forecasts and get a single value of all clear for each date range
    dict = {'Start Date': [], 'End Date': [],
            'Observed SEP Threshold Crossing Time': [],
            'Observed SEP All Clear': [],
            'Predicted SEP All Clear': [],
            'Number of Forecasts': [],
            'Number of SEP Forecasts': [],
            'First Prediction Window': [],
            'Last Prediction Window': []}

    ######################SEP EVENTS
    #Identify all the SEP events in the observations and calculate hits and
    #misses
    sep_events = resume.identify_unique(df,'Observed SEP Threshold Crossing Time')

    #Pull out all forecasts for each SEP event
    n_caught = 0
    sep_caught_str = ''
    n_miss = 0
    sep_miss_str = ''
    for sep in sep_events:
        #Check if the sep event is within the date range of interes
        check = (sep >= date_range_st) & (sep < date_range_end)
        if not check.any: continue
        st_date = date_range_st[check][0]
        end_date = date_range_end[check][0]

        sep_sub = df.loc[df['Observed SEP Threshold Crossing Time'] == sep]
        if sep_sub.empty: continue
        
        pred_win_first = sep_sub['Prediction Window Start'].iloc[0]
        pred_win_last = sep_sub['Prediction Window Start'].iloc[len(sep_sub['Prediction Window Start'])-1]
        
        hit = all_clear_any(sep_sub, False, False) #Was there a hit?
        miss = all_clear_all(sep_sub, False, True) #Did the model miss completely?
        
        print(f"SEP Event: {sep}, Model Hit?: {hit}, Model Miss?: {miss}")

        dict['Start Date'].append(st_date)
        dict['End Date'].append(end_date)
        dict['Observed SEP Threshold Crossing Time'].append(sep)
        dict['Observed SEP All Clear'].append(False)
        dict['Number of Forecasts'].append(len(sep_sub))
        dict['Number of SEP Forecasts'].append(len(sep_sub))
        dict['First Prediction Window'].append(pred_win_first)
        dict['Last Prediction Window'].append(pred_win_last)
        if hit:
            n_caught += 1
            sep_caught_str += str(sep) + ';'
            dict['Predicted SEP All Clear'].append(False)
        if miss:
            n_miss += 1
            sep_miss_str += str(sep) + ';'
            dict['Predicted SEP All Clear'].append(True)

        #Remove the forecasts associated with the SEP event from the main df
        #so don't evaluate a second time in the next loop
        df = df.loc[df['Observed SEP Threshold Crossing Time'] != sep]



    ####################REMAINING FORECASTS
    for start, end in zip(date_range_st,date_range_end):
        
        sub = df.loc[(df['Prediction Window End'] > start) & (df['Prediction Window Start'] < end)]

        if sub.empty:
            print(f"No forecasts for {start} to {end}. Continuing.")
            continue

        #Prediction window within the date range
        check_in = (sub['Prediction Window Start'] >= start) & (sub['Prediction Window End'] <= end)
        
        #Forecasts that start before the date range of interest and overlap
        #more than the previous date are included
        check_pre = ((sub['Prediction Window End'] - start) > (start - sub['Prediction Window Start'])) & (sub['Prediction Window Start'] < start)

        #Forecasts that end after the date of interest and overlap more
        #than the next date are included
        check_post = ((sub['Prediction Window End'] - end) < (end - sub['Prediction Window Start'])) & (sub['Prediction Window End'] > end)
        
        #Forecasts with prediction window extending beyond both sides of
        #the date range; if prediction window much larger than td,
        #then won't get any matches
#        check_overlap = ((sub['Prediction Window Start'] < start) & (sub['Prediction Window End'] > end)) & ((end-start) > (start - sub['Prediction Window Start'])) & ((end-start) > (sub['Prediction Window End']-end))
        
        keep = check_in | check_pre | check_post
        
        sub = sub.loc[keep]
        if sub.empty:
            print(f"No forecasts for {start} to {end}. Continuing.")
            continue

        pred_win_first = sub['Prediction Window Start'].iloc[0]
        pred_win_last = sub['Prediction Window Start'].iloc[len(sub['Prediction Window End'])-1]

        #Hits and Misses should already have been accounted for
        #Hit - check all observed not clear
        hit = all_clear_any(sub, False, False)
        #Miss  - check all observed not clear
        miss = all_clear_all(sub, False, True)


        #False Alarm  - check all observed clear
        fa = all_clear_any(sub, True, False)
        #Correct Negative  - check all observed clear
        cn = all_clear_all(sub, True, True)

        print(f"{start} - {end}: Hit: {hit}, Miss: {miss}, False Alarm: {fa}, Correct Negative: {cn}")
        
        #If an SEP event has already been recorded for this date range
        #and the rest of the forecasts are correct negatives, then only
        #want to keep the result for the SEP event. No need to record
        #a correct negative as well.
        if start in dict['Start Date'] and cn:
            idx = dict['Start Date'].index(start)
            dict['Number of Forecasts'][idx] = dict['Number of Forecasts'][idx] + len(sub)
            continue
        
        #May get multiple answers in a given time period
        dict['Start Date'].append(start)
        dict['End Date'].append(end)
        dict['Observed SEP Threshold Crossing Time'].append(pd.NaT)
        dict['Number of Forecasts'].append(len(sub))
        dict['Number of SEP Forecasts'].append(0)
        dict['First Prediction Window'].append(pred_win_first)
        dict['Last Prediction Window'].append(pred_win_last)
#        if hit:
#            dict['Observed SEP All Clear'].append(False)
#            dict['Predicted SEP All Clear'].append(False)
#        if miss:
#            dict['Observed SEP All Clear'].append(False)
#            dict['Predicted SEP All Clear'].append(True)


        if fa:
            dict['Observed SEP All Clear'].append(True)
            dict['Predicted SEP All Clear'].append(False)
        elif cn:
            dict['Observed SEP All Clear'].append(True)
            dict['Predicted SEP All Clear'].append(True)
        else:
            dict['Observed SEP All Clear'].append(None)
            dict['Predicted SEP All Clear'].append(None)
            #print(sub)


    #Deoverlapped dataframe
    df_do = pd.DataFrame(dict)
    df_do = df_do.sort_values('Start Date')
    fnameout = filename.replace('.csv','_deoverlap.csv')
    df_do.to_csv(fnameout)
    print(f"Wrote out {fnameout}.")

    #Calculate metrics
    all_clear_dict = validation.initialize_all_clear_dict()
    scores = metrics.calc_contingency_all_clear(df_do,
        'Observed SEP All Clear', 'Predicted SEP All Clear')

    
    validation.fill_all_clear_dict(all_clear_dict, model, energy_key, thresh_key,
        pred_energy_key, pred_thresh_key, scores, n_caught, sep_caught_str, n_miss,
        sep_miss_str)
        
    all_clear_metrics_df = pd.DataFrame(all_clear_dict)
    if not all_clear_metrics_df.empty:
        fout = fnameout.replace('selections','metrics')
        all_clear_metrics_df.to_csv(fout)
        print(f"Wrote out {fout}.")

    return df_do, all_clear_metrics_df




def all_clear_grid(csv_path, models, dates_file, energy_min, energy_max,
    threshold, deoverlap_models=[]):
    """ For a list of models, reads in all_clear_selections files output
        by sphinx and checks whether the model predicted a hit/miss or
        false alarm/correct negative for a set of dates, stored in df_dates.
        
        df_dates is expected to consist of two columns, labeled 
        "SEP Events" and "Non-Events".
        
        This subroutine can use either the original all_clear_selections
        files or by calling the deoverlap_all_clear subroutine and using
        the deoverlapped results.
        
        INPUT:
        
            :csv_path: (string) path the all_clear_selections csv files.
            :models: (list) list of models to be included in the grid; 
                model names must exactly match the short_name field in 
                the forecast jsons or SPHINX output files.
            :dates: (string) csv file containing list of dates that want to use to 
                generate grids with a column titled "SEP Events" with the 
                SEP start times and columns titled "Non-Event Start" 
                and "Non-Event End". 
            :energy_min: (float) low edge of energy channel of interest
            :energy_max: (float) high edge of energy channel of interest (-1)
                for infinity (>10 MeV -> energy_min = 10, energy_max = -1)
            :threshold: (float) threshold applied for SEP event definition
                (e.g. >10 MeV exceeds 10 pfu -> threshold = 10)
            :deoverlap_models: (list) list of models that need to be
                deoverlapped first; model names must exactly match the 
                short_name field in the forecast jsons or SPHINX output files.
                
        OUTPUT:
        
            Write out two csv files containing desired grid with 
            dates on one axis, models on other axis, and outcomes
            as the entries. "No Data" entries indicate a forecast
            was not provided for a given date.
         
    """
    
    #Find date range that fully spans SEP event and non-event periods for
    #models that need to be deoverlapped.
    print(f"all_clear_grid: Reading in dates file {dates_file}.")
    df_dates = pd.read_csv(dates_file, parse_dates=['SEP Events','Non-Event Start','Non-Event End'])

    sep_results = {"SEP Events": df_dates["SEP Events"].to_list()}
    nonsep_results = {"Non-Event Start": df_dates["Non-Event Start"].to_list(),
        "Non-Event End": df_dates["Non-Event End"].to_list()}
    all_clear_dict = validation.initialize_all_clear_dict()

    #Find first and last date out of all the dates
    max_date = df_dates.max().max()
    min_date = df_dates.min().min()
    
    if deoverlap_models:
        print(f"all_clear_grid: Models will be deoverlapped between {min_date} and {max_date}.")

    energy_key = f"min.{float(energy_min):.1f}.max.{float(energy_max):.1f}.units.MeV"
    print(energy_key)
    thresh_key = f"threshold_{float(threshold):.1f}"
    print(thresh_key)

    for model in models:
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

        if not os.path.isfile(fname):
            print(f"all_clear_grid: File does not exist. Check model and csv_path. {fname}. Skipping model.")
            continue

        print(f"all_clear_grid: Reading in file {fname}.")
        
        if model in deoverlap_models:
            df_ac, ac_metrics = deoverlap_all_clear(fname,
                date_range_st = min_date, date_range_end=max_date)
        else:
            df_ac = pd.read_csv(fname, parse_dates=['Observed SEP Threshold Crossing Time','Prediction Window Start','Prediction Window End'])

        #Date columns indicating SEP range
        key_st = 'Prediction Window Start'
        key_end = 'Prediction Window End'
        #Deoverlapped dataframe organized differently than original
        #all_clear_selections files.
        if model in deoverlap_models:
            key_st = 'Start Date'
            key_end = 'End Date'

        #Go through SEP Events and Non-Events and record results
        for sep, non_st, non_end in df_dates.itertuples(index=False):

            #SEP EVENTS
            #For each SEP event, get Hit/Miss/No Data
            sep_outcome = None
            if not pd.isnull(sep):
                #Get results for a single SEP event
                sub = df_ac[df_ac['Observed SEP Threshold Crossing Time'] == sep]
                print(f"{model} forecasts associated with {sep}")
                print(sub[[key_st, key_end, 'Observed SEP All Clear', 'Predicted SEP All Clear']])
                
                if sub.empty:
                    sep_outcome = 'No Data'
                else:
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
                        hit += 1
                        sep_caught_str += str(sep) + ';'
                    else:
                        sep_outcome = 'Miss'
                        miss += 1
                        sep_miss_str += str(sep) + ';'

            sep_outcomes.append(sep_outcome)
            
            #NON-EVENTS
            #For each non-event, get False Alarm/Correct Negative/No Data
            #Here we need to extract the appropriate columns
            #All forecasts with prediction windows that start at the
            #Non-Event Start dates and all predictions with start timess all the
            #way through to the End date
            nonsep_outcome = None
            if not pd.isnull(non_st):
                sub = df_ac[(df_ac[key_st] >= non_st) & (df_ac[key_st] < non_end)]
                sub = sub[sub['Observed SEP All Clear']==True]
                #Remove any forecasts in sub that are associated to a SEP
                #event (Observed SEP All Clear False), because those should
                #not be considered to be part of the non-event time period
                sub = sub[sub['Observed SEP All Clear']==True]
                print(f"{model} forecasts associated with {non_st} to {non_end}")
                print(sub[[key_st, key_end, 'Observed SEP All Clear', 'Predicted SEP All Clear']])
                
                if sub.empty:
                    nonsep_outcome = 'No Data'
                else:
                    compare = sub['Observed SEP All Clear'] == sub['Predicted SEP All Clear']
                    #Hits, any True, False columns will give a False in compare
                    if compare.all():
                        nonsep_outcome = 'CN'
                        cn += 1
                    else:
                        nonsep_outcome = 'FA'
                        fa += 1

            nonsep_outcomes.append(nonsep_outcome)

        scores = metrics.contingency_scores(hit, miss, fa, cn)
        validation.fill_all_clear_dict(all_clear_dict, model, energy_key, thresh_key,
            energy_key, thresh_key, scores, hit, sep_caught_str,
            miss, sep_miss_str)
        
        sep_results.update({model: sep_outcomes})
        nonsep_results.update({model: nonsep_outcomes})

    
    df_sep = pd.DataFrame(sep_results)
    df_sep.to_csv(os.path.join(csv_path,'all_clear_grid_SEP.csv'))
    
    df_nosep = pd.DataFrame(nonsep_results)
    df_nosep.to_csv(os.path.join(csv_path,'all_clear_grid_NonEvent.csv'))

    df_scores = pd.DataFrame(all_clear_dict)
    df_scores.to_csv(os.path.join(csv_path,'all_clear_grid_metrics.csv'))
