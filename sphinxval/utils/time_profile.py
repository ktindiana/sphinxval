from . import validation_json_handler as vjson
import numpy as np
import datetime
import os
import csv
import sys

__version__ = "0.6"
__author__ = "Phil Quinn, Kathryn Whitman"
__maintainer__ = "Kathryn Whitman"
__email__ = "kwhitman@nasa.gov"

'''Script for comparing the time-series of
    a model and observation.  Reads in the data,
    calculates the metric for each point in time,
    then plots the metric as a function of time.
    Written on 2020-07-23 by Phil Quinn. Updated
    thereafter by Katie Whitman.
'''
#2021-04-02, Changes in 0.4: Modified code to work with validation.py.
#   Added read_time_profile.
#   Removed subroutines that weren't needed but did
#   not change functionality related to calculating metrics.
#2021-04-05, Changes in 0.4.1: In interp_time, if observations
#   do not extend as far in time as predictions, will
#   set interpolation value to None. These time points should
#   be exlcluded when metrics are calculated.
#   Added remove_none subroutine.
#2021-08-21, changes in 0.5: Adding subroutine to read in a time
#   profile with zulu time in first column and only one set
#   of fluxes in the second column. CCMC SEP Scoreboard requires
#   one time profile txt file per energy channel.
#2022-02-07, changes in 0.6: Adding RMSE and RMSLE for time profile
#   assessment

def read_generic_time_profile(filename):
    '''Reads in any comma-separated time profile values.
        The first column must have the datetime in
        YYYY-MM-DD HH:MM:SS format.
        The remaining columns must have values with time.
    '''
    if not os.path.exists(filename):
        sys.exit("read_generic_time_profile: Cannot read file!! Exiting. \"" + filename +"\"")

    print('read_generic_time_profile: Reading in file ' + filename)
    dates = []
    profiles = []
    with open(filename) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        #Define arrays that hold dates
        for row in readCSV:
            if row == '': continue
            row[0] = row[0].strip()
            if row[0][0] == '#': continue
  
            #Read values into lists
            dates.append(datetime.datetime.strptime(row[0],
                                        "%Y-%m-%d %H:%M:%S"))
            if not profiles:
                profiles = [[]]*len(row[1:])
            vals = []
            for val in row[1:]:
                vals.append(float(val))
            if not profiles:
                profiles = vals
            else:
                profiles.append(vals)
                
            for i in range(len(row[1:])):
                if not profiles[i]:
                    profiles[i] = [float(row[i+1])]
                else:
                    profiles[i].append(float(row[i+1]))

    csvfile.close()
    return dates, profiles


def read_single_time_profile(filename):
    ''' Reads the flux time profile files generated by
        operational_sep_quantities.py for the CCMC SEP
        Scoreboard.

        The time profiles have zulu time in the first column
        and the flux time profile in the second column.
    '''
    dates = []
    fluxes = []
    if not os.path.exists(filename):
        print("read_time_profile: Cannot read file!! Exiting. \"" + filename +"\"")
        return dates, fluxes

    print('read_single_time_profile: Reading in file ' + filename)
    with open(filename) as ofile:
        for row in ofile:
            if row == '': continue
            row = row.lstrip().strip()
            if row[0] == "#": continue
    
            row = row.split()
            zulutime = row[0]
            dt = vjson.zulu_to_time(zulutime)
            flux = float(row[1])
            
            dates.append(dt)
            fluxes.append(flux)

    ofile.close()
    return dates, fluxes




def read_time_profile(filename):
    '''Reads in the integral flux time profile files generated by
        operational_sep_quantities.py and saved in the output directory.
        e.g. integral_fluxes_SEPMOD_multi_1hr_integral_2012_5_17.csv

        The flux time profiles files always have datetime in the first column,
        >10 and >100 MeV fluxes in the next two columns, then can have integral
        fluxes of other energies, depending on whether the user specified an
        additional threshold for calculating quantities.
    '''
    channels = []
    dates = []
    fluxes = []
    if not os.path.exists(filename):
        print("read_time_profile: Cannot read file!! Exiting. \"" + filename +"\"")
        return dates, channels, fluxes

    print('read_time_profile: Reading in file ' + filename)
    Nchan = -1
    with open(filename) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        #Define arrays that hold dates
        for row in readCSV:
            if row == '': continue

            #Read header row with column headings; specifies integral channel
            if row[0] == '#Date':
                for energy in row:
                    if energy[0] != '#': channels.append(float(energy))
                Nchan = len(channels)
                fluxes = [[]]*Nchan  #Get all the fluxes in all the columns

            if row[0][0] == '#': continue

            #Check that the correct channels were identified
            if Nchan == -1 or Nchan == 0:
                print("read_time_profile: Could not find header specifying "
                        "integral flux channels.")
                return dates, channels, fluxes

            #Read values into lists
            dates.append(datetime.datetime.strptime(row[0],
                                        "%Y-%m-%d %H:%M:%S"))
            for i in range(Nchan):
                if not fluxes[i]:
                    fluxes[i] = [float(row[i+1])]
                else:
                    fluxes[i].append(float(row[i+1]))

    csvfile.close()
    return dates, channels, fluxes


def remove_none(obs, model):
    '''Remove None values from corresponding observations and model lists.
        Only values that are real in both lists are kept.
        obs is a single list of observations
        model is a single list of model forecasts
    '''
    #Error checking
    if len(obs) != len(model):
        sys.exit('remove_none: Both input arrays must be the same length! '
                'Exiting.')
    #Clean None values from observations and remove correponding entries in
    #the model
    bad_index = [bad for bad, value in enumerate(obs) if value == None]
    obs_clean = list(obs)
    model_clean = list(model)
    for bad in sorted(bad_index, reverse=True):
        del obs_clean[bad]
        del model_clean[bad]

    #Clean None values from the model and remove correponding entries in
    #the observations
    bad_index = [bad for bad, value in enumerate(model_clean) if value == None]
    for bad in sorted(bad_index, reverse=True):
        del obs_clean[bad]
        del model_clean[bad]

    return obs_clean, model_clean


def remove_zero(obs, model):
    '''Remove None values from corresponding observations and model lists.
        Only values that are real in both lists are kept.
        obs is a single list of observations
        model is a single list of model forecasts
    '''
    #Error checking
    if len(obs) != len(model):
        sys.exit('remove_none: Both input arrays must be the same length! '
                'Exiting.')
    #Clean None values from observations and remove correponding entries in
    #the model
    bad_index = [bad for bad, value in enumerate(obs) if value == 0]
    obs_clean = list(obs)
    model_clean = list(model)
    for bad in sorted(bad_index, reverse=True):
        del obs_clean[bad]
        del model_clean[bad]

    #Clean None values from the model and remove correponding entries in
    #the observations
    bad_index = [bad for bad, value in enumerate(model_clean) if value == 0]
    for bad in sorted(bad_index, reverse=True):
        del obs_clean[bad]
        del model_clean[bad]

    return obs_clean, model_clean


def interp_timeseries(x_true, y_true, scale_true, x_pred):
    """
    Uses interpolation to get the observational data point
    for each corresponding model data point
    
    Need to use a custom function since pre-made interp
    functions can't handle datetime objects
    
    Parameters
    ----------
    x_true : array-like
        Observed (true) datetime objects
    
    y_true : array-like
        Observed (true) values
    
    scale_true : string
        Scale of the observed (true) values
        Can only be "linear" or "log"
    
    x_pred : array-like
        Forecasted (estimated) datetime objects
    
    Returns
    -------
    ynew : array-like
        Observed values interpolated onto forecasted scale
    """
    
    if scale_true == "log":
        y_true = np.log10(y_true)
    elif scale_true == "linear":
        pass
    else:
        raise ValueError("scale_true must be 'linear' or 'log'.")
    
    ynew = []
    for i in range(len(x_pred)):
        y = None #prediction dates may extend past observation dates
        for j in range(1, len(x_true)):
            
            if x_true[j] > x_pred[i]:
                #s = (x_true[j] - x_pred[i])/(x_true[j] - x_true[j-1])
                #y = y_true[j-1]*s + y_true[j]*(s-1)
                
                s = (x_pred[i] - x_true[j-1])/(x_true[j] - x_true[j-1])
                y = y_true[j-1]*(1.0-s) + y_true[j]*s
                break
    
        if y != None:
            if scale_true == "log":
                y = 10**y
    
        ynew.append(y)
    
    return ynew


def get_error(date_obs, flux_obs, date_mod, flux_mod, metric):
    """
    Calculates error between model and observation time profiles
    for each non-zero time point in the model
    
    Parameters
    ----------
    date_obs : array-like
        Dates of observations
    
    flux_obs : array-like
        Flux of observations
    
    date_mod : array-like
        Dates of model
    
    flux_mod : array-like
        Flux of model
    
    metric : string
        Abbreviation of metric to calculate
        Must be a metric abbreviation found in metrics.py 
    
    Returns
    -------
    date_mod : array-like
        Dates of model corresponding to times of non-zero flux
    
    error : array-like
        Error between model and observations for times of non-zero flux
    """
    
    # getting rid of zeroes
    flux_mod, date_mod = zip(*filter(lambda x:x[0]>0.0, \
                             zip(flux_mod, date_mod)))
    
    # interpolating obs flux onto model time
    flux_obs_interp = interp_timeseries(date_obs, flux_obs, "log", date_mod)
    
    #Check for None values and remove
    flux_obs_clean, flux_mod_clean = remove_none(flux_obs_interp,flux_mod)
    
    # calculating error
    error = metrics.switch_error_func(metric, flux_obs_clean, flux_mod_clean)
    
    return date_mod, error


def get_mean_or_coeff(error, metric, scale="log"):
    """
    Returns mean error or correlation coefficient
    
    Parameters
    ----------
    error : array-like
        Error between model and observations
    
    metric : string
        Metric used to calculate error
    
    scale : string
        Scale of quantity ("linear" or "log")
        Optional. Defaults to log
    
    Returns
    -------
    ret : float
        Mean error or corellation coefficient
    """
    
    if metric == "r":
        if scale == "log":
            ret = np.round(error[1], 2)
        else:
            ret = np.round(error[0], 2)
    else:
        ret = np.round(metrics.calc_mean(error), 2)
    
    return ret




def create_plots(keys, dict, metric, model, exp_name):
    """
    Creates plots of error profiles for each event, integral energy,
    across available model subtypes
    
    Parameters
    ----------
    keys : array-like
        First set of keys for dict
    
    dict : dictionary
        Dictionary of error time profiles and means
    
    metric : string
        Abbreviation of metric to calculate
        Must be a metric abbreviation found in metrics.py
    
    model : string
        Model name

    exp_name : string
        Name of observational experiment
    
    Returns
    -------
    None
    """
    
    for event in keys:
        for energy in gl.str_op_thresh:
            dates = []
            errors = []
            labels = []
            means = []
            for mod in gl.model_keys:
                if mod in dict[event][energy[0]].keys():
                    dates.append(dict[event][energy[0]][mod]["date"])
                    errors.append(dict[event][energy[0]][mod]["error"])
                    labels.append(mod)
                    means.append(dict[event][energy[0]][mod]["mean"])
                    
            save_name = "metric_profile_" + model + "_" + \
                        exp_name + "_" + energy[0] + "_" + \
                        event
            
            ptools.plot_metric_profile(dates, errors, labels, means, \
                                title=event, save=save_name, y_label=metric, \
                                closeplot=True)
        

def create_box_plots(keys, dict, model, exp_name, metric, mean_metric=False, \
                     print_means=False):
    """
    Sets everything up for creating box plots of the error
    
    Parameters
    ----------
    keys : array-like
        First set of keys for dict
    
    dict : dictionary
        Dictionary of error time profiles and means
    
    model : string
        Model name

    exp_name : string
        Name of observational experiment
    
    metric : string
        Abbreviation of metric to calculate
        Must be a metric abbreviation found in metrics.py
    
    mean_metric : boolean
        Boolean for changing the metric to mean metric for labeling in plots
        Optional.  Defaults to False
    
    print_means : boolean
        Boolean for printing the mean error values to screen
        Optional.  Defaults to False
    
    Returns
    -------
    None
    """
    
    for energy in gl.str_op_thresh:
        means = []
        names = []
        for mod in gl.model_keys:
            if mod=='SEPMOD_1hr' or mod=='SEPMOD_2p5hr':
                pass
            else:
                means_mod = []
                for event in keys:
                    try:
                        means_mod.append(dict[event][energy[0]][mod]["mean"])
                        if print_means:
                            print(event, energy[0], mod, \
                                  dict[event][energy[0]][mod]["mean"])
                    except:
                        pass
                
                means.append(means_mod)
                names.append(dict[event][energy[0]][mod]["name"])
                
        save_name = "plots/TimeProfileError/" + "box_" + model + "_" + \
                    exp_name + "_" + energy[0] + "_" + metric
        title_name = "$\geq$" + energy[0] + " MeV"
        if mean_metric:
            ylabel = "M" + metric
        else:
            ylabel = metric
        
        # box plots
        ptools.box_plot(means, names, save=save_name, title=title_name, \
                 x_label="Model", y_label=ylabel, closeplot=True)


def combine_time_profiles(dates, profiles):
    """ For predictions windows that match with multiple observation files,
        take multiple time profiles and string them together into a single
        time profile.
        
        ***FOR NOW, ASSUME NO OVERLAP BETWEEN TIME PROFILES. THIS MEANS
        THAT OBSERVATIONS MUST BE PREPARED WITH NO OVERLAPPING DATES.
        
        INPUTS:
        
        :dates: (datetime mxn list) m lists of n dates each; n
            may vary
        :profiles: (float mxn list) m profiles that are n dates long
        
    """
    
    if dates == [] or profiles == []:
        return [],[]
    
    nprof = len(dates)
    full_dates = []
    full_profile = []
    
    firstdates = []
    for i in range(nprof):
        firstdates.append(dates[i][0])
    
    idx = np.argsort(firstdates)
    for i in range(nprof):
        full_dates.extend(dates[idx[i]])
        full_profile.extend(profiles[idx[i]])
        
    return full_dates, full_profile
