from . import config as cfg
from . import plotting_tools
from . import units_handler as vunits
from . import metrics_dicts
from . import plotting_tools as plt_tools
import argparse
import numpy as np
import datetime
import math
from itertools import zip_longest
import matplotlib.pylab as plt
from pandas.plotting import register_matplotlib_converters
from matplotlib.dates import DateFormatter
from sklearn.utils.validation import check_consistent_length
from sklearn.utils.validation import check_array
from scipy.stats import pearsonr
from scipy.optimize import curve_fit 
from lmfit import minimize, Parameters, fit_report
import os
import sys
import re
import csv
import seaborn as sns
from pathlib import Path
import logging
import pandas as pd

"""
Fluence Spectrum validation has its own dedicated python file to house its subroutines
to keep the length of validation.py in check. Additionally this file will act in similar
ways to metrics.py since it will be calculating metrics. 


The first part of this code involves fitting the fluence spectrum to a Band Function or an Ellison Ramaty 
(powerlaw with an exponential rollover) and comparing the fit parameters for the observations and the model. 
The idea is that the physics can more easily be seen in the fit parameters (rollover energy, 
low energy spectral slope, high energy spectral slope) compared to just looking at the error in individual
energy bins. Looking at how these fit parameters vary across the event set can tell you if the physics 
contained within a fluence spectrum is being appropriately modeled (rollover energy and spectral slope).

Second part of handles a simpler error calculation between the model energy bins individual fluence
and the observed fluence for the same energy bin, interpolated if required.

"""

#Create logger
logger = logging.getLogger(__name__)
mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)

def from_differential_to_integral_flux(min_energy, energy_bins,
    fluxes, energy_bin_centers=[]):
    """ If user selected differential fluxes, convert to integral fluxes to
        caluculate operational threshold crossings (>10 MeV protons exceed 10
        pfu, >100 MeV protons exceed 1 pfu).
        Assume that the measured fluxes correspond to the center of the energy
        bin and use power law interpolation to extrapolate integral fluxes
        above user input min_energy.
        The intent is to calculate >10 MeV and >100 MeV fluxes, but leaving
        flexibility for user to define the minimum energy for the integral flux.
        An integral flux will be provided for each timestamp (e.g. every 5 mins).

        Note that if bad values were not interpolated in previous steps,
        they will have been set to None in check_for_bad_data, which translated
        to NaN in numpy arrays.

       If GOES-13, -14, -15, then the 110 - 900 MeV bin is not included when
       estimating the integral fluxes .
       This bin's energy appears to be unreliable and the other energy
       bins cover this range.

        INPUTS:

        :min_energy: (float) - bottom energy for integral flux calculation
        :energy_bins: (float nx2 array) - bins for each energy channel
            [[Emin1, Emax1], [Emin2, Emax2], [], ...]
        :fluxes: (float nxm array) - fluxes with time for each energy channel
       

        OUTPUTS:

        :integral_fluxes: (float 1xm array) - estimate integral flux for >min_energy
            (Returns all zero values if no energy bins above min_energy)

    """
    print(f"Converting differential flux to integral flux for >{min_energy} MeV.")
    nbins = len(energy_bins)
    nflux = len(fluxes[0])
    #Check requested min_energy inside data energy range
    if min_energy < energy_bins[0][0] or min_energy >= energy_bins[nbins-1][0]:
        print(f"The selected minimum energy {min_energy} to create "
               + "integral fluxes is outside of the range of the data: "
                + f"{energy_bins[0][0]}  - {max(energy_bins[nbins-1][0],energy_bins[nbins-1][1])}")
        print(f"Setting all >{min_energy} fluxes to zero.")
        integral_fluxes = [0]*nflux
        return integral_fluxes

    #Calculate bin center in log space for each energy bin
    bin_center = []
    if len(energy_bin_centers) != 0:
        bin_center = energy_bin_centers
    else:
        for i in range(nbins):
            if energy_bins[i][1] != -1:
                centerE = math.sqrt(energy_bins[i][0]*energy_bins[i][1])
            else:
                centerE = -1 #integral bin
            bin_center.append(centerE)

    #The highest energy EPEAD bin overlaps with all of the HEPAD bins
    #For this reason, excluding the highest energy EPEAD bin in
    #integral flux estimation, 110 - 900 MeV
    if (experiment == "GOES-13" or experiment == "GOES-14" or
        experiment == "GOES-15"):
        remove_bin = -1
        for i in range(nbins):
            if energy_bins[i][0] == 110 and energy_bins[i][1] == 900:
                remove_bin = i
        if remove_bin == -1:
            sys.exit("Attempting to remove 110 - 900 MeV bin for "
                    + experiment + '. Cannot locate bin. Please check '
                    'define_energy_bins to see if GOES-13 to GOES-15 bins '
                    'include 110 - 900 MeV. if not please comment out this '
                    'section in tools.py from_differential_to_integral_flux.')
        fluxes = np.delete(fluxes,remove_bin,0)
        energy_bins = np.delete(energy_bins,remove_bin,0)
        bin_center = np.delete(bin_center,remove_bin,0)
        nbins = nbins-1

    #integrate by power law interpolation; assume spectrum the shape of a
    #power law from one bin center to the next. This accounts for the case
    #where the minimum energy falls inside of a bin or there are overlapping
    #energy bins or gaps between bins.
    #An energy bin value of -1 (e.g. [700,-1]) indicates infinity - already an
    #integral channel. This happens for HEPAD. If the integral channel is the
    #last channel, then the flux will be added. If it is a middle bin, it will
    #be skipped.
    integral_fluxes = []
    integral_fluxes_check = []
    for j in range(nflux):  #flux at each time
        sum_flux = 0
        ninc = 0 #number of energy bins included in integral flux estimate
        for i in range(nbins-1):
            if bin_center[i+1] < min_energy:
                continue
            else:
                if energy_bins[i][1] == -1 or energy_bins[i+1][1] == -1:
                    #bin is already an integral flux, e.g. last HEPAD bin
                    continue
                if pd.isnull(fluxes[i,j]) or pd.isnull(fluxes[i+1,j]): #data gap
                    continue

                if fluxes[i,j] < 0 or fluxes[i+1,j] < 0: #bad data
                    sys.exit("from_differential_to_integral_flux: "
                            + f"Bad flux data value of {fluxes[i,j]} and "
                            + f"{fluxes[i+1,j]} found for bin {i}, {j}. "
                            + "This should not happen. Did you call check_for_bad_data() first?")


                if fluxes[i,j] == 0 or fluxes[i+1,j] == 0: #add 0 flux
                    ninc = ninc + 1
                    continue

                F1 = fluxes[i,j]
                F2 = fluxes[i+1,j]
                if F1 == 0 or pd.isnull(F1):
                    sys.exit("from_differential_to_integral_flux: found bin flux of "
                            + f"{fluxes[i,j]}. Should not happen here, "
                            + f"bin [i,j] [{i},{j}]." )

                if F2 == 0 or pd.isnull(F2):
                    sys.exit("from_differential_to_integral_flux: found bin flux of "
                            + f"{fluxes[i+1,j]}. Should not happen here, "
                            + f"bin [i+1,j] [{i+1},{j}]." )

                logF1 = np.log(F1)
                logF2 = np.log(F2)
                logE1 = np.log(bin_center[i])
                logE2 = np.log(bin_center[i+1])
                endE = bin_center[i+1]
                if i+1 == nbins-1:
                    endE = energy_bins[nbins-1][1] #extend to edge of last bin

                f = lambda x:exp(logF1
                            + (np.log(x)-logE1)*(logF2-logF1)/(logE2-logE1))
                startE = max(bin_center[i],min_energy)
                fint = scipy.integrate.quad(f,startE,endE)
                if math.isnan(fint[0]):
                    print("from_differential_to_integral_flux: flux integral"
                        "across bins is NaN. Setting to zero. Bin values are "
                        + str(F1) + ' and ' + str(F2))
                    fint = [0]
                if fint[0] < 1e-10:
                    fint = [0]
                sum_flux = sum_flux + fint[0]
                ninc = ninc + 1

        #if last bin is integral, add (HEPAD)
        if energy_bins[nbins-1][1] == -1 and fluxes[nbins-1,j] >= 0:
            intflx = fluxes[nbins-1,j]
            sum_flux = sum_flux + intflx
            ninc = ninc + 1

        if ninc == 0:
            sum_flux = np.nan
        integral_fluxes.append(sum_flux)

    return integral_fluxes

def make_diff_from_int(integral_fluxes, energy_bins):
    """ Make differential fluxes from integral channels.
        Subtract each integral flux channel from the consecutive
        one to create differential bins.
    """

    new_bins = []
    nchan = len(integral_fluxes)
    nflx = len(integral_fluxes[0])
    diff_fluxes = np.zeros((nchan-1,nflx))
    diff_bins = np.zeros((nchan-1,2))
    
    for i in range(nchan-1):
        diff_bins[i][0] = energy_bins[i][0]
        diff_bins[i][1] = energy_bins[i+1][0]
        bin_width = diff_bins[i][1] - diff_bins[i][0]
        for j in range(nflx):
            diff_fluxes[i][j] = \
                (integral_fluxes[i][j] - integral_fluxes[i+1][j])/bin_width
            
    
    logger.info("make_diff_from_int converted to energy bins " + str(diff_bins))
    
    return diff_bins, diff_fluxes


def differential_to_int_fluence(fluence_array):
    # Quick and easy calculation of integral fluence from
    # differential fluence:
    # Add all fluence of higher energies to the lower energies fluence
    int_fluence_array = []
    for i in range(len(fluence_array)):
        int_fluence_array.append(sum(fluence_array[i:]))

    return int_fluence_array



#################### Ellison Ramaty Function Subroutines ##################################################
def fit_ellisonramaty_function(energy_bins, fluence_array, date, fig_obj):
    """
    Fitting the Ellison-Ramaty Function to energy spectra
    Should work with observations or models 
    Inputs:
    energy_bin : array of the same size as the fluence array - contains energy bins (x axis of spectra)
    fluence_array: array of the same size as energy_bin -  contains the fluences corresponding to the
        energy bins (y axis of spectra)
    energy: array to fit the final parameters to, creating a line that'll be plotted in the main function

    Outputs:
    minimize_er - a MinimizerResult from lmfit that contains all of the best fit parameters and their
        uncertanties - use minimize_er.params.valuesdict() after the function call in the main function
        to extract these values
    fit_er -  the final fit corresponding to the energy array - used for final plotting
    transition_er - the roll_over energy of the ellison-ramaty fit
    """
    #from lmfit import minimize, Parameters, Model
    import lmfit
    import numpy as np

    # new testing
    model = lmfit.Model(ellisonramaty_function)
    params = model.make_params()
    params.add('norm', value = 10**9, min = 10**4, max = 10**12)
    params.add('gamma_er', value = 1.25, min = 0.1, max =10)
    params.add('E_r', value = 35.0, min = 1.0, max = 2000)
    

    


    def fitting_routine(p, x, y):
        fit_val = ellisonramaty_function(x, p['norm'], p['gamma_er'], p['E_r'])


        
        fit_val_safe = np.clip(fit_val, 1, None)
        y_safe = np.clip(y, 1, None)
        r = 0.20 # r is an error estimator that I'm using as a constant 20% across the spectra
        sigma_log10 = r / (np.log(10))
        
        return (np.log10(y_safe) - np.log10(fit_val_safe)) / sigma_log10
        

    fitter = lmfit.Minimizer(fitting_routine, params, fcn_args=(energy_bins, fluence_array))
    result_b = fitter.minimize(max_nfev= cfg.fit_iter)
    plot_er_func(result_b.params, date, fig_obj)
    print('fit report', energy_bins, lmfit.fit_report(result_b))

    return result_b



def plot_er_func(first_guess, date, figure_obj):
    """
    Defining the Ellison-Ramaty function for plotting the fit
    """
    energy = np.arange(0,2000,0.1) # Creating an array of energies across all bins (0 - 2000 MeV) to use for plotting the end fit
    parvals = first_guess.valuesdict()
    norm = parvals['norm']
    gamma = parvals['gamma_er']
    E_r = parvals['E_r']
   
    fit = ellisonramaty_function(energy, norm, gamma, E_r)

    figure_obj.plot(energy, fit)
    figure_obj.set_title('Ellison-Ramaty Fit ' + str(date))
    return fit, E_r



def ellisonramaty_function(x, norm, gamma_er, E_r):
    return norm*x**(-gamma_er)*np.exp(-x/E_r)


###################### Band Function Subroutines ##############################################
def fit_band_function(energy_bins, fluence_array, date, fig_obj):
    """ Fitting the Band function to spectral data - see
    description of fit_ellisonramaty_function for full details of 
    inputs/outputs
    """
    import lmfit
    import numpy as np
    params_band = Parameters()
    params_band.add('norm', value = 10**5, min = 10**4, max = 10**12)
    params_band.add('gamma_a', value = 0.70, min = 0.1, max =4)
    params_band.add('gamma_b', value = 5.0, min = 4.1, max = 10)
    params_band.add('E_0', value = 70.0, min = 10.0, max = 1000)
    
    

    
    def fitting_routine(p, x, y):
        model_val = band_function(x, p['norm'], p['gamma_a'], p['gamma_b'], p['E_0'])
        model_val_safe = np.clip(model_val, 1, None)
        y_safe = np.clip(y, 1, None)
        r = 0.20 # r is an error estimator that I'm using as a constant 20% across the spectra
        sigma_log10 = r / (np.log(10))
        return (np.log10(y_safe) - np.log10(model_val_safe)) / sigma_log10

    fitter = lmfit.Minimizer(fitting_routine, params_band, fcn_args=(energy_bins, fluence_array))
    result_b = fitter.minimize(max_nfev= cfg.fit_iter)
    plot_band_func(result_b.params, date, fig_obj)
    print('fit report', energy_bins, lmfit.fit_report(result_b))

    return result_b


def band_function(x_bins, norm, gamma_a, gamma_b, E_0):

    x_break = (gamma_b - gamma_a) * E_0
    low_e_val = norm * (x_bins ** -gamma_a) * np.exp(-x_bins / E_0)
    high_e_factor = (x_break ** (gamma_b - gamma_a)) * np.exp(gamma_a - gamma_b)
    high_e_val = norm * (x_bins ** -gamma_b) * high_e_factor

    return np.where(x_bins <= x_break, low_e_val, high_e_val)


def plot_band_func(first_guess, date, figure_obj):#, energy):
    """
    Defining the Band function for plotting the fit
    """
    energy = np.arange(0,2000,0.1) # Creating an array of energies across all bins (0 - 2000 MeV) to use for plotting the end fit

    parvals = first_guess.valuesdict()
    norm =  parvals['norm']
    gamma_a = parvals['gamma_a']
    gamma_b = parvals['gamma_b']
    E_0 = parvals['E_0']

    band = band_function(energy, norm, gamma_a, gamma_b, E_0)

    figure_obj.plot(energy, band)
    figure_obj.set_title('Band Function Fit ' + str(date))
    return band


###################### Powerlaw Function Subroutines ##############################################
def fit_single_powerlaw(energy_bins, fluence_array, date, fig_obj):
    """ Fitting the a powerlaw to spectral data - see
    description of fit_ellisonramaty_function for full details of 
    inputs/outputs
    """
    import lmfit
    import numpy as np
    params_powerlaw = Parameters()
    params_powerlaw.add('norm', value = 10**5, min = 10**4, max = 10**12)
    params_powerlaw.add('gamma', value = 0.70, min = 0.1, max =10)
    
    

    
    def fitting_routine(p, x, y):
        model_val = powerlaw_function(x, p['norm'], p['gamma'])
        model_val_safe = np.clip(model_val, 1, None)
        y_safe = np.clip(y, 1, None)
        r = 0.20 # r is an error estimator that I'm using as a constant 20% across the spectra
        sigma_log10 = r / (np.log(10))
        return (np.log10(y_safe) - np.log10(model_val_safe)) / sigma_log10

    fitter = lmfit.Minimizer(fitting_routine, params_powerlaw, fcn_args=(energy_bins, fluence_array))
    result_b = fitter.minimize(max_nfev= cfg.fit_iter)
    plot_powerlaw_func(result_b.params, date, fig_obj)
    print('fit report', energy_bins, lmfit.fit_report(result_b))

    return result_b

def powerlaw_function(x, norm, gamma):
    return norm*x**(-gamma)


def plot_powerlaw_func(first_guess, date, figure_obj):#, energy):
    """
    Defining the Powerlaw function for plotting the fit
    """
    energy = np.arange(0,2000,0.1) # Creating an array of energies across all bins (0 - 2000 MeV) to use for plotting the end fit

    parvals = first_guess.valuesdict()
    norm =  parvals['norm']
    gamma = parvals['gamma']

    powerlaw = powerlaw_function(energy, norm, gamma)

    figure_obj.plot(energy, powerlaw)
    figure_obj.set_title('Single Powerlaw Fit ' + str(date))
    return powerlaw

def fluence_histograms(obs_specs, pred_specs, model, energy_key, thresh_fnm, validation_type):
    error_dictionary = {}
    x_loc = 0
    
    #iterate through all spectra, interpolate observations to model energies, find log error between the two for
    # each energy. Save in dictionary
    for m in range(len(pred_specs)):
        model_fluence_array = []
        model_energy_bin = []
        model_bin_min = []
        model_bin_max = []
        diff_bin_dictionary = {}
        for n in range(len(pred_specs[m])):
            
            if pred_specs[m][n]['energy_max'] == pred_specs[m][n]['energy_min'] and obs_specs[m][0]['energy_max'] == -1 \
                or pred_specs[m][n]['energy_max'] == -1 and obs_specs[m][0]['energy_max'] == -1:
                    if m == 0 or str(pred_specs[m][n]['energy_min']) not in list(error_dictionary.keys()):
                        error_dictionary[str(pred_specs[m][n]['energy_min'])] = []
                    model_energy_bin.append(pred_specs[m][n]['energy_min'])
                    model_fluence_array.append(pred_specs[m][n]['fluence'])
            elif pred_specs[m][n]['energy_max'] != -1 and obs_specs[m][n]['energy_max'] != -1:
                string_bin = str(pred_specs[m][n]['energy_min']) + '-' + str(pred_specs[m][n]['energy_max'])
                if m == 0 or string_bin not in list(error_dictionary.keys()):
                        error_dictionary[string_bin] = []
                if string_bin not in list(diff_bin_dictionary.keys()):
                    diff_bin_dictionary[string_bin] = (pred_specs[m][n]['energy_max']+pred_specs[m][n]['energy_min'])/2
                model_energy_bin.append((pred_specs[m][n]['energy_max']+pred_specs[m][n]['energy_min'])/2)
                model_bin_min.append(model_energy_bin[m][n] - pred_specs[m][n]['energy_min'])
                model_bin_max.append(pred_specs[m][n]['energy_max'] - model_energy_bin[m][n])
                model_fluence_array.append(pred_specs[m][n]['fluence'])
        


        
        obs_fluence_array = []
        obs_energy_bin = []
        obs_bin_min = []
        obs_bin_max = []
        
        n = 0
        for n in range(len(obs_specs[m])):
            # if integral flux turn to differential flux (rudimentary for now)
            if obs_specs[m][n]['energy_max'] == obs_specs[m][n]['energy_min'] or obs_specs[m][n]['energy_max'] == -1:
                obs_energy_bin.append(obs_specs[m][n]['energy_min'])
                obs_fluence_array.append(obs_specs[m][n]['fluence'])
            else:
                obs_energy_bin.append((obs_specs[m][n]['energy_max']+obs_specs[m][n]['energy_min'])/2)
                # obs_bin_min.append(obs_energy_bin[m][n] - obs[m][n]['energy_min'])
                # obs_bin_max.append(obs[m][n]['energy_max'] - obs_energy_bin[m][n])
                obs_fluence_array.append(obs_specs[m][n]['fluence'])
        if len(obs_fluence_array) != len(obs_specs[m]):
            obs_fluence_array.append(obs_specs[m][n+1]['fluence'])
            obs_energy_bin.append(obs_specs[m][n]['energy_min'])
            
        
    
        interpolated_obs_fluence, interpolated_bins = interpolate_fluence(obs_energy_bin, obs_fluence_array, model_energy_bin, model_fluence_array)
        
    
        fluence_error, error_bins = error_fluence_calc(interpolated_bins, interpolated_obs_fluence, model_energy_bin, model_fluence_array)
        for bins in error_bins:
            if diff_bin_dictionary != {}:
                
                current_bin = str(next((k for k, v in diff_bin_dictionary.items() if v == bins), None))
            else:
                current_bin = str(bins)
            error_dictionary[current_bin].append(fluence_error[str(bins)])


    fig1 = plt.figure(figsize=(10,8),  layout='constrained')
    ax = plt.subplot(111)
    # Now that all errors have been found/calcuated. Run through histogram generation for plotting    
    # bins_hist = [-4,-3.5,-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3,3.5,4]
    
    bin_edges = [-4, -3.5, -3.0, -2.5, -2.0, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
    hist_range = (np.min(bin_edges), np.max(bin_edges))
    for bins in list(error_dictionary.keys()):
        if error_dictionary[bins] == []:
            continue
        
        filtered_list = [x for x in error_dictionary[bins] if x is not None]
        vert_hist = np.histogram(filtered_list, range = hist_range, bins=16)[0]#/len(filtered_list)

        
        normalized_histogram = vert_hist# / np.max(vert_hist)
        
        binned_maximums = np.max(normalized_histogram)
                        
                        
        lefts = x_loc      
        heights = np.diff(bin_edges)
        centers = bin_edges[:-1]  + heights / 2
        ax.barh(centers, normalized_histogram, height=heights, left = lefts, label = bins)

        x_loc = x_loc + binned_maximums + 2
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 'small')
    plt.grid(visible=True, which='major', axis='y')
    plt.xticks([])
    plt.ylabel('Log Error')
    plt.xlabel('Energy Bins (MeV)')
    plt.title('Histograms of ' + model + ' Fluence Log Error for Each Energy Bin')
    figname = cfg.outpath + '/plots/Fluence_LE_histogram_' + model + "_" + validation_type + ".pdf"
    # plt.show()
    plt.savefig(figname, dpi=300)
    plt.close()



    return figname, error_dictionary



def interpolate_fluence(obs_energy_bin, obs_fluence_array, model_energy_bin, model_fluence_array):
    """ 
    Interpolating the obs. energy bins to match to model energy bins 
    to find fluence error 
    """
    nbin_obs = len(obs_energy_bin)
    nbin_mod = len(model_energy_bin)
    # print(obs_fluence_array)
    new_obs_fluence = []
    new_bins = []
    i = 0
    for i in range(nbin_obs):
        if obs_fluence_array[i] == 0.0:
            obs_fluence_array[i] = 1.0
        try:
            if obs_fluence_array[i+1] == 0.0:
                obs_fluence_array[i+1] = 1.0
        except:
            continue
        
        rise =np.log10(obs_fluence_array[i+1]) - np.log10(obs_fluence_array[i]) 
        run = np.log10(obs_energy_bin[i+1]) - np.log10(obs_energy_bin[i])

        slope = rise / run
        k = 0
        for k in range(nbin_mod):
            # print(k)
            if model_energy_bin[k] >= obs_energy_bin[i] and model_energy_bin[k] < obs_energy_bin[i+1]:
                new_bins.append(model_energy_bin[k])
                new_obs_fluence.append(obs_fluence_array[i]*(model_energy_bin[k]/obs_energy_bin[i])**(slope))
                
        
        #     k += 1
        # i += 1

    # This replaces any 0s with 1s b/c we take the log10 of the fluence elements for
    # the LE calc so log10(0) is undefined whereas log10(1) = 0
    if any(new_obs_fluence) == 0.0:
        for element in new_obs_fluence:
            new_obs_fluence[element] = float(1) 
               
    return new_obs_fluence, new_bins


def error_fluence_calc(obs_energy_bin, obs_fluence, model_energy_bin, model_fluence):
    """
    Use interpolated obs_energy_bins from interpolate_fluence 

    Calculates the Log Error of the Fluence between the model fluence and 
    the observed fluence for each of the model's energy bins
    """ 
    fluence_error = {}
    error_bins = []
    i = 0
    for i in range(len(obs_energy_bin)):
        for j in range(len(model_energy_bin)):
            if obs_energy_bin[i] == model_energy_bin[j]:
                # print(model_energy_bin[j])
                error_bins.append(model_energy_bin[j])
                fluence_error[str(model_energy_bin[j])] = np.log10(model_fluence[j])-np.log10(obs_fluence[i])
                if fluence_error[str(model_energy_bin[j])] == np.inf or fluence_error[str(model_energy_bin[j])] == -np.inf:
                    fluence_error[str(model_energy_bin[j])] = None
        i += 1
    


    return fluence_error, error_bins



def fitting_procedure(fit_dict, current_event_df, energy_key, date, thresh_key, validation_type):
    print(current_event_df)
    # input()
    obs_spectrum = current_event_df._11 # Observed SEP Fluence Spectrum
    pred_spectrum = current_event_df._13 # Predicted SEP Fluence Spectrum
    obs_units = current_event_df._12 # Observed SEP Fluence Spectrum Units
    pred_units = current_event_df._14 # Predicted SEP Fluence Spectrum Units
    print('obs units', obs_units, 'pred units', pred_units)

    # Will always convert model spectrum to the units of the observations - typically from differential to integral
    if obs_units != pred_units or cfg.model_diff_to_int_boolean: # boolean condition is temporary, since ipath does not have integral bins but some of its units are listed as integral incorrectly
        print('In this loop')
        pred_bins = [n['energy_min'] for n in pred_spectrum]
        obs_bins = [n['energy_min'] for n in obs_spectrum]
        obs_fluence_spectrum = [n['fluence'] for n in obs_spectrum]
        pred_fluence_spectrum = [n['fluence'] for n in pred_spectrum]
        pred_fluence_spectrum  = differential_to_int_fluence(pred_fluence_spectrum)

        # return fit_dict
    else:
        if 'MeV' in vunits.convert_units_to_string(obs_units): #differential so use bin means with widths
            obs_bins = [(n['energy_min']+n['energy_max'])/2 for n in obs_spectrum]
            obs_fluence_spectrum = [n for n in obs_spectrum['fluence']]      
        elif 'cm' in vunits.convert_units_to_string(obs_units): #integral so use energy_min as bins
            obs_bins = [n['energy_min'] for n in obs_spectrum]
            obs_fluence_spectrum = [n['fluence'] for n in obs_spectrum]
        
        if 'MeV' in vunits.convert_units_to_string(pred_units): #differential so use bin means with widths
            pred_bins = [(n['energy_min']+n['energy_max'])/2 for n in pred_spectrum]
            pred_fluence_spectrum = [n['fluence'] for n in pred_spectrum] 
        elif 'cm' in vunits.convert_units_to_string(pred_units): #integral so use energy_min as bins
            pred_bins = [n['energy_min'] for n in pred_spectrum]
            pred_fluence_spectrum = [n['fluence'] for n in pred_spectrum]

    obs_bins = np.array(obs_bins)
    pred_bins = np.array(pred_bins)
    obs_fluence_spectrum = np.array(obs_fluence_spectrum)
    pred_fluence_spectrum = np.array(pred_fluence_spectrum)

    fit_dict['Model'].append(current_event_df.Model)
    fit_dict['Energy Channel'].append(current_event_df._1)
    fit_dict['Threshold'].append(current_event_df._2)
    fit_dict['Prediction Energy Channel'].append(current_event_df._4)
    fit_dict['Prediction Threshold'].append(current_event_df._5)
    fit_dict['Predicted SEP Fluence Spectrum'].append(pred_spectrum)
    fit_dict['Predicted SEP Fluence Spectrum Units'].append(pred_units)
    fit_dict['Observed SEP Fluence Spectrum'].append(obs_spectrum)
    fit_dict['Observed SEP Fluence Spectrum Units'].append(obs_units)
    functions = ['Band Function', 'Ellison Ramaty', 'Powerlaw'] #  
    for f in functions:
        fig, ax = plt.subplots(figsize=(12, 8))

        plt.scatter(obs_bins, obs_fluence_spectrum, color = 'blue', label = 'Observed')        
        obs_fit_output= switch_fit_func(f, obs_bins, obs_fluence_spectrum, date, ax)
        plt.scatter(pred_bins, pred_fluence_spectrum, color = 'orange', label = 'Predicted')
        pred_fit_output = switch_fit_func(f, pred_bins, pred_fluence_spectrum, date, ax)
        if obs_fit_output == None or pred_fit_output == None:
            continue
        print(obs_fit_output.chisqr, obs_fit_output.redchi, pred_fit_output.chisqr, pred_fit_output.redchi)
        plt.yscale('log')
        plt.xscale('log')
        plt.ylim(bottom=10**-5)  
        plt.legend()
        plt.xlabel("Energy (MeV)", fontsize = '12')
        plt.ylabel('Fluence (' + str(pred_units) + ')', fontsize = '12')
        # plt.show()
        name ='fluence_spectrum_' + current_event_df.Model + '_' + f.replace(' ', '_') + '_' + str(date).replace(' ', '_').replace(':', '-') + '_' + energy_key + '_' + validation_type + '.pdf'
        savename = os.path.join(cfg.outpath, 'plots', name)
        if os.path.isfile(savename): # this already exists for some reason, may be from an earlier run or potentially the model produced multiple predictions for the same event
            current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            name ='fluence_spectrum_' + current_event_df.Model + '_' + f.replace(' ', '_') + '_' + str(date).replace(' ', '_').replace(':', '-') + '_' + energy_key  + '_' + validation_type + '_' + current_time + '.pdf'
            savename = os.path.join(cfg.outpath, 'plots', name)
        plt.savefig(savename)
        plt.close()



        if f == 'Band Function':

            mapping = band_selections_mapping()

            for label, param in mapping:
                if 'Reduced' in label:
                    fit_dict['Observed ' + f + ' ' + label].append(float(obs_fit_output.redchi))
                    fit_dict['Model ' + f + ' ' + label].append(float(pred_fit_output.redchi))
                elif 'ChiSquare' in label:
                    fit_dict['Observed ' + f + ' ' + label].append(float(obs_fit_output.chisqr))
                    fit_dict['Model ' + f + ' ' + label].append(float(pred_fit_output.chisqr))
                else:
                    current_obs_param = str(obs_fit_output.params.create_uvars()[param]).rsplit('+/-')
                    
                    if '(' in current_obs_param[0]:
                        current_obs_param = str(obs_fit_output.params.create_uvars()[param]).rsplit(')')
                        exponential = current_obs_param[1]
                        obs_param = float(current_obs_param[0].rsplit('+/-')[0].replace('(', '') + exponential) 
                        obs_uncertainty = float(current_obs_param[0].rsplit('+/-')[1].replace(')', '') + exponential)
                    else:
                        obs_param = float(current_obs_param[0])
                        obs_uncertainty = float(current_obs_param[1])
                    current_pred_param = str(pred_fit_output.params.create_uvars()[param]).rsplit('+/-')
                    if '(' in current_pred_param[0]:
                        current_pred_param = str(pred_fit_output.params.create_uvars()[param]).rsplit(')')
                        exponential = current_pred_param[1]
                        pred_param = float(current_pred_param[0].rsplit('+/-')[0].replace('(', '') + exponential) 
                        pred_uncertainty = float(current_pred_param[0].rsplit('+/-')[1].replace(')', '') + exponential)
                    else:
                        pred_param = float(current_pred_param[0])
                        pred_uncertainty = float(current_pred_param[1])    

                    fit_dict['Observed ' + f + ' ' + label].append(obs_param)
                    fit_dict['Observed ' + f + ' ' + label + ' Uncertainty'].append(obs_uncertainty)
                    fit_dict['Model ' + f + ' ' + label].append(pred_param)
                    fit_dict['Model ' + f + ' ' + label + ' Uncertainty'].append(pred_uncertainty)
            fit_dict[f + ' Fit Plot'].append(name)
            
        elif f == 'Ellison Ramaty':
            mapping = er_selections_mapping()

            for label, param in mapping:
                if 'Reduced' in label:
                    fit_dict['Observed ' + f + ' ' + label].append(float(obs_fit_output.redchi))
                    fit_dict['Model ' + f + ' ' + label].append(float(pred_fit_output.redchi))
                elif 'ChiSquare' in label:
                    fit_dict['Observed ' + f + ' ' + label].append(float(obs_fit_output.chisqr))
                    fit_dict['Model ' + f + ' ' + label].append(float(pred_fit_output.chisqr))
                else:
                    current_obs_param = str(obs_fit_output.params.create_uvars()[param]).rsplit('+/-')
                    print(current_obs_param)
                    if '(' in current_obs_param[0]:
                        current_obs_param = str(obs_fit_output.params.create_uvars()[param]).rsplit(')')
                        exponential = current_obs_param[1]
                        obs_param = float(current_obs_param[0].rsplit('+/-')[0].replace('(', '') + exponential) 
                        obs_uncertainty = float(current_obs_param[0].rsplit('+/-')[1].replace(')', '') + exponential)
                    else:
                        obs_param = float(current_obs_param[0])
                        obs_uncertainty = float(current_obs_param[1])
                    current_pred_param = str(pred_fit_output.params.create_uvars()[param]).rsplit('+/-')
                    if '(' in current_pred_param[0]:
                        current_pred_param = str(pred_fit_output.params.create_uvars()[param]).rsplit(')')
                        exponential = current_pred_param[1]
                        pred_param = float(current_pred_param[0].rsplit('+/-')[0].replace('(', '') + exponential) 
                        pred_uncertainty = float(current_pred_param[0].rsplit('+/-')[1].replace(')', '') + exponential)
                    else:
                        pred_param = float(current_pred_param[0])
                        pred_uncertainty = float(current_pred_param[1])    
          
                    fit_dict['Observed ' + f + ' ' + label].append(obs_param)
                    fit_dict['Observed ' + f + ' ' + label + ' Uncertainty'].append(obs_uncertainty)
                    fit_dict['Model ' + f + ' ' + label].append(pred_param)
                    fit_dict['Model ' + f + ' ' + label + ' Uncertainty'].append(pred_uncertainty)
            fit_dict[f + ' Fit Plot'].append(name)

        elif f == 'Powerlaw':   
            mapping = powerlaw_selections_mapping()

            for label, param in mapping:
                if 'Reduced' in label:
                    fit_dict['Observed ' + f + ' ' + label].append(float(obs_fit_output.redchi))
                    fit_dict['Model ' + f + ' ' + label].append(float(pred_fit_output.redchi))
                elif 'ChiSquare' in label:
                    fit_dict['Observed ' + f + ' ' + label].append(float(obs_fit_output.chisqr))
                    fit_dict['Model ' + f + ' ' + label].append(float(pred_fit_output.chisqr))
                else:
                    current_obs_param = str(obs_fit_output.params.create_uvars()[param]).rsplit('+/-')
                    print(current_obs_param)
                    if '(' in current_obs_param[0]:
                        current_obs_param = str(obs_fit_output.params.create_uvars()[param]).rsplit(')')
                        exponential = current_obs_param[1]
                        obs_param = float(current_obs_param[0].rsplit('+/-')[0].replace('(', '') + exponential) 
                        obs_uncertainty = float(current_obs_param[0].rsplit('+/-')[1].replace(')', '') + exponential)
                    else:
                        obs_param = float(current_obs_param[0])
                        obs_uncertainty = float(current_obs_param[1])
                    current_pred_param = str(pred_fit_output.params.create_uvars()[param]).rsplit('+/-')
                    if '(' in current_pred_param[0]:
                        current_pred_param = str(pred_fit_output.params.create_uvars()[param]).rsplit(')')
                        exponential = current_pred_param[1]
                        pred_param = float(current_pred_param[0].rsplit('+/-')[0].replace('(', '') + exponential) 
                        pred_uncertainty = float(current_pred_param[0].rsplit('+/-')[1].replace(')', '') + exponential)
                    else:
                        pred_param = float(current_pred_param[0])
                        pred_uncertainty = float(current_pred_param[1])    
          
                    fit_dict['Observed ' + f + ' ' + label].append(obs_param)
                    fit_dict['Observed ' + f + ' ' + label + ' Uncertainty'].append(obs_uncertainty)
                    fit_dict['Model ' + f + ' ' + label].append(pred_param)
                    fit_dict['Model ' + f + ' ' + label + ' Uncertainty'].append(pred_uncertainty)
            fit_dict[f + ' Fit Plot'].append(name)

    return 



def band_selections_mapping():
    mapping = [
        ('Low Energy Spectral Slope', 'gamma_a'),
        ('High Energy Spectral Slope','gamma_b'),
        ('Rollover Energy', 'E_0'),
        ('Normalization Factor', 'norm'),
        ('ChiSquared', 'chisqr'),
        ('Reduced ChiSquared', 'redchi')
    ]
    return mapping

def er_selections_mapping():
    mapping = [
        ('Spectral Slope', 'gamma_er'),
        ('Rollover Energy', 'E_r'),
        ('Normalization Factor', 'norm'),
        ('ChiSquared', 'chisqr'),
        ('Reduced ChiSquared', 'redchi')
    ]
    return mapping


def powerlaw_selections_mapping():
    mapping = [
        ('Spectral Slope', 'gamma'),
        ('Normalization Factor', 'norm'),
        ('ChiSquared', 'chisqr'),
        ('Reduced ChiSquared', 'redchi')
    ]
    return mapping





def switch_fit_func(fitting_function, energy_bins, fluence_spectrum, date, fig_obj):
    func = {
        'Band Function': fit_band_function,
        'Ellison Ramaty': fit_ellisonramaty_function,
        'Powerlaw': fit_single_powerlaw
    }.get(fitting_function)

    fit_output = func(energy_bins, fluence_spectrum, date, fig_obj)
    return fit_output

def make_thresh_fname(thresh_key):
    """ Make threshold string for filenames.

    """
    thr = thresh_key.strip().split(".units") #threshold.10.0.units.1 / (cm2 s sr)
    thr = thr[0] #threshold.10.0
    thr = thr.strip().split("threshold.")
    thresh_fnm = "threshold_" + thr[1]
    return thresh_fnm

def fluence_spectrum_validation_selections(sub_sphinx_df, model, energy_key, thresh_key, validation_type):
    """
    Purpose is to generate the information that will be contained within the selections
    files, similar to other metrics/quantities. The biggest difference is that this function
    will also do the initial fitting of the spectra, then put the fit parameters in the selections
    files, as well as plot the fit. 

    """
    # pass in only what is needed from validation.py - an empty dataframe should never reach this point
    mismatch = bool(sub_sphinx_df.iloc[0]['Mismatch Allowed'])
    pred_energy_key = str(sub_sphinx_df.iloc[0]['Prediction Energy Channel Key'])
    pred_thresh_key = str(sub_sphinx_df.iloc[0]['Prediction Threshold Key'])
    # thresh_fnm = make_thresh_fname(thresh_key)
    event_list = sub_sphinx_df['Observed SEP Threshold Crossing Time'].to_list()
    fit_dict = metrics_dicts.initialize_fluence_spectrum_selections_dict()
    for event in event_list:
        current_event_df = sub_sphinx_df.loc[(sub_sphinx_df['Observed SEP Threshold Crossing Time'] == event)]     
        print(current_event_df)
        for valid_events in current_event_df.itertuples(index = False):
            fitting_procedure(fit_dict, valid_events, energy_key, event, thresh_key, validation_type)
    return fit_dict
    
        
            
def fluence_spectrum_validation_metrics_workflow(fluence_spec_metrics, selections_df, model, energy_key, thresh_key, validation_type):
    """
    Fitting has been completed and its time to generate some metrics:
    Fitting metrics: plots, correlation coefficients of fit parameters

    Error metrics: mean/median log error for each energy channel. Plots 
    of histograms of errors in each energy bin 

    """
    
    fluence_spec_metrics['Model'].append(model)
    fluence_spec_metrics['Energy Channel'].append(energy_key)
    fluence_spec_metrics['Threshold'].append(thresh_key)
    fit_parameter_dictionary = {
        'Band Function': ['Low Energy Spectral Slope', 'High Energy Spectral Slope', 'Rollover Energy'],
        'Ellison Ramaty': ['Spectral Slope', 'Rollover Energy'],
        'Powerlaw': ['Spectral Slope']
    }
    fit_types = ['Band Function', 'Ellison Ramaty', 'Powerlaw']
    thresh_fnm = make_thresh_fname(thresh_key)
    for forms in fit_types:
        
        
        fit_params = fit_parameter_dictionary[forms]
        #Correlation Plot
        for params in fit_params:
            selection_string = forms + ' ' + params

            obs = selections_df['Observed ' + selection_string].to_list()
            pred = selections_df['Model ' + selection_string].to_list()
            model_goodness_of_fit = selections_df['Model ' + forms + ' Reduced ChiSquared'].to_list()
            obs_goodness_of_fit = selections_df['Observed ' + forms + ' Reduced ChiSquared'].to_list()

            # overfit
            print(len(obs), len(pred), len(model_goodness_of_fit), len(obs_goodness_of_fit))
            for i in range(len(model_goodness_of_fit)):
            
                # overfit
                if model_goodness_of_fit[i] < 0.50 or obs_goodness_of_fit[i] < 0.50:
                    obs[i] = None
                    pred[i] = None
            # bad fit
                elif model_goodness_of_fit[i] > 20.0 or obs_goodness_of_fit[i] > 20.0:
                    obs[i] = None
                    pred[i] = None
            obs = [x for x in obs if x != None]
            pred = [x for x in pred if x != None]
            if len(obs) <= 2 and len(pred) <= 2:
                # not enough good fits to properly plot - move on without plotting
                fluence_spec_metrics[selection_string + ' Correlation Plot'].append(None)

                error = selections_df['Observed ' + selection_string] - selections_df['Model ' + selection_string]
                fluence_spec_metrics['Mean ' + selection_string + ' Error'].append(np.mean(error))
                fluence_spec_metrics['Median ' + selection_string + ' Error'].append(np.nanmedian(error))
                continue 
            if 'Rollover' in params:
                use_log = True
            else:
                use_log = False
            print(selection_string, len(obs), len(pred), obs, pred)
            try:
                corr_plot = plt_tools.correlation_plot(obs,pred,
                "Fluence Spectrum Correlation of " + selection_string, xlabel="Observations",
                ylabel="Model", use_log = use_log)

                figname = cfg.outpath + '/plots/Correlation_fluence_spectrum_' + selection_string.replace(' ', '_') + model + "_" \
                        + energy_key.strip() + "_" + thresh_fnm
                if cfg.do_mismatch:
                    figname = figname + "_mm"
                if validation_type != "" and validation_type != "All":
                    figname = figname + "_" + validation_type
                figname += ".pdf"
                corr_plot.savefig(figname, dpi=300, bbox_inches='tight')
                corr_plot.close()

                fluence_spec_metrics[selection_string + ' Correlation Plot'].append(figname)
            except:
                fluence_spec_metrics[selection_string + ' Correlation Plot'].append(None)
            error = selections_df['Observed ' + selection_string] - selections_df['Model ' + selection_string]
            fluence_spec_metrics['Mean ' + selection_string + ' Error'].append(np.mean(error))
            fluence_spec_metrics['Median ' + selection_string + ' Error'].append(np.nanmedian(error))

    #histograms
    obs_specs = selections_df['Observed SEP Fluence Spectrum'].to_list()
    pred_specs = selections_df['Predicted SEP Fluence Spectrum'].to_list()
    histo_name, histo_dict = fluence_histograms(obs_specs, pred_specs, model, energy_key, thresh_fnm, validation_type)
    fluence_spec_metrics['Log Errors Dictionary'].append(histo_dict)
    fluence_spec_metrics['Histogram of Log Errors Plot'].append(histo_name)

    return


# Specialized function to extract first/last/mean/max 





# Making this function for my own testing - will be deleted before merging to main
# Needed a way to test the fitting without solely requiring SPHINX - I want to test
# other datasets/instruments that we may not have models for to ensure robustness of
# the methodology
def temp_entry_function():
    import json
    fit_dict = metrics_dicts.initialize_fluence_spectrum_selections_dict()
    keys = ['Model','Energy Channel Key', 'Threshold Key',
            'Mismatch Allowed',
            'Prediction Energy Channel Key', 'Prediction Threshold Key',
            'Forecast Source',
            'Prediction Window Start', 'Prediction Window End',
            'Observed SEP Threshold Crossing Time',
            'Observed SEP End Time',
            'Observed SEP Fluence Spectrum',
            'Observed SEP Fluence Spectrum Units',
            'Predicted SEP Fluence Spectrum',
            'Predicted SEP Fluence Spectrum Units',
            'Fluence Spectrum Match Status']
     
    json_file = os.path.join('./junk', 'GOES-08_integral_enhance_idsep.1999-05-26T104000Z.json')
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    energy= data["sep_observation_submission"]['observations'][0]['energy_channel']
    energy_key = 'min.10.0.max.-1.0.units.MeV'
    thresh_key = 'threshold.10.0.units.1 / (cm2 s sr)'
    # print(data["sep_observation_submission"]['observations'][0]['fluence_spectra'])
    test_event = {key: [] for key in keys}
    test_event['Model'].append('Test')
    test_event['Energy Channel Key'].append(energy_key)
    test_event['Threshold Key'].append(thresh_key)
    test_event['Mismatch Allowed'].append(False)
    test_event['Prediction Energy Channel Key'].append(energy_key)
    test_event['Prediction Threshold Key'].append(thresh_key)
    test_event['Forecast Source'].append('Bleh')
    test_event['Prediction Window Start'].append(data["sep_observation_submission"]['observations'][0]['observation_window']['start_time'])
    test_event['Prediction Window End'].append(data["sep_observation_submission"]['observations'][0]['observation_window']['end_time'])
    test_event['Observed SEP Threshold Crossing Time'].append(data["sep_observation_submission"]['observations'][0]['event_lengths'][0]['start_time'])
    test_event['Observed SEP End Time'].append(data["sep_observation_submission"]['observations'][0]['event_lengths'][0]['end_time'])
    test_event['Observed SEP Fluence Spectrum'].append(data["sep_observation_submission"]['observations'][0]['fluence_spectra'][0]['fluence_spectrum'])
    test_event['Observed SEP Fluence Spectrum Units'].append(data["sep_observation_submission"]['observations'][0]['fluence_spectra'][0]['fluence_units'])
    test_event['Predicted SEP Fluence Spectrum'].append(data["sep_observation_submission"]['observations'][0]['fluence_spectra'][0]['fluence_spectrum'])
    test_event['Predicted SEP Fluence Spectrum Units'].append(data["sep_observation_submission"]['observations'][0]['fluence_spectra'][0]['fluence_units'])
    test_event['Fluence Spectrum Match Status'].append('SEP Event')

    current_event = pd.DataFrame(test_event)
    event = data["sep_observation_submission"]['observations'][0]['event_lengths'][0]['start_time']
    print(current_event)
    for valid_events in current_event.itertuples(index = False):
        fitting_procedure(fit_dict, valid_events, energy_key, event, thresh_key, 'TESTING')
    return




if __name__ == '__main__':
    temp_entry_function()

