import sys
import datetime

__version__ = "0.1"
__author__ = "Katie Whitman"
__maintainer__ = "Katie Whitman"
__email__ = "kathryn.whitman@nasa.gov"

""" utils/object_handler.py contains subroutines to interact
    with class objects defined in utils/classes.py.
    
"""

def build_model_list(all_model):
    """ Identify all of the models represented in the list from the entry in
        ['sep_forecast_submission']['model']['short_name']
        
        INPUTS:
        
        :all_model: dict of array of Forecast objects order by
            energy channel
            
        OUTPUTS:
        
        :model_names: (string 1xn array) list of SEP model names
            
    """
    model_names = []
    keys = all_model.keys()
    for key in keys:
        models = all_model[key] #array
        for fcast in models:
            name = fcast.short_name
            if name == "" or name == None: continue
            if name not in model_names:
                model_names.append(name)

    print("build_model_list: All models identified are: ")
    print(model_names)
    
    return model_names


def identify_all_thresholds(all_obj):
    """ Find all the thresholds applied to a given energy channel.
        Thresholds are applied in:
        All clear
        Event lengths
        Fluence spectra
        Threshold crossings
        Probabilities
    
    Inputs:
    
        :all_obj: (array of Forecast or Observation objects)
        
    Outputs:
    
        :all_thresholds: (array of dict)
            [{'threshold': 10, 'threshold_units': Unit('pfu')}]
    
    """
    all_thresholds = []
    
    for obj in all_obj:
        thresh = obj.all_clear.threshold
        units = obj.all_clear.threshold_units
        if thresh != None and units != None:
            dict = {'threshold':thresh, 'threshold_units': units}
            if dict not in all_thresholds:
                all_thresholds.append(dict)
        
        if obj.event_lengths != []:
            for entry in obj.event_lengths:
                thresh = entry.threshold
                units = entry.threshold_units
                if thresh != None and units != None:
                    dict = {'threshold':thresh, 'threshold_units': units}
                    if dict not in all_thresholds:
                        all_thresholds.append(dict)
        
        if obj.fluence_spectra != []:
            for entry in obj.fluence_spectra:
                thresh = entry.threshold_start
                units = entry.threshold_units
                if thresh != None and units != None:
                    dict = {'threshold':thresh, 'threshold_units': units}
                    if dict not in all_thresholds:
                        all_thresholds.append(dict)
       
        if obj.threshold_crossings != []:
            for entry in obj.threshold_crossings:
                thresh = entry.threshold
                units = entry.threshold_units
                if thresh != None and units != None:
                    dict = {'threshold':thresh, 'threshold_units': units}
                    if dict not in all_thresholds:
                        all_thresholds.append(dict)


        if obj.probabilities != []:
            for entry in obj.probabilities:
                thresh = entry.threshold
                units = entry.threshold_units
                if thresh != None and units != None:
                    dict = {'threshold':thresh, 'threshold_units': units}
                    if dict not in all_thresholds:
                        all_thresholds.append(dict)

    return all_thresholds


def print_observed_values(obj):
    print()
    print("--- " + obj.source + " -----")
    print(obj.energy_channel)
    print(obj.short_name)
    print(obj.issue_time)
    print(obj.location)
    print(obj.species)
    print(obj.observation_window_start)
    print(obj.observation_window_end)
    print(obj.sep_profile)
    print(vars(obj.all_clear))
    print(vars(obj.peak_intensity))
    print(vars(obj.peak_intensity_max))
    if obj.event_lengths != []:
        print(vars(obj.event_lengths[0]))
    if obj.fluences != []:
        print(vars(obj.fluences[0]))
    if obj.fluence_spectra != []:
        print(vars(obj.fluence_spectra[0]))
    if obj.threshold_crossings != []:
        print(vars(obj.threshold_crossings[0]))
    if obj.probabilities != []:
        print(vars(obj.probabilities[0]))


def print_forecast_values(obj):
    print()
    print("--- " + obj.source + " -----")
    print(obj.energy_channel)
    print(obj.short_name)
    print(obj.issue_time)
    print(obj.location)
    print(obj.species)
    print(obj.prediction_window_start)
    print(obj.prediction_window_end)
    print(obj.sep_profile)
    
    if obj.cmes != []:
        print(vars(obj.cmes[0]))
    if obj.cme_simulations != []:
        print(vars(obj.cme_simulations[0]))
    if obj.flares != []:
        print(vars(obj.flares[0]))
    if obj.particle_intensities != []:
        print(vars(obj.particle_intensities[0]))
 
    if obj.magnetic_connectivity != []:
        print(vars(obj.magnetic_connectivity[0]))
    if obj.magnetograms != []:
        print(vars(obj.magnetograms[0]))
    if obj.coronagraphs != []:
        print(vars(obj.coronagraphs[0]))
 
    print(vars(obj.all_clear))
    print(vars(obj.peak_intensity))
    print(vars(obj.peak_intensity_max))
    if obj.event_lengths != []:
        print(vars(obj.event_lengths[0]))
    if obj.fluences != []:
        print(vars(obj.fluences[0]))
    if obj.fluence_spectra != []:
        print(vars(obj.fluence_spectra[0]))
    if obj.threshold_crossings != []:
        print(vars(obj.threshold_crossings[0]))
    if obj.probabilities != []:
        print(vars(obj.probabilities[0]))


def last_trigger_time(obj):
    """ Out of all the triggers, find the last data time
        relevant for matching to observations.
        
        Matching is guided by the idea that that a forecast
        is only valid for time periods after the triggers.
        e.g. if there are multiple CMEs in a simulation,
        the forecast is only relevant for what is observed
        after the eruption of the last CME because observational
        truth has been available to the forecaster/mode up until
        that time.
        
    """
    last_time = None
    
    #Find the time of the latest CME in the trigger list
    last_cme_time = None
    if obj.cmes != []:
        for cme in obj.cmes:
            #start time and liftoff time could be essentially
            #the same time and both are indicators of when
            #the CME first erupted. Take the earliest of the
            #two times for matching.
            check_time = None
            start_time = cme.start_time
            liftoff_time = cme.liftoff_time
            
            if start_time == None and liftoff_time == None:
                continue
            
            if isinstance(start_time,datetime.date):
                check_time = start_time
                
            if isinstance(liftoff_time,datetime.date):
                check_time = liftoff_time
            
            if isinstance(start_time,datetime.date) and\
                isinstance(liftoff_time,datetime.date):
                check_time = min(start_time,liftoff_time)
            
            if last_cme_time == None:
                last_cme_time = check_time
            elif isinstance(check_time,datetime.date):
                last_cme_time = max(last_cme_time,check_time)
 

    #Find the time of the latest flare in the trigger list
    last_flare_time = None
    if obj.flares != []:
        for flare in flares:
            #The flare peak time is the most relevant for matching
            #to SEP events as the CME (if any) is often launched
            #around the time of the peak.
            check_time = None
            start_time = flare.start_time
            peak_time = flare.peak_time
            end_time = flare.end_time
            last_data_time = flare.last_data_time
            
            if isinstance(peak_time,datetime.date):
                check_time = peak_time
            elif isinstance(start_time,datetime.date):
                check_time = start_time
            elif isinstance(end_time,datetime.date):
                check_time = end_time
            elif isinstance(last_data_time,datetime.date):
                check_time = last_data_time
                
            if last_flare_time == None:
                last_flare_time = check_time
            elif insinstance(check_time, datetime.date):
                last_flare_time = max(last_flare_time,check_time)

    #Find the latest particle intensity data used by the model
    last_pi_time = None
    if obj.particle_intensities != []:
        for pi in obj.particle_intensities:
            check_time = pi.last_data_time
            if isinstance(check_time,datetime.date):
                if last_pi_time == None:
                    last_pi_time = check_time
                else:
                    last_pi_time = max(last_pi_time,check_time)


    #Take the latest of all the times
    if isinstance(last_cme_time,datetime.date):
        last_time = last_cme_time
        
    if isinstance(last_flare_time,datetime.date):
        if last_time == None:
            last_time = last_flare_time
        else:
            last_time = max(last_time,last_flare_time)
            
    if isinstance(last_pi_time,datetime.date):
        if last_time == None:
            last_time = last_pi_time
        else:
            last_time = max(last_time,last_pi_time)
            
    return last_time


def last_input_time(obj):
    """ Out of all the inputs, find the last data time
        relevant for matching to observations.
        
        Matching is guided by the idea that that a forecast
        is only valid for time periods after the last input.

    """
    last_time = None
    
    #Find time of last magnetogram used as input
    last_magneto_time = None
    if obj.magnetograms != []:
        for magneto in obj.magnetograms:
            if magneto.products == []: continue
            for prod in magneto.products:
                last_data_time = prod['last_data_time']
                if isinstance(last_data_time,datetime.date):
                    if last_magneto_time == None:
                        last_magneto_time = last_data_time
                    else:
                        last_magneto_time = max(last_magneto_time,last_data_time)
                
    #Find time of last coronagraph used as input
    last_corona_time = None
    if obj.coronagraphs != []:
        for corona in obj.coronagraphs:
            if corona.products == []: continue
            for prod in corona.products:
                last_data_time = prod['last_data_time']
                if isinstance(last_data_time,datetime.date):
                    if last_corona_time == None:
                        last_corona_time = last_data_time
                    else:
                        last_corona_time = max(last_corona_time,last_data_time)

    if isinstance(last_magneto_time,datetime.date):
        last_time = last_magneto_time
        
    if isinstance(last_corona_time,datetime.date):
        if last_time == None:
            last_time = last_corona_time
        else:
            last_time = max(last_time,last_corona_time)
    
    return last_time