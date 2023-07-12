#Subroutines related to validation
from . import object_handler as objh
from . import metrics
import sys
import pandas as pd


__version__ = "0.1"
__author__ = "Katie Whitman"
__maintainer__ = "Katie Whitman"
__email__ = "kathryn.whitman@nasa.gov"

""" utils/validation.py contains subroutines to validate forecasts after
    they have been matched to observations.
    
"""
    

def initialize_dict(model_names, all_energy_channels, all_obs_thresholds):
    """ Set up a dictionary for each model, each possible quantity,
        each observed energy channel, and fields to hold predicted and
        observed values.
        
        Not all models will forecast all values. Simply placeholders.
        
    """
    quantity_keys = ["All Clear", "Probability",
        "Peak Intensity (Onset Peak)",
        "Peak Intensity (Onset Peak) Time", "Peak Intensity Max (Max Flux)",
        "Peak Intensity Max (Max Flux) Time", "Threshold Crossing Time",
        "Start Time", "End Time", "Fluence", "Fluence Spectrum",
        "Time Profile"]

    quantity_dict = {}
    
    no_threshold = ["Peak Intensity (Onset Peak)",
        "Peak Intensity (Onset Peak) Time", "Peak Intensity Max (Max Flux)",
        "Peak Intensity Max (Max Flux) Time"]

    for model in model_names:
        quantity_dict.update({model:{}})

        for quantity in quantity_keys:
            quantity_dict[model].update({quantity: {}})
        
            for channel in all_energy_channels:
                energy_key = objh.energy_channel_to_key(channel)

                if quantity in no_threshold:
                    quantity_dict[model][quantity].update({energy_key:
                        {"Forecast Issue Time":[],
                        "Observed SEP Event Date":[],
                        "Observed":[], "Predicted":[],
                        "Matching Status": []}})
                else:
                    quantity_dict[model][quantity].update({energy_key: {}})
                    for threshold in all_obs_thresholds:
                        thresh_key = objh.threshold_to_key(threshold)
                        quantity_dict[model][quantity][energy_key].update({thresh_key:
                            {"Forecast Issue Time":[],
                            "Observed SEP Event Date":[],
                            "Observed":[], "Predicted":[],
                            "Matching Status": []}})
 
    return quantity_dict


def fill_all_clear(sphinx, model, quantity_dict):
    """ Pull out the predicted and observed values for a single sphinx
        object and add to quantity_dict under the right model, energy
        channel, and threshold.
        
    """
    qk = "All Clear"

    energy_channel = sphinx.energy_channel
    ek = objh.energy_channel_to_key(energy_channel)
    
    #Indicate whether a prediction was matched to an observation, discarded
    #because of the timing or the inputs or after the observed phenomena,
    #prediction reports ongoing event
    #None - no status determined
    #Matched - prediction was matched to an observation
    #Discarded - prediction made after event started
    #Ongoing - prediction reports that there is an ongoing event and isn't
    #   really a forecast
    match_status = "None"
    
    #Check if forecast for all clear
    pred_all_clear = sphinx.prediction.all_clear.all_clear_boolean
    if pred_all_clear == None:
        continue
        
    obs_all_clear = sphinx.observed_all_clear.all_clear_boolean
    if obs_all_clear == None:
        continue

    pred_threshold = {'threshold': sphinx.prediction.all_clear.threshold,
        'threshold_units': sphinx.prediction.all_clear.threshold_units}
    obs_threshold = {'threshold': sphinx.observed_all_clear.threshold,
        'threshold_units': sphinx.observed_all_clear.threshold_units}
        
    #Thresholds must match
    if pred_threshold != obs_threshold:
        continue
    
    tk = objh.threshold_to_key(obs_threshold)
    quantity_dict[model][qk][ek][tk]['Forecast Issue Time'].append(sphinx.prediction.issue_time)
    quantity_dict[model][qk][ek][tk]['Observed SEP Event Date'].append(sphinx.observed_threshold_crossing[tk].crossing_time)
    quantity_dict[model][qk][ek][tk]['Observed'].append(obs_all_clear)
    quantity_dict[model][qk][ek][tk]['Predicted'].append(pred_all_clear)

    return quantity_dict


def fill_probability(sphinx, model, quantity_dict):
    """ Pull out the predicted and observed values for a single sphinx
        object and add to quantity_dict under the right model, energy
        channel, and threshold.
        
    """
    qk = "Probability"

    energy_channel = sphinx.energy_channel
    ek = objh.energy_channel_to_key(energy_channel)
    
    #Check if a forecast exists for probability
    if sphinx.prediction.probability == []:
        continue

    #Check each forecast for probability
    for prob_obj in sphinx.prediction.probability:
        pred_prob = prob_obj.probability_value
        if pred_prob == None:
            continue

        pred_thresh = {'threshold': prob_obj.threshold,
            'threshold_units': prob_obj.threshold_units}
        #Check that predicted threshold was applied in the observations
        if pred_thresh not in sphinx.thresholds:
            continue

        tk = objh.threshold_to_key(pred_thresh)
        
        #Extact matching observed value for threshold
        obs_prob = sphinx.observed_probability[tk].probability_value
        if obs_prob == None:
            continue

        quantity_dict[model][qk][ek][tk]['Forecast Issue Time'].append(sphinx.prediction.issue_time)
        quantity_dict[model][qk][ek][tk]['Observed SEP Event Date'].append(sphinx.observed_threshold_crossing[tk].crossing_time)
        quantity_dict[model][qk][ek][tk]['Observed'].append(obs_prob)
        quantity_dict[model][qk][ek][tk]['Predicted'].append(pred_prob)

    return quantity_dict



def fill_threshold_crossing_time(sphinx, model, quantity_dict):
    """ Pull out the predicted and observed values for a single sphinx
        object and add to quantity_dict under the right model, energy
        channel, and threshold.
        
    """
    qk = "Threshold Crossing Time"

    energy_channel = sphinx.energy_channel
    ek = objh.energy_channel_to_key(energy_channel)
    
    #Check if a forecast exists for probability
    if sphinx.prediction.threshold_crossings == []:
        continue

    #Check each forecast for probability
    for obj in sphinx.prediction.threshold_crossings:
        predicted = obj.crossing_time
        if predicted == None or pd.isnull(predicted):
            continue

        pred_thresh = {'threshold': obj.threshold,
            'threshold_units': obj.threshold_units}
        #Check that predicted threshold was applied in the observations
        if pred_thresh not in sphinx.thresholds:
            continue

        tk = objh.threshold_to_key(pred_thresh)
        
        #Extact matching observed value for threshold
        observed = sphinx.observed_threshold_crossing[tk].crossing_time
        if observed == None or pd.isnull(observed):
            continue

        quantity_dict[model][qk][ek][tk]['Forecast Issue Time'].append(sphinx.prediction.issue_time)
        quantity_dict[model][qk][ek][tk]['Observed SEP Event Date'].append(sphinx.observed_threshold_crossing[tk].crossing_time)
        quantity_dict[model][qk][ek][tk]['Observed'].append(observed)
        quantity_dict[model][qk][ek][tk]['Predicted'].append(predicted)

    return quantity_dict



def fill_start_time(sphinx, model, quantity_dict):
    """ Pull out the predicted and observed values for a single sphinx
        object and add to quantity_dict under the right model, energy
        channel, and threshold.
        
    """
    qk = "Start Time"

    energy_channel = sphinx.energy_channel
    ek = objh.energy_channel_to_key(energy_channel)
    
    #Check if a forecast exists for probability
    if sphinx.prediction.event_lengths == []:
        continue

    #Check each forecast for probability
    for obj in sphinx.prediction.event_lengths:
        predicted = obj.start_time
        if predicted == None or pd.isnull(predicted):
            continue

        pred_thresh = {'threshold': obj.threshold,
            'threshold_units': obj.threshold_units}
        #Check that predicted threshold was applied in the observations
        if pred_thresh not in sphinx.thresholds:
            continue

        tk = objh.threshold_to_key(pred_thresh)
        
        #Extact matching observed value for threshold
        observed = sphinx.observed_start_time[tk]
        if observed == None or pd.isnull(observed):
            continue

        quantity_dict[model][qk][ek][tk]['Forecast Issue Time'].append(sphinx.prediction.issue_time)
        quantity_dict[model][qk][ek][tk]['Observed SEP Event Date'].append(sphinx.observed_threshold_crossing[tk].crossing_time)
        quantity_dict[model][qk][ek][tk]['Observed'].append(observed)
        quantity_dict[model][qk][ek][tk]['Predicted'].append(predicted)

    return quantity_dict


def fill_end_time(sphinx, model, quantity_dict):
    """ Pull out the predicted and observed values for a single sphinx
        object and add to quantity_dict under the right model, energy
        channel, and threshold.
        
    """
    qk = "End Time"

    energy_channel = sphinx.energy_channel
    ek = objh.energy_channel_to_key(energy_channel)
    
    #Check if a forecast exists for probability
    if sphinx.prediction.event_lengths == []:
        continue

    #Check each forecast for probability
    for obj in sphinx.prediction.event_lengths:
        predicted = obj.end_time
        if predicted == None or pd.isnull(predicted):
            continue

        pred_thresh = {'threshold': obj.threshold,
            'threshold_units': obj.threshold_units}
        #Check that predicted threshold was applied in the observations
        if pred_thresh not in sphinx.thresholds:
            continue

        tk = objh.threshold_to_key(pred_thresh)
        
        #Extact matching observed value for threshold
        observed = sphinx.observed_end_time[tk]
        if observed == None or pd.isnull(observed):
            continue

        quantity_dict[model][qk][ek][tk]['Forecast Issue Time'].append(sphinx.prediction.issue_time)
        quantity_dict[model][qk][ek][tk]['Observed SEP Event Date'].append(sphinx.observed_threshold_crossing[tk].crossing_time)
        quantity_dict[model][qk][ek][tk]['Observed'].append(observed)
        quantity_dict[model][qk][ek][tk]['Predicted'].append(predicted)

    return quantity_dict


def fill_fluence(sphinx, model, quantity_dict):
    """ Pull out the predicted and observed values for a single sphinx
        object and add to quantity_dict under the right model, energy
        channel, and threshold.
        
    """
    qk = "Fluence"

    energy_channel = sphinx.energy_channel
    ek = objh.energy_channel_to_key(energy_channel)
    
    #Check if a forecast exists for probability
    if sphinx.prediction.fluences == []:
        continue

    #Check each forecast for probability
    for obj in sphinx.prediction.fluences:
        predicted = obj.fluence
        if predicted == None or pd.isnull(predicted):
            continue

        pred_thresh = {'threshold': obj.threshold,
            'threshold_units': obj.threshold_units}
        #Check that predicted threshold was applied in the observations
        if pred_thresh not in sphinx.thresholds:
            continue

        tk = objh.threshold_to_key(pred_thresh)
        
        #Extact matching observed value for threshold
        observed = sphinx.observed_fluence[tk].fluence
        if observed == None or pd.isnull(observed):
            continue

        quantity_dict[model][qk][ek][tk]['Forecast Issue Time'].append(sphinx.prediction.issue_time)
        quantity_dict[model][qk][ek][tk]['Observed SEP Event Date'].append(sphinx.observed_threshold_crossing[tk].crossing_time)
        quantity_dict[model][qk][ek][tk]['Observed'].append(observed)
        quantity_dict[model][qk][ek][tk]['Predicted'].append(predicted)

    return quantity_dict


def fill_fluence_spectrum(sphinx, model, quantity_dict):
    """ Pull out the predicted and observed values for a single sphinx
        object and add to quantity_dict under the right model, energy
        channel, and threshold.
        
    """
    qk = "Fluence Spectrum"

    energy_channel = sphinx.energy_channel
    ek = objh.energy_channel_to_key(energy_channel)
    
    #Check if a forecast exists for probability
    if sphinx.prediction.fluences == []:
        continue

    #Check each forecast for probability
    for obj in sphinx.prediction.fluence_spectra:
        predicted = obj.fluence_spectrum
        if predicted == None or pd.isnull(predicted):
            continue

        pred_thresh = {'threshold': obj.threshold_start,
            'threshold_units': obj.threshold_units}
        #Check that predicted threshold was applied in the observations
        if pred_thresh not in sphinx.thresholds:
            continue

        tk = objh.threshold_to_key(pred_thresh)
        
        #Extact matching observed value for threshold
        observed = sphinx.observed_fluence_spectrum[tk].fluence_spectrum
        if observed == None or pd.isnull(observed):
            continue

        quantity_dict[model][qk][ek][tk]['Forecast Issue Time'].append(sphinx.prediction.issue_time)
        quantity_dict[model][qk][ek][tk]['Observed SEP Event Date'].append(sphinx.observed_threshold_crossing[tk].crossing_time)
        quantity_dict[model][qk][ek][tk]['Observed'].append(observed)
        quantity_dict[model][qk][ek][tk]['Predicted'].append(predicted)

    return quantity_dict



def fill_dict(matched_sphinx, model_names, all_energy_channels,
    all_obs_thresholds):
    """ Fill in a dictionary with the all clear predictions and observations
        organized by model and energy channel.
    """
    #sorted by model, quantity, energy channel, threshold
    quantity_dict = initialize_dict(model_names, all_energy_channels,
                all_obs_thresholds)

    #Loop through the forecasts for each model and fill in quantity_dict
    #as appropriate
    for model in model_names:
        for sphinx in matched_sphinx[model]:
            quantity_dict = fill_all_clear(sphinx, model, quantity_dict)
            quantity_dict = fill_probability(sphinx, model, quantity_dict)
            quantity_dict = fill_threshold_crossing_time(sphinx, model,
                quantity_dict)
            quantity_dict = fill_start_time(sphinx, model, quantity_dict)
            quantity_dict = fill_end_time(sphinx, model, quantity_dict)
            quantity_dict = fill_fluence(sphinx, model, quantity_dict)
            quantity_dict = fill_fluence_spectrum(sphinx, model, quantity_dict)



def intuitive_validation(matched_sphinx, model_names, all_energy_channels,
    all_observed_threhsolds, observed_sep_events):
    """ In the intuitive_validation subroutine, forecasts are validated in a
        way similar to which people would interpret forecasts.
    
        Forecasts are assessed (or useful to end users) up until the observed
        phenomenon happens. For example, only forecasts of peak flux are
        useful up until the observed peak happens. After that, a human would
        mentally filter out any additional forecasts for peak coming in from
        a model. Or, if the model's prediction window is large enough,
        continued peak flux forecasts could/would be interpreted for the
        next possible SEP event.
        
        In match.py, observed values have been matched to predicted values
        only if the last trigger or input time for the prediction was before
        the observed phenomenon.
        
        If a forecast was issued after the observed phenomenon, that forecast
        is ignored or, if the prediction window is large and extends past the
        current SEP event, is considered as a forecast for a next SEP event.
        
        This subroutine compared the predicted values to the matched
        observed values
        
        
    Input:
    
        :matched_sphinx: (SPHINX object) contains a Forecast object,
            Observation objects that are inside the forecast prediction
            window, and the observed values that are appropriately matched up
            to the forecast given the timing of the triggers/inputs and
            observed phenomena
        :model_names: (str array) array of the models whose predictions were
            read into the code
        :all_observed_thresholds: (dict) dictionary organized by energy
            channel and thresholds that were applied to observations (only
            predictions corresponding to thresholds that were applied to the
            observations can be validated)
        :observed_sep_events: (dict) dictionary organized by model name,
            energy channel, and threshold containing all unique observed SEP
            events that fell inside a forecast prediction window
    
    Output:
    
    
    
    """

    #For each model and predicted quantity, create arrays of paired up values
    #so can calculate metrics
    all_clear_dict = {}
    probability_dict = {}
    peak_intensity_dict = {}
    peak_intensity_time_dict = {}
    peak_intensity_max_dict = {}
    peak_intensity_max_time_dict = {}
    thresh_cross_time_dict = {}
    start_time_dict = {}
    end_time_dict = {}
    fluence_dict = {}
    fluence_spectrum_dict = {}
    profile_dict = {}
    
    
