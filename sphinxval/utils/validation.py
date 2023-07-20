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
    

def initialize_dict():
    """ Set up a pandas df to hold each possible quantity,
        each observed energy channel, and predicted and
        observed values.
        
    """
    #Convert to Pandas dataframe
    #Include triggers with as much flattened info
    #If need multiple dimension, then could be used as tooltip info
    #Last CME, N CMEs, Last speed, last location, Timestamps array of all CMEs used
    

    dict = {"Model": [],
            "Energy Channel Key": [],
            "Threshold Key": [],
            "Forecast Source": [],
            "Forecast Issue Time":[],
            "Prediction Window Start": [],
            "Prediction Window End": [],
            "Number of CMEs": [],
            "CME Start Time": [], #Timestamp of 1st
                #coronagraph image CME is visible in
            "CME Liftoff Time": [], #Timestamp of coronagraph
                #image with 1st indication of CME liftoff (used by
                #CACTUS)
            "CME Latitude": [],
            "CME Longitude": [],
            "CME Speed": [],
            "CME Half Width": [],
            "CME PA": [],
            "CME Catalog": [],
            "Number of Flares": [],
            "Flare Latitude": [],
            "Flare Longitude": [],
            "Flare Start Time": [],
            "Flare Peak Time": [],
            "Flare End Time": [],
            "Flare Last Data Time": [],
            "Flare Intensity": [],
            "Flare Integrated Intensity": [],
            "Flare NOAA AR": [],
            "Observatory": [],
            "Observed Time Profiles": [], #string of comma
                                          #separated filenames
            "Observed SEP All Clear": [],
            "Observed SEP Probability": [],
            "Observed SEP Threshold Crossing Time": [],
            "Observed SEP Start Time":[],
            "Observed SEP End Time": [],
            "Observed SEP Peak Intensity (Onset Peak)": [],
            "Observed SEP Peak Intensity (Onset Peak) Units": [],
            "Observed SEP Peak Intensity (Onset Peak) Time": [],
            "Observed SEP Peak Intensity Max (Max Flux)": [],
            "Observed SEP Peak Intensity Max (Max Flux) Units": [],
            "Observed SEP Peak Intensity Max (Max Flux) Time": [],
            "Observed SEP Fluence": [],
            "Observed SEP Fluence Units": [],
            "Observed SEP Fluence Spectrum": [],
            "Observed SEP Fluence Spectrum Units": [],
            "Predicted SEP All Clear": [],
            "All Clear Match Status": [],
            "Predicted SEP Probability": [],
            "Probability Match Status": [],
            "Predicted SEP Threshold Crossing Time": [],
            "Threshold Crossing Time Match Status": [],
            "Predicted SEP Start Time":[],
            "Start Time Match Status": [],
            "Predicted SEP End Time": [],
            "End Time Match Status": [],
            "Predicted SEP Peak Intensity (Onset Peak)": [],
            "Predicted SEP Peak Intensity (Onset Peak) Units": [],
            "Predicted SEP Peak Intensity (Onset Peak) Time": [],
            "Peak Intensity Match Status": [],
            "Predicted SEP Peak Intensity Max (Max Flux)": [],
            "Predicted SEP Peak Intensity Max (Max Flux) Units": [],
            "Predicted SEP Peak Intensity Max (Max Flux) Time": [],
            "Peak Intensity Max Match Status": [],
            "Predicted SEP Fluence": [],
            "Predicted SEP Fluence Units": [],
            "Fluence Match Status": [],
            "Predicted SEP Fluence Spectrum": [],
            "Predicted SEP Fluence Spectrum Units": [],
            "Fluence Spectrum Match Status": [],
            "Time Profile Match Status": []}

    return dict




def fill_dict_row(sphinx, dict, energy_key, thresh_key):
    """ Add a row to a dataframe with all of the supporting information
        for the forecast and observations that needs to be passed to
        SPHINX-Web.
        
    Input:
    
        :sphinx: (SPHINX object) contains all prediction and matched observation
            information
        :predicted: The predicted value for one specific type of quantity (e.g.
            peak_intensity, all_clear, start_time)
        :observed: The matched up observed value of the same quantity
        :df: (pandas DataFrame) contains all matched and observed values for
            a specific quantity along with supporting information
        
    Output:
    
        :updated_df: (pandas DataFrame) The dataframe is updated another
            another row
        
    """

    ncme = len(sphinx.prediction.cmes)
    if ncme > 0:
        cme_start = sphinx.prediction.cmes[-1].start_time
        cme_liftoff = sphinx.prediction.cmes[-1].liftoff_time
        cme_lat = sphinx.prediction.cmes[-1].lat
        cme_lon = sphinx.prediction.cmes[-1].lon
        cme_pa = sphinx.prediction.cmes[-1].pa
        cme_half_width = sphinx.prediction.cmes[-1].half_width
        cme_speed = sphinx.prediction.cmes[-1].speed
        cme_catalog = sphinx.prediction.cmes[-1].catalog
    else:
        cme_start = None
        cme_liftoff = None
        cme_lat = None
        cme_lon = None
        cme_pa = None
        cme_half_width = None
        cme_speed = None
        cme_catalog = None
        
    nfl = len(sphinx.prediction.flares)
    if nfl > 0:
        fl_lat = sphinx.prediction.flares[-1].lat
        fl_lon = sphinx.prediction.flares[-1].lon
        fl_last_data_time = sphinx.prediction.flares[-1].last_data_time
        fl_start_time = sphinx.prediction.flares[-1].start_time
        fl_peak_time = sphinx.prediction.flares[-1].peak_time
        fl_end_time = sphinx.prediction.flares[-1].end_time
        fl_intensity = sphinx.prediction.flares[-1].intensity
        fl_integrated_intensity = sphinx.prediction.flares[-1].integrated_intensity
        fl_AR = sphinx.prediction.flares[-1].noaa_region
    else:
        fl_lat = None
        fl_lon = None
        fl_last_data_time = None
        fl_start_time = None
        fl_peak_time = None
        fl_end_time = None
        fl_intensity = None
        fl_integrated_intensity = None
        fl_AR = None

    observatory = ""
    obs_time_prof = ""
    for i in range(len(sphinx.prediction_observation_windows_overlap)):
        if i == 0:
            observatory = sphinx.prediction_observation_windows_overlap[i].short_name
            obs_time_prof = sphinx.observed_sep_profiles[i]
        else:
            observatory += "," + sphinx.prediction_observation_windows_overlap[i].short_name
            obs_time_prof += "," + sphinx.observed_sep_profiles[i]
    

    ####PREDICTED VALUES
    pred_all_clear, ac_match_status = sphinx.return_predicted_all_clear()
    pred_prob, prob_match_status = sphinx.return_predicted_probability(thresh_key)
    pred_thresh_cross, tc_match_status =\
        sphinx.return_predicted_threshold_crossing_time(thresh_key)
    pred_start_time, st_match_status =\
        sphinx.return_predicted_start_time(thresh_key)
    pred_end_time, et_match_status =\
        sphinx.return_predicted_end_time(thresh_key)
    pred_fluence, pred_fl_units, fl_match_status =\
        sphinx.return_predicted_fluence(thresh_key)
    pred_fl_spec, pred_flsp_units, flsp_match_status =\
        sphinx.return_predicted_fluence_spectrum(thresh_key)
    pred_peak_intensity, pred_pi_units, pred_pi_time, pi_match_status =\
        sphinx.return_predicted_peak_intensity()
    pred_peak_intensity_max, pred_pimax_units, pred_pimax_time,\
        pimax_match_status = sphinx.return_predicted_peak_intensity_max()


    dict["Model"].append(sphinx.prediction.short_name)
    dict["Energy Channel Key"].append(energy_key)
    dict["Threshold Key"].append(thresh_key)
    dict["Forecast Source"].append(sphinx.prediction.source)
    dict["Forecast Issue Time"].append(sphinx.prediction.issue_time)
    dict["Prediction Window Start"].append(sphinx.prediction.prediction_window_start)
    dict["Prediction Window End"].append(sphinx.prediction.prediction_window_end)
    dict["Number of CMEs"].append(ncme)
    dict["CME Start Time"].append(cme_start) #Timestamp of 1st
            #coronagraph image CME is visible in
    dict["CME Liftoff Time"].append(cme_liftoff) #Timestamp of coronagraph
            #image with 1st indication of CME liftoff (used by
            #CACTUS)
    dict["CME Latitude"].append(cme_lat)
    dict["CME Longitude"].append(cme_lon)
    dict["CME Speed"].append(cme_speed)
    dict["CME Half Width"].append(cme_half_width)
    dict["CME PA"].append(cme_pa)
    dict["CME Catalog"].append(cme_catalog)
    dict["Number of Flares"].append(nfl)
    dict["Flare Latitude"].append(fl_lat)
    dict["Flare Longitude"].append(fl_lon)
    dict["Flare Start Time"].append(fl_start_time)
    dict["Flare Peak Time"].append(fl_peak_time)
    dict["Flare End Time"].append(fl_end_time)
    dict["Flare Last Data Time"].append(fl_last_data_time)
    dict["Flare Intensity"].append(fl_intensity)
    dict["Flare Integrated Intensity"].append(fl_integrated_intensity)
    dict["Flare NOAA AR"].append(fl_AR)
    dict["Observatory"].append(observatory)
    dict["Observed Time Profiles"].append(obs_time_prof) #string of comma
                              #separated filenames
    dict["Observed SEP All Clear"].append(sphinx.observed_all_clear.all_clear_boolean)
    
    try:
        dict["Observed SEP Probability"].append(sphinx.observed_probability[thresh_key].probability_value)
    except:
        dict["Observed SEP Probability"].append(None)

    try:
        dict["Observed SEP Threshold Crossing Time"].append(sphinx.observed_threshold_crossing[thresh_key].crossing_time)
    except:
        dict["Observed SEP Threshold Crossing Time"].append(None)

    try:
        dict["Observed SEP Start Time"].append(sphinx.observed_start_time[thresh_key])
    except:
        dict["Observed SEP Start Time"].append(None)

    try:
        dict["Observed SEP End Time"].append(sphinx.observed_end_time[thresh_key])
    except:
        dict["Observed SEP End Time"].append(None)

    dict["Observed SEP Peak Intensity (Onset Peak)"].append(sphinx.observed_peak_intensity.intensity)
    dict["Observed SEP Peak Intensity (Onset Peak) Units"].append(sphinx.observed_peak_intensity.units)
    dict["Observed SEP Peak Intensity (Onset Peak) Time"].append(sphinx.observed_peak_intensity.time)
    dict["Observed SEP Peak Intensity Max (Max Flux)"].append(sphinx.observed_peak_intensity_max.intensity)
    dict["Observed SEP Peak Intensity Max (Max Flux) Units"].append(sphinx.observed_peak_intensity_max.units)
    dict["Observed SEP Peak Intensity Max (Max Flux) Time"].append(sphinx.observed_peak_intensity_max.time)
    
    try:
        dict["Observed SEP Fluence"].append(sphinx.observed_fluence[thresh_key].fluence)
    except:
        dict["Observed SEP Fluence"].append(None)

    try:
        dict["Observed SEP Fluence Units"].append(sphinx.observed_fluence[thresh_key].units)
    except:
        dict["Observed SEP Fluence Units"].append(None)


    try:
        dict["Observed SEP Fluence Spectrum"].append(sphinx.observed_fluence_spectrum[thresh_key].fluence_spectrum)
    except:
        dict["Observed SEP Fluence Spectrum"].append(None)

    try:
        dict["Observed SEP Fluence Spectrum Units"].append(sphinx.observed_fluence_spectrum[thresh_key].fluence_units)
    except:
        dict["Observed SEP Fluence Spectrum Units"].append(None)


    dict["Predicted SEP All Clear"].append(pred_all_clear)
    dict["All Clear Match Status"].append(ac_match_status)
    dict["Predicted SEP Probability"].append(pred_prob)
    dict["Probability Match Status"].append(prob_match_status)
    dict["Predicted SEP Threshold Crossing Time"].append(pred_thresh_cross)
    dict["Threshold Crossing Time Match Status"].append(tc_match_status)
    dict["Predicted SEP Start Time"].append(pred_start_time)
    dict["Start Time Match Status"].append(st_match_status)
    dict["Predicted SEP End Time"].append(pred_end_time)
    dict["End Time Match Status"].append(et_match_status)
    dict["Predicted SEP Peak Intensity (Onset Peak)"].append(pred_peak_intensity)
    dict["Predicted SEP Peak Intensity (Onset Peak) Units"].append(pred_pi_units)
    dict["Predicted SEP Peak Intensity (Onset Peak) Time"].append(pred_pi_time)
    dict["Peak Intensity Match Status"].append(pi_match_status)
    dict["Predicted SEP Peak Intensity Max (Max Flux)"].append(pred_peak_intensity_max)
    dict["Predicted SEP Peak Intensity Max (Max Flux) Units"].append(pred_pimax_units)
    dict["Predicted SEP Peak Intensity Max (Max Flux) Time"].append(pred_pimax_time)
    dict["Peak Intensity Max Match Status"].append(pimax_match_status)
    dict["Predicted SEP Fluence"].append(pred_fluence)
    dict["Predicted SEP Fluence Units"].append(pred_fl_units)
    dict["Fluence Match Status"].append(fl_match_status)
    dict["Predicted SEP Fluence Spectrum"].append(pred_fl_spec)
    dict["Predicted SEP Fluence Spectrum Units"].append(pred_flsp_units)
    dict["Fluence Spectrum Match Status"].append(flsp_match_status)
    dict["Time Profile Match Status"].append(None)

 



def fill_df(matched_sphinx, model_names, all_energy_channels,
    all_obs_thresholds):
    """ Fill in a dictionary with the all clear predictions and observations
        organized by model and energy channel.
    """
    #sorted by model, quantity, energy channel, threshold
    dict = initialize_dict()

    #Loop through the forecasts for each model and fill in quantity_dict
    #as appropriate
    for model in model_names:
        for channel in all_energy_channels:
            ek = objh.energy_channel_to_key(channel)

            print("---Model: " + model + ", Energy Channel: " + ek)
            for sphinx in matched_sphinx[model][ek]:
                for thresh in all_obs_thresholds[ek]:
                    tk = objh.threshold_to_key(thresh)
                    fill_dict_row(sphinx, dict, ek, tk)
                
    
    df = pd.DataFrame(dict)

    df.to_csv("../output/SPHINX_dataframe.csv")


    return df


def intuitive_validation(matched_sphinx, model_names, all_energy_channels,
    all_observed_thresholds, observed_sep_events):
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
    df = fill_df(matched_sphinx, model_names,
            all_energy_channels, all_observed_thresholds)

