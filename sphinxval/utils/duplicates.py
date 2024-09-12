from . import object_handler as objh
import sys
import datetime
import pandas as pd
import logging

__author__ = "Katie Whitman"
__maintainer__ = "Katie Whitman"
__email__ = "kathryn.whitman@nasa.gov"

""" utils/resume.py contains subroutines to aid in resuming
    the validation process from a starting dataframe.
    
"""

#Create logger
logger = logging.getLogger(__name__)


def initialize_forecast_dict():
    """ Set up a dictionary for a pandas df to hold each possible quantity
        stored in Forecast objects.
        
    """
    #Convert to Pandas dataframe
    #Include triggers with as much flattened info
    #If need multiple dimension, then could be used as tooltip info
    #Last CME, N CMEs, Last speed, last location, Timestamps array of all CMEs used
    

    dict = {"Model": [],
            "Energy Channel Key": [],
            "Threshold Key": [],
            "Forecast Source": [],
            "Forecast Path": [],
            "Forecast Issue Time":[],
            "Prediction Window Start": [],
            "Prediction Window End": [],
            
            #OBSERVATIONS
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
            
            #PREDICTIONS
            "Predicted SEP All Clear": [],
            "Predicted SEP Probability": [],
            "Predicted SEP Threshold Crossing Time": [],
            "Predicted SEP Start Time":[],
            "Predicted SEP End Time": [],
            "Predicted SEP Fluence": [],
            "Predicted SEP Fluence Units": [],
            "Predicted SEP Fluence Spectrum": [],
            "Predicted SEP Fluence Spectrum Units": [],
            "Predicted SEP Peak Intensity (Onset Peak)": [],
            "Predicted SEP Peak Intensity (Onset Peak) Units": [],
            "Predicted SEP Peak Intensity (Onset Peak) Time": [],
            "Predicted SEP Peak Intensity Max (Max Flux)": [],
            "Predicted SEP Peak Intensity Max (Max Flux) Units": [],
            "Predicted SEP Peak Intensity Max (Max Flux) Time": [],
            
            "Predicted Point Intensity": [],
            "Predicted Point Intensity Units": [],
            "Predicted Point Intensity Time": [],

            "Predicted Time Profile": []
            
            }

    return dict


def fill_forecast_dict_row(prediction, dict):
    """ Add a row to a dataframe with all of the supporting information
        inside of a forecast.

        This dictionary is created for the purpose of removing
        duplicates. Not all predicted values are preserved in correct
        formats. Some are converted to strings if they are a complex data type.
        
    Input:
    
        :prediction: (Forecast object) contains all prediction and matched observation
            information
        :dict: (Dictionary) dictionary initialized with initialize_forecast_dict()
        
    Output:
    
        None; dict is updated by reference
        
    """
    
    energy_key = objh.energy_channel_to_key(prediction.energy_channel)
    all_thresholds = prediction.identify_all_thresholds()

    ncme = len(prediction.cmes)
    if ncme > 0:
        cme_start = prediction.cmes[-1].start_time
        cme_liftoff = prediction.cmes[-1].liftoff_time
        cme_lat = prediction.cmes[-1].lat
        cme_lon = prediction.cmes[-1].lon
        cme_pa = prediction.cmes[-1].pa
        cme_half_width = prediction.cmes[-1].half_width
        cme_speed = prediction.cmes[-1].speed
        cme_catalog = prediction.cmes[-1].catalog
    else:
        cme_start = None
        cme_liftoff = None
        cme_lat = None
        cme_lon = None
        cme_pa = None
        cme_half_width = None
        cme_speed = None
        cme_catalog = None
        
    nfl = len(prediction.flares)
    if nfl > 0:
        fl_lat = prediction.flares[-1].lat
        fl_lon = prediction.flares[-1].lon
        fl_last_data_time = prediction.flares[-1].last_data_time
        fl_start_time = prediction.flares[-1].start_time
        fl_peak_time = prediction.flares[-1].peak_time
        fl_end_time = prediction.flares[-1].end_time
        fl_intensity = prediction.flares[-1].intensity
        fl_integrated_intensity = prediction.flares[-1].integrated_intensity
        fl_AR = prediction.flares[-1].noaa_region
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


    #Predicted probabilities
    pred_prob = repr(sorted([prob.probability_value for prob in prediction.probabilities]))

    #Threshold crossings
    pred_thresh_cross = repr(sorted([tc.crossing_time for tc in prediction.threshold_crossings]))

    #Start times
    pred_start_time = repr(sorted([ev.start_time for ev in prediction.event_lengths]))

    #End times
    pred_end_time = repr(sorted([ev.end_time for ev in prediction.event_lengths]))
    
    #Fluence
    pred_fluence = repr(sorted([fl.fluence for fl in prediction.fluences]))
    pred_fl_units = repr(sorted([fl.units for fl in prediction.fluences]))

    #Fluence spectra
    pred_fl_spec = repr(sorted([flsp.fluence_spectrum for flsp in prediction.fluence_spectra]))
    pred_flsp_units = repr(sorted([flsp.fluence_units for flsp in prediction.fluence_spectra]))

    #Point intensity
    pred_point_intensity = prediction.point_intensity.intensity
    pred_pti_units = prediction.point_intensity.units
    pred_pti_time = prediction.point_intensity.time

    #Peak intensity
    pred_peak_intensity = prediction.peak_intensity.intensity
    pred_pi_units = prediction.peak_intensity.units
    pred_pi_time = prediction.peak_intensity.time

    #Peak intensity max
    pred_peak_intensity_max = prediction.peak_intensity_max.intensity
    pred_pimax_units = prediction.peak_intensity_max.units
    pred_pimax_time = prediction.peak_intensity_max.time

    #SEP time profile
    pred_time_profile = prediction.sep_profile

    dict["Model"].append(prediction.short_name)
    dict["Energy Channel Key"].append(energy_key)
    dict["All Thresholds in Prediction"].append(repr(prediction.all_thresholds))
    dict["Forecast Source"].append(prediction.source)
    dict["Forecast Path"].append(prediction.path)
    dict["Forecast Issue Time"].append(prediction.issue_time)
    dict["Prediction Window Start"].append(prediction.prediction_window_start)
    dict["Prediction Window End"].append(prediction.prediction_window_end)
    dict["Number of CMEs"].append(ncme)
    dict["CME Start Time"].append(cme_start) #Timestamp of 1st
            #coronagraph image CME is visible in
    dict["CME Liftoff Time"].append(cme_liftoff) #Timestamp of coronagraph
            #image with 1st indication of CME liftoff (used by CACTUS)
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
   

    #PREDICTION INFORMATION
    dict["Predicted SEP All Clear"].append(prediction.all_clear.all_clear_boolean)
    dict["Predicted SEP Probability"].append(pred_prob)
    dict["Predicted SEP Threshold Crossing Time"].append(pred_thresh_cross)
    dict["Predicted SEP Start Time"].append(pred_start_time)
    dict["Predicted SEP End Time"].append(pred_end_time)
    dict["Predicted Point Intensity"].append(pred_point_intensity)
    dict["Predicted Point Intensity Units"].append(pred_pti_units)
    dict["Predicted Point Intensity Time"].append(pred_pti_time)
    dict["Predicted SEP Peak Intensity (Onset Peak)"].append(pred_peak_intensity)
    dict["Predicted SEP Peak Intensity (Onset Peak) Units"].append(pred_pi_units)
    dict["Predicted SEP Peak Intensity (Onset Peak) Time"].append(pred_pi_time)
    dict["Predicted SEP Peak Intensity Max (Max Flux)"].append(pred_peak_intensity_max)
    dict["Predicted SEP Peak Intensity Max (Max Flux) Units"].append(pred_pimax_units)
    dict["Predicted SEP Peak Intensity Max (Max Flux) Time"].append(pred_pimax_time)
    dict["Predicted SEP Fluence"].append(pred_fluence)
    dict["Predicted SEP Fluence Units"].append(pred_fl_units)
    dict["Predicted SEP Fluence Spectrum"].append(pred_fl_spec)
    dict["Predicted SEP Fluence Spectrum Units"].append(pred_flsp_units)
    dict["Predicted Time Profile"].append(pred_time_profile)



def identify_forecast_duplicates(df):
    """ Check the Forecast dataframe for duplicate entries. Issue warning
        and remove repeated forecasts, combined with observatory information.
        
        Forecasts will be considered duplicate if all fields in the
        dataframe are exactly the same.
        
        Output:
        
            :df: (dataframe) with unique entries
        
    """
    #Extract key rows from the df that uniquely identify a forecast
    #Cannot use all df entries, because the hash command cannot hash lists.
    sub = df[["Model", "Energy Channel Key", "All Thresholds in Prediction",
            "Prediction Window Start", "Prediction Window End",
            "Number of CMEs","CME Start Time", "CME Liftoff Time",
            "CME Latitude", "CME Longitude", "CME Speed", "CME Half Width", "CME PA",
            "Number of Flares", "Flare Latitude", "Flare Longitude", "Flare Start Time",
            "Flare Peak Time", "Flare End Time", "Flare Last Data Time", "Flare Intensity",
            "Flare Integrated Intensity", "Flare NOAA AR",
            "Predicted SEP All Clear", "Predicted SEP Probability",
            "Predicted SEP Threshold Crossing Time",
            "Predicted SEP Start Time",
            "Predicted SEP End Time",
            "Predicted SEP Fluence",
            "Predicted SEP Peak Intensity (Onset Peak)",
            "Predicted SEP Peak Intensity Max (Max Flux)",
            "Predicted Point Intensity", "Predicted Time Profile"]]
    
    #Create a hash for each row of the dataframe
    hash = pd.util.hash_pandas_object(sub, index=False)
    duplicates = hash.duplicated(keep='first')
    dup = pd.DataFrame(duplicates)
    
    #Duplicated entries
    dup_df = df.loc[(dup[0] == True)]
    dup_indices = dup_df.index.to_list()
    
    #Keep only the entries that are marked as False for duplicates
    unique_df = df.loc[(dup[0] == False)]
    
    return unique_df, dup_indices



def fill_forecast_df(model_objs):
    """ Fill in a dictionary with the information from each forecast read into SPHINX.
    """
    #sorted by model, quantity, energy channel, threshold
    dict = initialize_forecast_dict()

    #Loop through the forecasts for each model and fill in quantity_dict
    #as appropriate
    for fcast in model_objs:
        fill_forecast_dict_row(fcast, dict)
 
    df = pd.DataFrame(dict)
    
    return df
    
    




def remove_forecast_duplicates(all_energy_channels, model_objs):
    """ Remove any duplicated Forecast objects from the model_objs array.
    
    """

    for energy_key in all_energy_channels:
        df = fill_forecast_df(model_objs[energy_key])

        #Check for duplicated forecasts and remove
        df, dup_indices = identify_forecast_duplicates(df)

        for i in range(len(dup_indices)-1,-1,-1):
            logger.warning(f"DUPLICATE: Removing duplicated forecast for energy channel {energy_key},  {model_objs[energy_key][dup_indices[i]].source}")
            model_objs[energy_key].pop(dup_indices[i])
    
    return model_objs





def remove_sphinx_duplicates(df):
    """ Check the SPHINX dataframe for duplicate entries. Issue warning
        and remove repeated forecasts, combined with observatory information.
        
        Forecasts will be considered duplicate if all fields in the
        dataframe are exactly the same.
        
        Output:
        
            :df: (dataframe) with unique entries
        
    """
    #Extract key rows from the df that uniquely identify a forecast
    #Cannot use all df entries, because the hash command cannot hash lists.
    sub = df[["Model", "Energy Channel Key", "Threshold Key", "Mismatch Allowed",
            "Prediction Energy Channel Key", "Prediction Threshold Key", "Prediction Window Start",
            "Prediction Window End", "Number of CMEs","CME Start Time", "CME Liftoff Time",
            "CME Latitude", "CME Longitude", "CME Speed", "CME Half Width", "CME PA",
            "Number of Flares", "Flare Latitude", "Flare Longitude", "Flare Start Time",
            "Flare Peak Time", "Flare End Time", "Flare Last Data Time", "Flare Intensity",
            "Flare Integrated Intensity", "Flare NOAA AR", "Observatory", "Observed SEP All Clear",
            "Predicted SEP All Clear", "All Clear Match Status", "Predicted SEP Probability",
            "Probability Match Status", "Predicted SEP Threshold Crossing Time",
            "Threshold Crossing Time Match Status", "Predicted SEP Start Time",
            "Start Time Match Status", "Predicted SEP End Time", "End Time Match Status",
            "Predicted SEP Duration", "Duration Match Status", "Predicted SEP Fluence",
            "Fluence Match Status", "Predicted SEP Peak Intensity (Onset Peak)",
            "Peak Intensity Match Status", "Predicted SEP Peak Intensity Max (Max Flux)",
            "Peak Intensity Max Match Status", "Predicted Point Intensity",
            "Predicted Time Profile", "Time Profile Match Status"]]
    
    #Create a hash for each row of the dataframe
    hash = pd.util.hash_pandas_object(sub, index=False)
    duplicates = hash.duplicated(keep='first')
    dup = pd.DataFrame(duplicates)
    
    #Duplicated entries
    dup_df = df.loc[(dup[0] == True)]
    for entry in dup_df["Forecast Source"]:
        logger.warning("DUPLICATE: " + str(entry) + " is a duplicated forecast in the SPHINX dataframe. Removing." )
    
    #Keep only the entries that are marked as False for duplicates
    unique_df = df.loc[(dup[0] == False)]
    
    return unique_df
