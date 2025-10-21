from ..utils import validation_json_handler as vjson
from ..utils import config as cfg
from ..utils import object_handler as objh
from ..utils import validation as valid
from ..utils import match
from ..utils import resume
from ..utils import report
from ..utils import duplicates
import logging
import logging.config
import datetime
import pickle
import sys
import os


logging.getLogger("MARKDOWN").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def validate(data_list, model_list, top=None, Resume=None, resume_obs = None, resume_model = None):
    """ Validate ingests a list of observations (data_list) and a
        list of predictions (model_list).
        
        The observations and predictions are automatically matched
        according to various criteria and then metrics are calculate.
        
        If resume is set to True, then the matched observation and
        prediction pairs will be appended to a previously existing
        Pandas DataFrame (filename df_pkl). The last prediction window
        for each model will be checked in the existing dataframe.
        Only new forecasts past that prediction window will be appended.
        
        INPUT:
        
            :data_list: (string array) filenames of observation jsons,
                preferably covering a continuous time period
            :model_list: (string array) filenames of model prediction jsons
            :top: (string) top directory in which to search for time profile
                files. Directory containing model jsons and .txt files.
            :Resume: (string) filename of pickle file containing previously
                calculated dataframe. New predictions will be appended.
                
        OUTPUT:
        
            :sphinx_df: (pandas dataframe) SPHINX dataframe.
            
    """
    #Reconstruct command line execution of sphinx
    logger.info("SPHINX called with: " + " ".join(sys.argv))
    logger.info("Starting SPHINX Validation and reading in files.")



    #### RESUME ####
    #If resuming, read in the dataframe specified by the user.
    #Can use SPHINX_dataframe.pkl from a previous run, because will not
    #have overwritten by this point.
    r_df = None
    r_obs_prof = None
    r_model_prof = None
    if Resume is not None:
        logger.info("RESUME: Reading in previous dataframe: "
            + Resume)
        r_df = resume.read_in_df(Resume)
    if resume_obs is not None and resume_model is not None:
        r_obs_prof, r_model_prof = resume.read_in_profile_dicts(resume_obs, resume_model)
    elif resume_obs is not None and resume_model is None or resume_model is not None and resume_obs is None:
        logger.error("Cannot use resume on the profiles without specifying both files")
        sys.exit()

    #Create Observation and Forecast objects from jsons (edge cases)
    #Unique identifier - issue time, triggers, prediction window - ignore for now
    #Use last prediction window for model or energy_channel to include new
    #obs_objs and model_objs organized by energy channel
    all_energy_channels, obs_objs, model_objs, removed_model_objs =\
        vjson.load_objects_from_json(data_list, model_list)
    logger.info("Loaded all JSON files into Observation and Forecast Objects.")

    #Exclude invalid forecasts first
    model_objs, removed_invalid = objh.remove_invalid_forecasts(model_objs, all_energy_channels)

    #Check for duplicates in the set of forecasts and remove any
    logger.info("Checking for and removing duplicate forecasts from the input Model List.")
    model_objs, removed_fcast = duplicates.remove_forecast_duplicates(all_energy_channels, model_objs)

    #Dictionary containing the location of all the .txt files
    #in the subdirectories below top.
    #Used to specify where the time profile .txt files are.
    profname_dict = None
    if top is not None:
        profname_dict = vjson.generate_profile_dict(top)
        logger.info("Generated dictionary of all txt files in " + top + " to store the locations of time profiles.")

    #Identify the unique models represented in the forecasts
    model_names = objh.build_model_list(model_objs, cfg.shortname_grouping)
    logger.info("Identified unique model names.")
    
    #### RESUME ####
    #Compare the newly read in forecasts to the resume dataframe and remove
    #any duplicates from the new forecasts
    if Resume is not None:
        model_objs, removed_resume = duplicates.remove_resume_duplicates(r_df, model_objs)
    ################
    
    
    #Dictionary of SPHINX objects containing all matching criteria
    #and matched observed values for each forecast (evaluated_sphinx) as well as
    #a dictionary organized by model, energy channel, and threshold with the
    #unique SEP events that were present inside of the forecast prediction
    #windows (observed_sep_events).
    logger.info("Starting matching process.")
    evaluated_sphinx, removed_sphinx, all_observed_thresholds, observed_sep_events =\
        match.match_all_forecasts(all_energy_channels, model_names,
            obs_objs, model_objs)
    logger.info("Completed matching process and starting intuitive validation.")


    #Add the excluded duplicates to removed_sphinx
    removed_sphinx = duplicates.add_to_not_evaluated(removed_sphinx, removed_model_objs)
    removed_sphinx = duplicates.add_to_not_evaluated(removed_sphinx, removed_invalid)
    removed_sphinx = duplicates.add_to_not_evaluated(removed_sphinx, removed_fcast, "Duplicate input forecast")
    if Resume is not None:
        removed_sphinx = duplicates.add_to_not_evaluated(removed_sphinx, removed_resume, "Duplicate forecast already present in the resume dataframe")


    #Perform intuitive validation
    sphinx_df = valid.intuitive_validation(evaluated_sphinx, removed_sphinx, model_names,
        all_energy_channels, all_observed_thresholds, observed_sep_events, profname_dict, r_df=r_df, r_obs = r_obs_prof, r_mod = r_model_prof)
    logger.info("Completed validation.")


    return sphinx_df


    
