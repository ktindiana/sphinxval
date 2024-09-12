from ..utils import validation_json_handler as vjson
from ..utils import config as cfg
from ..utils import object_handler as objh
from ..utils import validation as valid
from ..utils import match
from ..utils import resume
from ..utils import report
from ..utils import duplicates
import datetime
import pickle
import logging
import logging.config
import sys
import os
import pathlib
import json

logging.getLogger("MARKDOWN").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

#Create logger
logger = logging.getLogger(__name__)


def setup_logging():
    # Create the logs/ directory if it does not yet exist
    if not os.path.exists(cfg.logpath):
        os.mkdir(cfg.logpath)

    config_file = pathlib.Path('sphinxval/log/log_config.json')
    with open(config_file) as f_in:
        config = json.load(f_in)
    logging.config.dictConfig(config)


def validate(data_list, model_list, top=None, Resume=None):
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
        
            None
            
    """
    setup_logging()
    
    #Reconstruct command line execution of sphinx
    logger.info("SPHINX called with: " + " ".join(sys.argv))
    logger.info("Starting SPHINX Validation and reading in files.")

    #Create Observation and Forecast objects from jsons (edge cases)
    #Unique identifier - issue time, triggers, prediction window - ignore for now
    #Use last prediction window for model or energy_channel to include new
    #obs_objs and model_objs organized by energy channel
    all_energy_channels, obs_objs, model_objs =\
        vjson.load_objects_from_json(data_list, model_list)
    logger.info("Loaded all JSON files into Objects.")

    #Check for duplicates in the set of forecasts and remove any
    model_objs = duplicates.remove_forecast_duplicates(all_energy_channels, model_objs)

    #Dictionary containing the location of all the .txt files
    #in the subdirectories below top.
    #Used to specify where the time profile .txt files are.
    profname_dict = None
    if top != None:
        profname_dict = vjson.generate_profile_dict(top)
        logger.info("Generated dictionary of all txt files in " + top)

    #Identify the unique models represented in the forecasts
    model_names = objh.build_model_list(model_objs)
    logger.info("Built model list.")
    
    #### RESUME ####
    #If resuming, read in the dataframe specified by the user.
    #Can use SPHINX_dataframe.pkl from a previous run, because will not
    #have overwritten by this point.
    r_df = None
    if Resume is not None:
        logger.info("RESUME: Reading in previous dataframe: "
            + Resume)
        r_df = resume.read_in_df(Resume)
    ################
    
    
    #Dictionary of SPHINX objects containing all matching criteria
    #and matched observed values for each forecast (matched_sphinx) as well as
    #a dictionary organized by model, energy channel, and threshold with the
    #unique SEP events that were present inside of the forecast prediction
    #windows (observed_sep_events).
    logger.info("Starting matching process.")
    matched_sphinx, all_observed_thresholds, observed_sep_events =\
        match.match_all_forecasts(all_energy_channels, model_names,
            obs_objs, model_objs)
    logger.info("Completed matching process and starting intuitive validation.")


    #Perform intuitive validation
    valid.intuitive_validation(matched_sphinx, model_names,
        all_energy_channels, all_observed_thresholds, observed_sep_events, profname_dict, r_df=r_df)
    logger.info("Completed validation.")


    
