from ..utils import validation_json_handler as vjson
from ..utils import config as cfg
from ..utils import object_handler as objh
from ..utils import tools
from ..utils import match
import logging
import sys
import os



    
def validate(data_list, model_list):

    #Create Observation and Forecast objects from jsons
    all_energy_channels, obs_objs, model_objs =\
        vjson.load_objects_from_json(data_list, model_list)
    
    #Identify the unique models represented in the forecasts
    model_names = objh.build_model_list(model_objs)
    
    #Dictionary of SPHINX objects containing all matching criteria
    #and matched observed values for each forecast
    match_sphinx =\
        match.match_all_forecasts(all_energy_channels,
        obs_objs, model_objs)
