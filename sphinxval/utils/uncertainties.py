import numpy as np
from scipy import stats

"""
Independent module intended to determine metric uncertainties within
SPHINX with extensions for VIVID. 

Plan/Outline:
    For SPHINX - 
        Grab _selections files for each model/predicted quantity subset
        Use Scipy.stats bootstrap to resample and calculate the standard error
        Add column to metrics file for the metric uncertainties

    VIVID -
        Add wrapper from VIVID feeder to generate the same subset that 
            is given to the _selections files but doesn't need to actually
            read in the file
        Do the same bootstrapping
        Give out metric uncertainties (need to talk to Phil about VIVID inputs)
"""



def feeder_from_sphinx():


    return


def feeder_from_vivid():


    return