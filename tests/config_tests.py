from sphinxval.utils import units_handler as vunits
# from sphinxval.utils import object_handler as objh
import os
import git
from mock import Mock
import types
import sys


outpath = './tests/output'
logpath = './tests/logs'


#Advanced Warning Time
# awt_cut = 96
#when assessing advanced warning time, exclude forecasts that are issued
#more than awt_cut hours after an event. This is to exclude
#historical forecasts that might record issue times significantly
#after an event. Allow forecasts for up to a certain period of time
#after an event starts.

#Peak Flux
# peak_flux_cut = 8e-1
#When comparing with peak flux values, if the observed
#peak flux is below peak_flux_cut, don't include that in the metrics.
#Set peak_flux_cut to something above the floor of the detector
#background


##### MISMATCH #######
#Allow Mismatching Energy Channels and Thresholds
#Allow different observed and predicted energy channels and thresholds
#to be compared to each other.
#e.g. if want to validate with observations that are "close" to the
#predicted energy channels and thresholds, but not exactly the same.
#Only one excepted case allowed in current version.
#Set do_mismatch = True to allow comparison of mismatched energy channels and thresholds
do_mismatch = True

#mm stands for "mismatch"
#WRITE UNITS IN SAME FORMAT AS REQUESTED FOR SEP SCOREBOARDS
#Write energy units as "MeV", "GeV", etc
e_units = vunits.convert_string_to_units("MeV")

#Write flux threshold units as, e.g.:
#"pfu" or "cm^-2*sr^-1*s^-1"(integral)
#"MeV^-1*s^-1*cm^-2*sr^-1" (differential)
t_units = vunits.convert_string_to_units("pfu")
t2_units = vunits.convert_string_to_units("MeV^-1*s^-1*cm^-2*sr^-1")

######SET MODEL INFO#####
mm_model = "Test_model_0" #Model short name contains this string
mm_pred_energy_channel = {"min": 10, "max": -1, "units": e_units}
mm_pred_threshold = {"threshold": 10, "threshold_units": t_units}

######SET OBSERVATION INFO#######
mm_obs_energy_channel = {"min": 10, "max": -1, "units": e_units}
mm_obs_threshold = {"threshold": 1, "threshold_units": t_units}

###AUTOMATIC
mm_pred_ek = "min." +str(float(mm_pred_energy_channel['min'])) + ".max." \
        + str(float(mm_pred_energy_channel['max'])) + ".units." \
        + vunits.convert_units_to_string(e_units)
mm_pred_tk = "threshold." +str(float(mm_pred_threshold['threshold'])) \
        + ".units." + vunits.convert_units_to_string(t_units)

mm_obs_ek = "min." +str(float(mm_obs_energy_channel['min'])) + ".max." \
        + str(float(mm_obs_energy_channel['max'])) + ".units." \
        + vunits.convert_units_to_string(e_units)
mm_obs_tk = "threshold." +str(float(mm_obs_threshold['threshold'])) \
        + ".units." + vunits.convert_units_to_string(t_units)

mm_energy_key = mm_obs_ek + "_" + mm_pred_ek
mm_thresh_key = mm_obs_tk + "_" + mm_pred_tk

#Dictionaries throughout the code will use mm_energy_key to
#organize observation and model objects.
#The observed threshold key, mm_obs_tk, will be used in
#organizing observed and predicted values by threshold.

