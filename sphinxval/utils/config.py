from . import units_handler as vunits
from . import object_handler as objh
import os
import git

datapath = './data/observations'
modelpath = './data/forecasts'
outpath = './output'
referencepath = './reference'
reportpath = './reports'
logpath = './logs'
baseurlpath = None
#baseurlpath = 'https://web-dev.ccmc.smce.nasa.gov:8001/sphinx'

######SHORTNAME#####
# Set to a list of items if you want to group a model's submodules to share
# the same shortname (example: 'UMASEP-10 WCP' -> 'UMASEP-10')
# shortname_grouping = False
shortname_grouping = [
    ('UMASEP-10 .*', 'UMASEP-10'),
    ('UMASEP-100 .*', 'UMASEP-100'),
    ('UMASEP-30 .*', 'UMASEP-30'),
    ('UMASEP-50 .*', 'UMASEP-50'),
    ('UMASEP-500 .*', 'UMASEP-500')
]

# SEP Profile Path Appendages
# Modifies SEP profile paths for models that produce time profiles.
# Models that do not predict SEP time profiles are unaffected.
# Paths are relative to the directory where the forecast JSON is stored.
sep_profile_path_relative_to_json = {
    'SAWS-ASPECS 0-6 hrs' : '../../../Profile/{year}/{month}/',
    'SAWS-ASPECS 0-6 hrs 50%' : '../../../Profile/{year}/{month}/',
    'SAWS-ASPECS 0-6 hrs 90%' : '../../../Profile/{year}/{month}/',
    'SAWS-ASPECS 0-12 hrs' : '../../../Profile/{year}/{month}/',
    'SAWS-ASPECS 0-12 hrs 50%' : '../../../Profile/{year}/{month}/',
    'SAWS-ASPECS 0-12 hrs 90%' : '../../../Profile/{year}/{month}/',
    'SAWS-ASPECS 0-24 hrs' : '../../../Profile/{year}/{month}/',
    'SAWS-ASPECS 0-24 hrs 50%' : '../../../Profile/{year}/{month}/',
    'SAWS-ASPECS 0-24 hrs 90%' : '../../../Profile/{year}/{month}/',
    'SAWS-ASPECS 0-48 hrs' : '../../../Profile/{year}/{month}/',
    'SAWS-ASPECS 0-48 hrs 50%' : '../../../Profile/{year}/{month}/',
    'SAWS-ASPECS 0-48 hrs 90%' : '../../../Profile/{year}/{month}/',
    'SAWS-ASPECS 0-72 hrs' : '../../../Profile/{year}/{month}/',
    'SAWS-ASPECS 0-72 hrs 50%' : '../../../Profile/{year}/{month}/',
    'SAWS-ASPECS 0-72 hrs 90%' : '../../../Profile/{year}/{month}/',
    'SAWS-ASPECS flare' : '../../../Profile/{year}/{month}/',
    'SAWS-ASPECS flare 50%' : '../../../Profile/{year}/{month}/',
    'SAWS-ASPECS flare 90%' : '../../../Profile/{year}/{month}/',
    'ZEUS+iPATH_CME' : '../../../{energy}MeV/{year}/{month}/',
    'ZEUS+iPATH_Flare' : '../../../{energy}MeV/{year}/{month}/',
    'SEPMOD' : './',
}


#Advanced Warning Time after observed event
awt_cut = 96
#when assessing advanced warning time, exclude forecasts that are issued
#more than awt_cut hours after an event. This is to exclude
#historical forecasts that might record issue times significantly
#after an event. Allow forecasts for up to a certain period of time
#after an event starts.

#Max time between issue time and prediction window start
max_warning_hours = 84
#Current models typically have the capability to issue a prediction
#for up to 72 hours in the future. Set max_warning_hours to be
#larger than the largest valid forecast horizon (issue time to prediction
#window start time) out of all the input models. If encounter a longer
#forecast horizon, will exclude as an erroneous forecast. Used in
#classes.py Forecast.valid_forecast().


#Peak Flux - NOT USED
peak_flux_cut = 8e-1
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
do_mismatch = False

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
mm_model = "REleASE" #Model short name contains this string
mm_pred_energy_channel = {"min": 15.8, "max": 39.8, "units": e_units}
mm_pred_threshold = {"threshold": 0.1, "threshold_units": t2_units}
#mm_pred_energy_channel = {"min": 28.2, "max": 50.1, "units": e_units}
#mm_pred_threshold = {"threshold": 0.1, "threshold_units": t_units}

#mm_model = "UNSPELL" #Model short name contains this string
#mm_pred_energy_channel = {"min": 5, "max": -1, "units": e_units}
#mm_pred_threshold = {"threshold": 5, "threshold_units": t_units}

#mm_model = "SEPMOD" #Model short name contains this string
#mm_pred_energy_channel = {"min": 10, "max": -1, "units": e_units}
#mm_pred_threshold = {"threshold": 0.001, "threshold_units": t_units}

######SET OBSERVATION INFO#######
#mm_obs_energy_channel = {"min": 25, "max": 40.9, "units": e_units}
#mm_obs_threshold = {"threshold": 0.1, "threshold_units": t_units}

mm_obs_energy_channel = {"min": 10, "max": -1, "units": e_units}
mm_obs_threshold = {"threshold": 10, "threshold_units": t_units}

###AUTOMATIC
mm_pred_ek = objh.energy_channel_to_key(mm_pred_energy_channel)
mm_pred_tk = objh.threshold_to_key(mm_pred_threshold)
mm_obs_ek = objh.energy_channel_to_key(mm_obs_energy_channel)
mm_obs_tk = objh.threshold_to_key(mm_obs_threshold)
mm_energy_key = mm_obs_ek + "_" + mm_pred_ek
mm_thresh_key = mm_obs_tk + "_" + mm_pred_tk

#Dictionaries throughout the code will use mm_energy_key to
#organize observation and model objects.
#The observed threshold key, mm_obs_tk, will be used in
#organizing observed and predicted values by threshold.
######## END MISMATCH ############



# METRICS TO BE REPORTED AS A PERCENTAGE
in_percent = ["Mean Percent Error (MPE)",
              "Mean Absolute Percent Error (MAPE)",
              "Mean Symmetric Percent Error (MSPE)",
              "Mean Symmetric Absolute Percent Error (SMAPE)",
              "Median Symmetric Accuracy (MdSA)",
              "Mean Accuracy Ratio (MAR)",
              "Prevalence Threshold"]

# SAVES THE CURRENT GIT COMMIT SHA HASH FOR LATER USE
git_repo = git.Repo(search_parent_directories=True)
git_repo_url = 'https://github.com/ktindiana/sphinxval'
git_commit_sha = git_repo.head.object.hexsha
git_is_dirty = git_repo.is_dirty()
git_changed_files = [item.a_path for item in git_repo.index.diff(None)]
git_untracked_files = git_repo.untracked_files
