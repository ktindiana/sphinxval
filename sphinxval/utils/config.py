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
#profile paths
model_prof_path = './output/json/model_profiles.json'
obs_prof_path = './output/json/observed_profiles.json'
#partitionpath = '/data/SPHINX/active/partitions'
partitionpath = './data/partitions'
os.makedirs(partitionpath, exist_ok=True)
baseurlpath = 'https://web-dev.ccmc.smce.nasa.gov:8001/sphinx'

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

#KEY COLUMNS THAT UNIQUELY IDENTIFY A FORECAST ROW. SHARED BY
#sphinxval.utils.duplicates.remove_sphinx_duplicates AND
#sphinxval.utils.duplicates.remove_new_duplicates_against_index SO BOTH
#FUNCTIONS AGREE ON WHAT COUNTS AS A DUPLICATE. CANNOT USE ALL DF ENTRIES
#BECAUSE THE HASH COMMAND CANNOT HASH LISTS.
SPHINX_KEY_COLUMNS = ["Model", "Energy Channel Key", "Threshold Key", "Mismatch Allowed",
        "Prediction Energy Channel Key", "Prediction Threshold Key", "Prediction Window Start",
        "Prediction Window End", "Prediction Number of CMEs","Prediction CME Start Time",
        "Prediction CME Liftoff Time", "Prediction CME Latitude", "Prediction CME Longitude",
        "Prediction CME Speed", "Prediction CME Half Width", "Prediction CME PA",
        "Prediction Number of Flares", "Prediction Flare Latitude", "Prediction Flare Longitude",
        "Prediction Flare Start Time", "Prediction Flare Peak Time", "Prediction Flare End Time",
        "Prediction Flare Last Data Time", "Prediction Flare Intensity",
        "Prediction Flare Integrated Intensity", "Prediction Flare NOAA AR",
        "Observed SEP CME Start Time",
        "Observed SEP CME Liftoff Time", "Observed SEP CME Latitude", "Observed SEP CME Longitude",
        "Observed SEP CME Speed", "Observed SEP CME Half Width", "Observed SEP CME PA",
        "Observed SEP Flare Latitude", "Observed SEP Flare Longitude",
        "Observed SEP Flare Start Time", "Observed SEP Flare Peak Time",
        "Observed SEP Flare End Time",
        "Observed SEP Flare Intensity",
        "Observed SEP Flare Integrated Intensity", "Observed SEP Flare NOAA AR",
        "Observatory", "Observed SEP All Clear",
        "Predicted SEP All Clear", "Predicted SEP All Clear Probability Threshold",
        "All Clear Match Status", "Predicted SEP Probability",
        "Probability Match Status", "Predicted SEP Threshold Crossing Time",
        "Threshold Crossing Time Match Status", "Predicted SEP Start Time",
        "Start Time Match Status", "Predicted SEP End Time", "End Time Match Status",
        "Predicted SEP Duration", "Duration Match Status", "Predicted SEP Fluence",
        "Fluence Match Status", "Predicted SEP Peak Intensity (Onset Peak)",
        "Peak Intensity Match Status", "Predicted SEP Peak Intensity Max (Max Flux)",
        "Peak Intensity Max Match Status", "Predicted Point Intensity",
        "Predicted Time Profile", "Time Profile Match Status"]

#COLUMNS THAT HOLD LIVE astropy.units.Unit OBJECTS RATHER THAN STRINGS.
#PARQUET CANNOT SERIALIZE ARBITRARY PYTHON OBJECTS (UNLIKE PICKLE), SO
#THESE MUST BE CONVERTED TO STRINGS BEFORE write_partition_df AND BACK TO
#Unit OBJECTS AFTER READING PARTITIONS BACK INTO A DATAFRAME.
UNITS_COLUMNS = [
    "Observed SEP Peak Intensity (Onset Peak) Units",
    "Observed SEP Peak Intensity Max (Max Flux) Units",
    "Observed Point Intensity Units",
    "Observed Max Flux in Prediction Window Units",
    "Observed SEP Fluence Units",
    "Observed SEP Fluence Spectrum Units",
    "Predicted Point Intensity Units",
    "Predicted SEP Peak Intensity (Onset Peak) Units",
    "Predicted SEP Peak Intensity Max (Max Flux) Units",
    "Predicted SEP Fluence Units",
    "Predicted SEP Fluence Spectrum Units",
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
#
#Multiple rules are supported. Each rule is fully independent -- any
#combination of (model, pred_energy_channel, pred_threshold,
#obs_energy_channel, obs_threshold) is allowed, including:
#  - the same model with multiple mismatched energy-channel pairs
#  - the same model + energy pair with multiple mismatched thresholds
#  - different models each with their own rule(s)
#Set do_mismatch = True to allow comparison of mismatched energy channels
#and thresholds. If do_mismatch is True but mismatch_rules is empty, no
#mismatching will actually occur.
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

#Each entry is one independent mismatch rule. "model" is matched as a
#substring against a forecast's short_name. Add as many rules as
#needed, including multiple entries for the same model.
mismatch_rules = [
    {
        "model": "REleASE",
        "pred_energy_channel": {"min": 15.8, "max": 39.8, "units": e_units},
        "pred_threshold": {"threshold": 0.1, "threshold_units": t2_units},
        "obs_energy_channel": {"min": 10, "max": -1, "units": e_units},
        "obs_threshold": {"threshold": 10, "threshold_units": t_units},
    },
    {
        "model": "REleASE",
        "pred_energy_channel": {"min": 28.2, "max": 50.1, "units": e_units},
        "pred_threshold": {"threshold": 0.1, "threshold_units": t2_units},
        "obs_energy_channel": {"min": 10, "max": -1, "units": e_units},
        "obs_threshold": {"threshold": 10, "threshold_units": t_units},
    },
    {
        "model": "REleASE",
        "pred_energy_channel": {"min": 28.2, "max": 50.1, "units": e_units},
        "pred_threshold": {"threshold": 0.1, "threshold_units": t2_units},
        "obs_energy_channel": {"min": 100, "max": -1, "units": e_units},
        "obs_threshold": {"threshold": 1, "threshold_units": t_units},
    },
]

###AUTOMATIC -- derive lookup keys for each rule. Do not edit below this line.
for _mm_rule in mismatch_rules:
    _mm_rule["pred_ek"] = objh.energy_channel_to_key(_mm_rule["pred_energy_channel"])
    _mm_rule["pred_tk"] = objh.threshold_to_key(_mm_rule["pred_threshold"])
    _mm_rule["obs_ek"] = objh.energy_channel_to_key(_mm_rule["obs_energy_channel"])
    _mm_rule["obs_tk"] = objh.threshold_to_key(_mm_rule["obs_threshold"])
    #Dictionaries throughout the code use energy_key to organize
    #observation and model objects.
    _mm_rule["energy_key"] = _mm_rule["obs_ek"] + "_" + _mm_rule["pred_ek"]
    #The observed threshold key is used in organizing observed and
    #predicted values by threshold.
    _mm_rule["thresh_key"] = _mm_rule["obs_tk"] + "_" + _mm_rule["pred_tk"]
del _mm_rule

#Deduplicated view by energy_key -- safe for extraction-value lookups
#only (e.g. "which pred_energy_channel to pull from a forecast json for
#this energy_key"), not for applicability checks ("does a rule apply to
#this model"). Any rules sharing an energy_key are guaranteed to have
#identical obs_ek/pred_ek (energy_key is literally built from them), so
#picking any one representative rule per energy_key is safe for value
#lookups. It is not safe for checking which model(s) a rule applies to:
#if two different models happen to share the same energy_key, deduping
#by energy_key alone would silently drop one model's rule from this
#view. Applicability checks search the full mismatch_rules list instead.
mismatch_rules_by_energy_key = {}
for _mm_rule in mismatch_rules:
    if _mm_rule["energy_key"] not in mismatch_rules_by_energy_key:
        mismatch_rules_by_energy_key[_mm_rule["energy_key"]] = _mm_rule
del _mm_rule

#Plain, model-agnostic set of distinct energy_key strings -- used for
#bucket creation (obs_objs/model_objs dictionary keys, all_energy_channels
#append), where only the key string itself matters, not which model(s)
#use it.
mismatch_energy_keys = sorted(set(_mm_rule["energy_key"] for _mm_rule in mismatch_rules))
######## END MISMATCH ############


###Uncertainty Boolean
uncert_boolean = False
uncert_n_resamples = 10000
uncert_fraction = 0.75

# METRICS TO BE REPORTED AS A PERCENTAGE
in_percent = ["Mean Percent Error (MPE)",
              "Mean Absolute Percent Error (MAPE)",
              "Mean Symmetric Percent Error (MSPE)",
              "Mean Symmetric Absolute Percent Error (SMAPE)",
              "Median Symmetric Accuracy (MdSA)",
              "Mean Accuracy Ratio (MAR)",
              "Prevalence Threshold",
              "Percentage within an Order of Magnitude (%)",
              "Percentage within a factor of 2 (%)"]

# SAVES THE CURRENT GIT COMMIT SHA HASH FOR LATER USE
git_repo = git.Repo(search_parent_directories=True)
git_repo_url = 'https://github.com/ktindiana/sphinxval'
git_commit_sha = git_repo.head.object.hexsha
git_is_dirty = git_repo.is_dirty()
git_changed_files = [item.a_path for item in git_repo.index.diff(None)]
git_untracked_files = git_repo.untracked_files
