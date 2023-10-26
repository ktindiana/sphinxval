from . import units_handler as vunits

datapath = './data'
modelpath = './model'
outpath = './output'
referencepath = './reference'
reportpath = './reports'

#Advanced Warning Time
awt_cut = 96
#when assessing advanced warning time, exclude forecasts that are issued
#more than awt_cut hours after an event. This is to exclude
#historical forecasts that might record issue times significantly
#after an event. Allow forecasts for up to a certain period of time
#after an event starts.

#Peak Flux
peak_flux_cut = 8e-1
#When comparing with peak flux values, if the observed
#peak flux is below peak_flux_cut, don't include that in the metrics.
#Set peak_flux_cut to something above the floor of the detector
#background

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

#MODEL INFO
mm_model = "SEPMOD" #Model short name contains this string
mm_pred_energy_channel = {"min": 10, "max": -1, "units": e_units}
mm_pred_threshold = {"threshold": 0.001, "threshold_units": t_units}

#OBSERVATION INFO
mm_obs_energy_channel = {"min": 10, "max": -1, "units": e_units}
mm_obs_threshold = {"threshold": 10, "threshold_units": t_units}



email = "kathryn.whitman@nasa.gov"  #Your email for output JSON files
