datapath = '/Users/kwhitman/Documents/Programs/Validation/SPHINX/data'
modelpath = '/Users/kwhitman/Documents/Programs/Validation/SPHINX/model'
outpath = '/Users/kwhitman/Documents/Programs/Validation/SPHINX/output'
plotpath = '/Users/kwhitman/Documents/Programs/Validation/SPHINX/plots'
reportpath = '/Users/kwhitman/Documents/Programs/Validation/SPHINX/validation_reports'
figpath = reportpath + '/figures'

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

email = "kathryn.whitman@nasa.gov"  #Your email for output JSON files
