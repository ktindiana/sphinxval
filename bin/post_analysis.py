#!/usr/bin/env python

import argparse
import sphinxval.utils.post_analysis as pa
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--AllClearOutcomes", type=str, default='', \
        help=("pkl file containing all_clear_selections*.pkl"))
parser.add_argument("--DeoverlapAllClear", type=str, default='', \
        help=("path to csv directory containing all_clear_selections*.csv. "
            "If deoverlapping, also use --Include to specify a single model name "
            "using the exact short_name as well as --EnergyBin and --Threshold."))
parser.add_argument("--DeoverlapProbability", type=str, default='', \
        help=("path to csv directory containing probabililty_selections*.csv. "
            "If deoverlapping, also use --Include to specify a single model name "
            "using the exact short_name as well as --EnergyBin and --Threshold."))
parser.add_argument("--MaxFluxOutcomes", type=str, default='', \
        help=("pkl file containing max_flux_in_pred_win_selections*.pkl"))
parser.add_argument("--Threshold", type=float, default=10., \
        help=("Threshold for Max Flux false alarms of deoverlapping. Default 10."))
parser.add_argument("--EnergyBin", type=str, default='', \
        help=("Comma separated low energy and high energy edges written as e.g. "
            "\"5.1,9.3\" or \"10,-1\", for deoverlapping."))

parser.add_argument("--Path", type=str, default='./', \
        help=("Path to the output directory. No trailing slash. "
            "Default ./"))
parser.add_argument("--Include", type=str, default='',
        help=("List of model names to include in metric boxplots or deoverlapping "
            "(surrounded by quotes and separated by commas). May specify \"All\" for "
            "boxplots."))
parser.add_argument("--Quantity", type=str, default='',
        choices=["All Clear", "Advanced Warning Time", "Probability",
        "Threshold Crossing Time", "Start Time", "End Time", "Onset Peak Time",
        "Onset Peak", "Max Flux Time", "Max Flux", "Fluence",
        "Max Flux in Prediction Window", "Duration", "Time Profile"],
        help=("Forecasted quantity to evaluate (surround with quotes "
            "and separated by commas)."))
#parser.add_argument("--Metrics", type=str, default='All',
#        help=("List containing metric names to generate boxplots."))
parser.add_argument("--anonymous",
        help=("Make all of the plots with generic labels for models."),
        action="store_true")
parser.add_argument("--Highlight", type=str, default='',
        help=("Model name to highlight on anonymous plots."))
parser.add_argument("--Scoreboard",
        help=("Save plots to file in summary directory."),
        action="store_true")
parser.add_argument("--Exclude", type=str, default='',
        help=("Models to be excluded from the plots. Can be multiple surrounded "
            "by quotes and separated by commas."))
parser.add_argument("--saveplot",
        help=("Save plots to file in summary directory."),
        action="store_true")
parser.add_argument("--showplot",
        help=("Show plots to screen."),
        action="store_true")
parser.add_argument('--manual_run', type=str, default = '', 
        help=("Use to run sphinx for an already existing sphinx dataframe"))


args = parser.parse_args()
acfa_filename = args.AllClearOutcomes
deoverlap_ac_path = args.DeoverlapAllClear
deoverlap_prob_path = args.DeoverlapProbability
mf_filename = args.MaxFluxOutcomes
threshold = args.Threshold
path = args.Path
quantity = args.Quantity
anonymous = args.anonymous
highlight = args.Highlight
scoreboard = args.Scoreboard
saveplot = args.saveplot
showplot = args.showplot
manual_run = args.manual_run

exclude = args.Exclude.strip().split(",")
include = args.Include.strip().split(",")

# bin = args.EnergyBin.strip().split(",")
# energy_min = float(bin[0])
# energy_max = float(bin[1])




if acfa_filename != '':
    pa.export_all_clear_incorrect(acfa_filename, threshold, doplot=True)

if deoverlap_ac_path != '':
    threshold = float(threshold)
    model = include[0]
    pa.deoverlap_forecasts("All Clear", deoverlap_ac_path, model, energy_min, energy_max,
        threshold)

if deoverlap_prob_path != '':
    threshold = float(threshold)
    model = include[0]
    pa.deoverlap_forecasts("Probability", deoverlap_prob_path, model, energy_min, energy_max, threshold)

if mf_filename != '':
    pa.export_max_flux_incorrect(mf_filename, threshold, doplot=True)

#Make summary box plots
if quantity != '':
    df = pa.read_in_metrics(path, quantity, include, exclude)
    pa.make_box_plots(df, path, quantity, anonymous, highlight, scoreboard,
        saveplot, showplot)

if manual_run != '':
    sphinx_filename = manual_run
    print(sphinx_filename)
    pa.manual_sphinx(sphinx_filename)

