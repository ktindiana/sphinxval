#!/usr/bin/env python

import argparse
import sphinxval.utils.post_analysis as pa
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--AllClearFalseAlarms", type=str, default='', \
        help=("pkl file containing all_clear_selections*.pkl"))

parser.add_argument("--Path", type=str, default='./', \
        help=("Path to the output directory. No trailing slash. "
            "Default ./"))
parser.add_argument("--Include", type=str, default='All',
        help=("List of model names to include in metric boxplots "
            "(surrounded by quotes and separated by commas). "
            "Any unique substring that is in the model short name is sufficient. "
            "May specify All for all models in a given dataframe. Default = All."))
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
parser.add_argument("--Exclude", type=str, default='',
        help=("Models to be excluded from the plots. Can be multiple surrounded "
            "by quotes and separated by commas."))
parser.add_argument("--saveplot",
        help=("Save plots to file in summary directory."),
        action="store_true")
parser.add_argument("--showplot",
        help=("Show plots to screen."),
        action="store_true")


args = parser.parse_args()
acfa_filename = args.AllClearFalseAlarms
path = args.Path
include = args.Include
quantity = args.Quantity
anonymous = args.anonymous
highlight = args.Highlight
exclude = args.Exclude
saveplot = args.saveplot
showplot = args.showplot

exclude = exclude.strip().split(",")
include = include.strip().split(",")

if acfa_filename != '':
    pa.export_all_clear_false_alarms(acfa_filename, doplot=True)


#Make summary box plots
if quantity == '':
    sys.exit("Enter a quantity.")
df = pa.read_in_metrics(path, quantity, include, exclude)
pa.make_box_plots(df, path, quantity, anonymous, highlight, saveplot, showplot)

