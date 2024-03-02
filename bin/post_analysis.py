#!/usr/bin/env python

import argparse
import sphinxval.utils.post_analysis as pa
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--AllClearFalseAlarms", type=str, default='', \
        help=("pkl file containing all_clear_selections*.pkl"))

parser.add_argument("--Path", type=str, default='output', \
        help=("Path to the output directory. No trailing slash. "
            "Default ./output"))
parser.add_argument("--Models", type=str, default='All',
        help=("List of model names to include in metric boxplots "
            "(surrounded by quotes and separated by commas). "
            "May specify All for all models in a given dataframe."))
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



args = parser.parse_args()
acfa_filename = args.AllClearFalseAlarms
path = args.Path
models = args.Models
quantity = args.Quantity
anonymous = args.anonymous
highlight = args.Highlight


if acfa_filename != '':
    pa.export_all_clear_false_alarms(acfa_filename, doplot=True)


#Make summary box plots
if quantity == '':
    sys.exit("Enter a quantity.")
df = pa.read_in_metrics(path, quantity)
pa.make_box_plots(df, path, quantity, anonymous, highlight)
