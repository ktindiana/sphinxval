import sphinxval.sphinx
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--ModelList", type=str, default='', \
        help=("List containing filenames of prediction json files."))
parser.add_argument("--DataList", type=str, default="",
        help=("List containing filenames of observation json files."))
parser.add_argument("--TopDirectory", default=None,
        help=("If json files and time profile txt files not in the same directory, "
            "specify the base path in which the model json and time profile txt files "
            "might be found. Optional. "))
parser.add_argument("--Resume", type=str, default=None,
        help=("Specify filename of existing Pandas DataFrame containing the results of a previous SPHINX run (pkl) to add forecasts to an existing "
            "dataframe and recalculate metrics."))
parser.add_argument("--RelativePathPlots", type=bool, default=True, \
        help=("Generate reports with relative paths for plots"))



args = parser.parse_args()

sphinxval.sphinx.validate(args.DataList, args.ModelList, top=args.TopDirectory, Resume=args.Resume)
sphinxval.sphinx.report.report(None, args.RelativePathPlots)
