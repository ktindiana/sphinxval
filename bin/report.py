import sphinxval.sphinx
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--OutputDir", type=str, default=None, \
        help=("Look for SPHINX output files in specified directory"))
parser.add_argument("--RelativePathPlots", type=bool, default=True, \
        help=("Generate reports with relative paths for plots"))

args = parser.parse_args()
output_dir = args.OutputDir
relative_path_plots = args.RelativePathPlots

sphinxval.sphinx.report.report(output_dir, relative_path_plots)
