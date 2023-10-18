import sphinxval.sphinx
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--OutputDir", type=str, default=None, \
        help=("Look for SPHINX output files in specified directory"))

args = parser.parse_args()
output_dir = args.OutputDir

sphinxval.sphinx.report.report(output_dir)
