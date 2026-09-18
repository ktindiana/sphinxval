import sphinxval.sphinx
import argparse
import logging
import logging.config
import os
from sphinxval.utils import config as cfg
import pathlib
import json


parser = argparse.ArgumentParser()
parser.add_argument("--OutputDir", type=str, default=None, \
        help=("Look for SPHINX output files in specified directory"))
parser.add_argument("--RelativePathPlots", type=bool, default=True, \
        help=("Generate reports with relative paths for plots"))
parser.add_argument("--PartitionPath", type=str, default=None, \
        help=("Directory holding partitioned SPHINX data. If not given, "
            "config.partitionpath (as set in config.py) is used."))


#Create logger
logger = logging.getLogger(__name__)


def setup_logging():
    # Create the logs/ directory if it does not yet exist
    if not os.path.exists(cfg.logpath):
        os.mkdir(cfg.logpath)

    config_file = pathlib.Path('sphinxval/log/log_config.json')
    with open(config_file) as f_in:
        config = json.load(f_in)
    logging.config.dictConfig(config)


args = parser.parse_args()

setup_logging()

output_dir = args.OutputDir
relative_path_plots = args.RelativePathPlots

if args.PartitionPath is not None:
    cfg.partitionpath = args.PartitionPath

try:
    sphinxval.sphinx.report.report(output_dir, relative_path_plots)
except Exception:
    logger.exception('report.py failed with an exception.')
