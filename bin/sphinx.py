import sphinxval.sphinx
import argparse
import logging
import logging.config
import os
from sphinxval.utils import config as cfg
import pathlib
import json

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
        help=("Set to any value to enable resume mode. History is read from "
            "the partitioned data at --PartitionPath (or config.partitionpath) "
            "via a lightweight index/metadata, not a single pkl file -- this "
            "is no longer interpreted as a filename."))
parser.add_argument("--ResumeProfiles", nargs = '+', default=None,
        help=("Specify the path and filename of existing profile dictionaries (as pkl) containing the observed and model profiles. "
            "This is in the form of two strings seperated by a space, each string with a set of quotes around it"))
parser.add_argument("--PartitionPath", type=str, default=None,
        help=("Directory for partitioned SPHINX_evaluated/SPHINX_removed data, the "
            "duplicate index, and metadata. Overrides config.partitionpath for this "
            "run. If not given, config.partitionpath (as set in config.py) is used."))
parser.add_argument("--RelativePathPlots", type=bool, default=True, \
        help=("Generate reports with relative paths for plots"))
parser.add_argument("--Uncertainty", action = 'store_true', default=cfg.uncert_boolean, \
        help=("Calculate Metric Uncertainties, set flag to turn True. Otherwise its default is in the config file"))


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

try:
    if args.ResumeProfiles != None:
        resume_obs = args.ResumeProfiles[0]
        resume_model = args.ResumeProfiles[1]
    else:
        resume_obs = None
        resume_model = None
    sphinx_df = sphinxval.sphinx.validate(
        args.DataList,
        args.ModelList,
        top=args.TopDirectory,
        Resume=args.Resume,
        resume_obs=resume_obs,
        resume_model=resume_model,
        uncertainty=args.Uncertainty,
        partitionpath=args.PartitionPath
        )
    sphinxval.sphinx.report.report(None, args.RelativePathPlots, sphinx_dataframe=sphinx_df)

except:
    logger.exception('SPHINX failed with an exception.')
