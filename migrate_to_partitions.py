"""
ONE-TIME MIGRATION SCRIPT.

Splits an existing monolithic SPHINX_cumulative.pkl / SPHINX_evaluated.pkl
into monthly Parquet partitions, and builds the lightweight duplicate index
and metadata files that sphinx.py / validation.py now expect for resuming.

RUN THIS ONCE, BEFORE THE NEXT RESUMED RUN, THEN NEVER AGAIN (unless you
need to rebuild the index/metadata from scratch for some reason).

IMPORTANT: the duplicate index and metadata files are shared, single files
(not per-dataset). If you are migrating MULTIPLE datasets (e.g. both
SPHINX_evaluated and SPHINX_removed), only the FIRST/PRIMARY dataset
(SPHINX_evaluated) should write the index/metadata -- pass
--skip-index-metadata for any additional dataset, or the later run will
silently overwrite the correct index/metadata with the wrong dataset's
much smaller stats.

USAGE:
    python migrate_to_partitions.py /path/to/SPHINX_evaluated.pkl SPHINX_evaluated
    python migrate_to_partitions.py /path/to/SPHINX_removed.pkl SPHINX_removed --skip-index-metadata
"""

import sys
import os
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sphinxval.utils import config
from sphinxval.utils import resume
from sphinxval.utils import duplicates
from sphinxval.utils import validation as valid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def migrate(source_pkl_path, dataset_name="SPHINX_evaluated", skip_index_metadata=False):
    """ Migrate an existing monolithic SPHINX dataframe pkl into monthly
        Parquet partitions, plus (optionally) the duplicate index and
        metadata files.

    INPUT:

        :source_pkl_path: (string) path to the existing SPHINX_cumulative.pkl
            or SPHINX_evaluated.pkl to migrate
        :dataset_name: (string) name to use for the partitioned dataset,
            e.g. "SPHINX_evaluated" -- must match what sphinx.py/
            validation.py expect when reading partitions back
        :skip_index_metadata: (bool) if True, only write partitions --
            do NOT touch SPHINX_evaluated_index.parquet or
            SPHINX_metadata.pkl. Use this for any dataset beyond the
            primary one (SPHINX_evaluated), since the index/metadata
            files are shared, single files, not per-dataset -- writing
            them from a second, smaller dataset overwrites the correct
            values rather than merging with them.

    OUTPUT:

        Writes partition files under config.partitionpath/dataset_name/,
        and (unless skip_index_metadata) the duplicate index at
        config.partitionpath/SPHINX_evaluated_index.parquet and the
        metadata file at config.partitionpath/SPHINX_metadata.pkl
    """
    if not os.path.isfile(source_pkl_path):
        logger.error(f"Source pkl not found: {source_pkl_path}")
        sys.exit()

    logger.info(f"Loading existing dataframe from {source_pkl_path}. "
        "This is the one and only time the full historical dataframe "
        "will be loaded in full.")
    df = resume.read_in_df(source_pkl_path)
    logger.info(f"Loaded dataframe with {len(df)} rows.")

    if not os.path.isdir(config.partitionpath):
        os.makedirs(config.partitionpath)

    #### PARTITION BY YEAR-MONTH OF PREDICTION WINDOW START ####
    if "Prediction Window Start" not in df.columns:
        logger.error("Column 'Prediction Window Start' not found; "
            "cannot determine partition boundaries.")
        sys.exit()

    df = df.assign(_partition_key=df["Prediction Window Start"].dt.strftime("%Y-%m"))
    n_partitions = df["_partition_key"].nunique(dropna=True)
    logger.info(f"Splitting into {n_partitions} monthly partitions.")

    for partition_key, group in df.groupby("_partition_key", dropna=True):
        group = group.drop(columns=["_partition_key"])
        valid.write_partition_df(group, dataset_name, partition_key)

    #ROWS WITH NO PREDICTION WINDOW START (SHOULDN'T NORMALLY HAPPEN, BUT
    #DON'T SILENTLY DROP THEM)
    no_key = df[df["_partition_key"].isnull()]
    if not no_key.empty:
        logger.warning(f"{len(no_key)} rows had no 'Prediction Window Start' "
            "and were written to an 'unknown' partition.")
        no_key = no_key.drop(columns=["_partition_key"])
        valid.write_partition_df(no_key, dataset_name, "unknown")

    if skip_index_metadata:
        logger.info("skip_index_metadata=True: partitions written, "
            "index/metadata files left untouched.")
        logger.info("Migration complete.")
        return

    #### BUILD THE DUPLICATE INDEX ####
    logger.info("Building duplicate index from full dataframe (one-time cost).")
    row_hash = duplicates.compute_row_hash(df.drop(columns=["_partition_key"]))
    index_df = df[["Forecast Source"]].assign(RowHash=row_hash.values)
    index_path = os.path.join(config.partitionpath, f"{dataset_name}_index.parquet")
    index_df.to_parquet(index_path, index=False)
    logger.info(f"Wrote duplicate index with {len(index_df)} rows to {index_path}")

    #### BUILD THE METADATA FILE ####
    logger.info("Building metadata (models, energy channels, thresholds).")
    models = resume.identify_unique(df, "Model")
    energy_channels = resume.identify_unique(df, "Energy Channel Key")
    thresholds = resume.identify_thresholds_per_energy_channel(df)

    metadata_path = os.path.join(config.partitionpath, "SPHINX_metadata.pkl")
    resume.write_metadata(metadata_path, models, energy_channels, thresholds)
    logger.info(f"Wrote metadata to {metadata_path}: "
        f"{len(models)} models, {len(energy_channels)} energy channels.")

    logger.info("Migration complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source_pkl_path")
    parser.add_argument("dataset_name", nargs="?", default="SPHINX_evaluated")
    parser.add_argument("--skip-index-metadata", action="store_true",
        help="Only write partitions; do not touch the shared index/metadata files. "
             "Use for any dataset beyond the primary SPHINX_evaluated migration.")
    args = parser.parse_args()

    migrate(args.source_pkl_path, args.dataset_name, args.skip_index_metadata)
