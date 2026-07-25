"""
MATERIALIZATION STEP FOR report.py.

report.py expects a single flat SPHINX_evaluated.pkl. Upstream, validation
now writes monthly Parquet partitions instead of maintaining that flat file
on every run (see validation.py write_partition_df / intuitive_validation).

This script rebuilds the flat pkl from the partitions, using a columnar
Parquet read (pyarrow) rather than pandas-native concatenation, so the
read itself is as efficient as the format allows. It still has to hold
the full historical dataframe in memory once, at the point of calling
to_pickle() -- that requirement comes from report.py's own contract
(pd.read_pickle), not from anything upstream, and can only be removed by
changing report.py to read partitions/DuckDB directly instead.

RUN THIS on whatever cadence report.py actually needs fresh data --
e.g. once per pipeline run, after intuitive_validation completes, NOT
inside intuitive_validation itself.

USAGE:
    python materialize_for_report.py [dataset_name]
    (dataset_name defaults to "SPHINX_evaluated")
"""

import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sphinxval.utils import config
from sphinxval.utils import validation as valid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def materialize(dataset_name="SPHINX_evaluated", write_csv=False):
    """ Rebuild the flat pkl (and optionally csv) that report.py expects,
        from the partitioned Parquet dataset.

    INPUT:

        :dataset_name: (string) name of the partitioned dataset, e.g.
            "SPHINX_evaluated" or "SPHINX_removed"
        :write_csv: (bool) also write the flat csv. Off by default --
            report.py's report() only reads the pkl directly; only turn
            this on if something else depends on SPHINX_evaluated.csv
            specifically.

    OUTPUT:

        Writes config.outpath/pkl/<dataset_name>.pkl (and .csv if
        requested), matching the paths write_df() has always used, so
        report.py needs no changes to find them.
    """
    partition_dir = os.path.join(config.partitionpath, dataset_name)
    if not os.path.isdir(partition_dir):
        logger.error(f"No partition directory found at {partition_dir}. "
            "Has migrate_to_partitions.py been run, and has at least one "
            "validation run written partitions since?")
        sys.exit()

    logger.info(f"Reading all partitions for {dataset_name} from {partition_dir}.")

    # DELIBERATELY USES validation.read_all_partitions (PER-FILE PANDAS
    # READ + CONCAT) RATHER THAN pyarrow.dataset(...).to_table() ACROSS
    # THE WHOLE DIRECTORY. THE LATTER INFERS ONE ARROW SCHEMA PER FILE AND
    # FAILS WITH ArrowNotImplementedError WHEN A COLUMN IS ENTIRELY NULL IN
    # ONE MONTH'S PARTITION (INFERRED AS TYPE `null`) BUT HAS REAL VALUES
    # IN ANOTHER MONTH'S PARTITION (INFERRED AS e.g. `double`) -- SEEN IN
    # PRODUCTION TESTING.
    try:
        df = valid.read_all_partitions(dataset_name)
    except MemoryError:
        logger.error(f"MemoryError reading partitions for {dataset_name}. "
            "Total history has grown large enough that even a per-file read "
            "no longer fits in memory. This is a sign report.py's flat-pkl "
            "contract needs to change, not something this script can "
            "work around.")
        sys.exit(1)

    logger.info(f"Read {len(df)} total rows across all partitions.")

    #CONVERT UNITS COLUMNS BACK FROM STRINGS TO LIVE astropy.units.Unit
    #OBJECTS SO report.py AND DOWNSTREAM CODE SEE THE SAME TYPES THEY
    #ALWAYS HAVE FROM THE ORIGINAL PICKLE-BASED PIPELINE
    df = valid._units_columns_from_string(df)

    #Sort to match the ordering fill_sphinx_df has always produced, so
    #report.py sees the same row order as before this refactor
    sort_cols = [c for c in ["Model", "Energy Channel Key", "Threshold Key",
        "Prediction Window Start", "Forecast Issue Time"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(by=sort_cols, ascending=[True] * len(sort_cols))

    try:
        if write_csv:
            valid.write_df(df, dataset_name)
        else:
            #WRITE PKL ONLY -- SKIP THE REDUNDANT FULL CSV WRITE UNLESS
            #SOMETHING SPECIFICALLY NEEDS IT
            pkl_path = os.path.join(config.outpath, "pkl", dataset_name + ".pkl")
            df.to_pickle(pkl_path)
            logger.debug(f"Wrote {pkl_path}")
    except MemoryError:
        logger.error(f"MemoryError while writing {dataset_name}.pkl. The read "
            "succeeded but serialization pushed memory over the edge. "
            "No partial/corrupt pkl was left behind (pandas writes to a "
            "temp path and only replaces the target on success), so the "
            "previous materialized pkl, if any, is still intact and "
            "report.py can still run against it.")
        sys.exit(1)

    logger.info(f"Materialization of {dataset_name}.pkl complete: {len(df)} rows.")


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "SPHINX_evaluated"
    materialize(name)
