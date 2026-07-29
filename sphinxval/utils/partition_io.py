"""
sphinxval/utils/partition_io.py

Standalone, dependency-light, SELF-CONTAINED functions for reading and
writing sphinxval's partitioned Parquet storage.

THIS FILE IS VENDORED, NOT IMPORTED ACROSS REPOS. sphinxval, pushvivid,
and mailsphinx are separate git repositories.

Instead: this exact file is copied identically into all three repos:
    sphinxval/sphinxval/utils/partition_io.py       (canonical source)
    pushvivid/conversion_code/partition_io.py         (vendored copy)
    mailsphinx/mailsphinx/utils/partition_io.py       (vendored copy)

If you fix a bug or change behavior here, propagate the same change to
all three locations. There is currently no automated sync -- keep the
copies in sync manually, or set up real package distribution
(pip-installable, version-pinned) if this becomes too easy to forget.

This module is intentionally self-contained: only pandas, pyarrow, and
astropy.units (for the small units-string conversion inlined below) are
required -- no dependency on sphinxval's other utils modules at all, so
copying this single file is sufficient; no other file needs to be
vendored alongside it.
"""

import os
import glob
import logging

import pandas as pd
from astropy import units as u

logger = logging.getLogger(__name__)


#REPLICATES units_handler.py's int_flux DEFINITION AND
#convert_units_to_string/convert_string_to_units FUNCTIONS. INLINED HERE
#(RATHER THAN IMPORTED FROM units_handler.py AS A SIBLING MODULE) SO THIS
#FILE IS FULLY SELF-CONTAINED -- REQUIRED SINCE THIS EXACT FILE IS VENDORED
#(IDENTICAL COPIES, NOT IMPORTS) INTO pushvivid AND mailsphinx, WHICH DO
#NOT HAVE units_handler.py AVAILABLE. IF units_handler.py's CONVERSION
#LOGIC EVER CHANGES, UPDATE HERE TOO (AND RE-PROPAGATE THIS FILE TO ALL
#THREE REPOS -- SEE MODULE DOCSTRING).
_int_flux = u.cm**-2 * u.sr**-1 * u.s**-1  # pfu


def _convert_string_to_units(str_units):
    str_units = str_units.replace("*", ".")
    str_units = str_units.replace("^", "")
    if str_units == "pfu":
        return _int_flux
    return u.Unit(str_units)


def _convert_units_to_string(units):
    return str(units)



#COLUMNS THAT HOLD LIVE astropy.units.Unit OBJECTS RATHER THAN STRINGS.
#PARQUET CANNOT SERIALIZE ARBITRARY PYTHON OBJECTS (UNLIKE PICKLE), SO
#THESE MUST BE CONVERTED TO STRINGS BEFORE WRITING AND BACK TO Unit
#OBJECTS AFTER READING.
UNITS_COLUMNS = [
    "Observed SEP Peak Intensity (Onset Peak) Units",
    "Observed SEP Peak Intensity Max (Max Flux) Units",
    "Observed Point Intensity Units",
    "Observed Max Flux in Prediction Window Units",
    "Observed SEP Fluence Units",
    "Observed SEP Fluence Spectrum Units",
    "Predicted Point Intensity Units",
    "Predicted SEP Peak Intensity (Onset Peak) Units",
    "Predicted SEP Peak Intensity Max (Max Flux) Units",
    "Predicted SEP Fluence Units",
    "Predicted SEP Fluence Spectrum Units",
]


def units_columns_to_string(df):
    """ Return a copy of df with UNITS_COLUMNS converted from live
        astropy.units.Unit objects to strings, safe for Parquet.
        None/NaN values are left as-is.
    """
    df = df.copy()
    for col in UNITS_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda val: _convert_units_to_string(val) if val is not None and not pd.isnull(val) else None)
    return df


def units_columns_from_string(df):
    """ Return a copy of df with UNITS_COLUMNS converted back from strings
        to live astropy.units.Unit objects, reversing
        units_columns_to_string.
    """
    df = df.copy()
    for col in UNITS_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda val: _convert_string_to_units(val) if val is not None and not pd.isnull(val) else None)
    return df


def write_partition_df(partitionpath, df, name, partition_key, verbose=True):
    """ Write ONLY new rows (e.g. one run's / one month's worth of data)
        to a partitioned Parquet dataset, instead of rewriting an entire
        cumulative pkl/csv file every run.

        INPUT:

            :partitionpath: (string) root directory for partitioned data
                (e.g. /data/SPHINX/active/partitions)
            :df: (dataframe) NEW rows only for this run
            :name: (string) dataset name, e.g. "SPHINX_evaluated" or
                "SPHINX_removed" -- becomes a subdirectory under
                partitionpath
            :partition_key: (string) identifies this run's partition
                file, e.g. "2025-08" -- must be filesystem-safe

        OUTPUT:

            :filepath: (string) path to the partition file written, or
                None if df was empty
    """
    if df.empty:
        if verbose:
            logger.debug('write_partition_df: df for ' + name
                + ' partition ' + partition_key + ' is empty, skipping write.')
        return None

    partition_dir = os.path.join(partitionpath, name)
    if not os.path.isdir(partition_dir):
        os.makedirs(partition_dir)

    filepath = os.path.join(partition_dir, name + '_' + partition_key + '.parquet')

    #CONVERT astropy.units.Unit OBJECTS TO STRINGS -- PARQUET CANNOT
    #SERIALIZE ARBITRARY PYTHON OBJECTS THE WAY PICKLE CAN
    parquet_safe_df = units_columns_to_string(df)
    parquet_safe_df.to_parquet(filepath, index=False)
    if verbose:
        logger.debug('Wrote partition ' + filepath)

    return filepath


def _read_table_safe(filepath, columns=None):
    """ Read one Parquet file via pyarrow directly (not pd.read_parquet's
        to_pandas_kwargs passthrough, which is not consistently supported
        across pandas versions -- seen in production: "TypeError:
        read_table() got an unexpected keyword argument 'to_pandas_kwargs'"),
        and restore proper datetime semantics.

        timestamp_as_object=True avoids an out-of-nanosecond-range crash
        (pandas' default coerces Arrow timestamps into numpy
        datetime64[ns], which only represents dates roughly 1677-2262;
        sentinel/placeholder values outside that range overflow the
        cast), but it also turns genuinely missing values from pd.NaT
        into plain Python None, which raises TypeError in arithmetic
        where NaT would propagate safely as NaN. Restoring via
        pd.to_datetime(errors='coerce') on every originally-timestamp
        column fixes both: in-range values and missing values get
        correct NaT-safe behavior back, and a genuinely out-of-range
        sentinel either gets preserved (if this pandas version supports
        non-nanosecond datetime64 resolution) or coerced to NaT (older
        pandas) -- either outcome is arithmetic-safe.
    """
    import pyarrow.parquet as pq
    import pyarrow.types as patypes

    table = pq.read_table(filepath, columns=columns)
    frame = table.to_pandas(timestamp_as_object=True)
    for field in table.schema:
        if patypes.is_timestamp(field.type):
            frame[field.name] = pd.to_datetime(frame[field.name], errors='coerce')
    return frame, table.schema


def read_all_partitions(partitionpath, name, columns=None):
    """ Read every partition file for a dataset and concatenate them into
        a single pandas dataframe.

        Deliberately reads each file individually via pandas and
        concatenates, rather than using pyarrow.dataset(...).to_table()
        across the whole directory. That approach infers one Arrow
        schema per file, and a column that is entirely null/NaN in one
        month's partition gets inferred as Arrow type `null` rather than
        the column's real dtype (e.g. `double`); when pyarrow.dataset
        tries to unify schemas across files it cannot cast a real value
        in one file into the `null`-typed schema inferred from another,
        and raises ArrowNotImplementedError ("Unsupported cast from
        double to null using function cast_null" or similar) -- seen in
        production. Reading each file individually avoids this entirely.

    INPUT:

        :partitionpath: (string) root directory for partitioned data
        :name: (string) partitioned dataset name, e.g. "SPHINX_evaluated"
        :columns: (list of string or None) columns to read from each
            partition file (column pruning still applies per-file)

    OUTPUT:

        :df: (pandas DataFrame) concatenation of all partitions, in
            filename-sorted order. Empty DataFrame if the partition
            directory doesn't exist yet or contains no partition files.
    """
    partition_dir = os.path.join(partitionpath, name)
    if not os.path.isdir(partition_dir):
        return pd.DataFrame()

    partition_files = sorted(glob.glob(os.path.join(partition_dir, "*.parquet")))
    if not partition_files:
        return pd.DataFrame()

    frames = []
    for f in partition_files:
        frame, _ = _read_table_safe(f, columns=columns)
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def read_partitions_for_date_range(partitionpath, name, date_column,
    start_date=None, end_date=None, columns=None):
    """ Like read_all_partitions, but skips partition files whose data
        falls entirely outside [start_date, end_date] -- without ever
        reading those files' full width.

        WHY THIS EXISTS: partition files are named by the RUN TIMESTAMP
        that wrote them, not by the date range of the data inside them,
        so filtering by filename isn't reliable. Consumers that only
        need a short date range (e.g. pushvivid, typically pushing a
        single week or month) would otherwise have to read ALL of
        history just to filter down afterward.

        Approach: for each partition file, first read ONLY date_column
        (cheap) to check whether ANY row falls in the requested range.
        Only if so, read that file's full requested width.

    INPUT:

        :partitionpath: (string) root directory for partitioned data
        :name: (string) partitioned dataset name, e.g. "SPHINX_evaluated"
        :date_column: (string) column to filter on, e.g.
            "Forecast Issue Time". Must be present in every partition
            file regardless of what `columns` requests.
        :start_date: (string, datetime, or None) inclusive lower bound
        :end_date: (string, datetime, or None) inclusive upper bound
        :columns: (list of string or None) columns to read from each
            included file's full read. date_column is always included
            even if omitted, and dropped afterward if it wasn't
            explicitly requested.

    OUTPUT:

        :df: (pandas DataFrame) concatenation of only the partitions
            whose data overlaps the requested range. If the partition
            directory doesn't exist or contains no files at all, returns
            a bare empty DataFrame (no columns -- there's no schema to
            derive one from). If files exist but none have data in the
            requested range, returns an empty DataFrame with the correct
            columns (derived from an existing file's schema) and zero
            rows -- callers can safely index expected columns either way.
    """
    partition_dir = os.path.join(partitionpath, name)
    if not os.path.isdir(partition_dir):
        return pd.DataFrame()

    partition_files = sorted(glob.glob(os.path.join(partition_dir, "*.parquet")))
    if not partition_files:
        return pd.DataFrame()

    start_ts = pd.to_datetime(start_date) if start_date is not None else None
    end_ts = pd.to_datetime(end_date) if end_date is not None else None

    date_col_requested = columns is None or date_column in columns
    full_columns = None
    if columns is not None:
        full_columns = list(columns)
        if date_column not in full_columns:
            full_columns.append(date_column)

    frames = []
    for f in partition_files:
        try:
            probe, _ = _read_table_safe(f, columns=[date_column])
        except Exception:
            logger.warning(f"read_partitions_for_date_range: could not read "
                f"{date_column} from {f} to check date range; skipping this "
                f"partition file entirely.")
            continue

        dates = probe[date_column].dropna()
        if dates.empty:
            continue

        file_min, file_max = dates.min(), dates.max()
        if start_ts is not None and file_max < start_ts:
            continue
        if end_ts is not None and file_min > end_ts:
            continue

        frame, _ = _read_table_safe(f, columns=full_columns)

        mask = pd.Series(True, index=frame.index)
        if start_ts is not None:
            mask &= frame[date_column] >= start_ts
        if end_ts is not None:
            mask &= frame[date_column] <= end_ts
        frame = frame.loc[mask]

        if not date_col_requested:
            frame = frame.drop(columns=[date_column])

        if not frame.empty:
            frames.append(frame)

    if not frames:
        # NO PARTITION FILE HAD ANY DATA IN THE REQUESTED RANGE. RETURN AN
        # EMPTY-BUT-CORRECTLY-COLUMNED DATAFRAME, NOT A BARE pd.DataFrame()
        # (ZERO COLUMNS). SEEN IN PRODUCTION: A BARE EMPTY DataFrame CAUSED
        # DOWNSTREAM CODE (e.g. df['Forecast Issue Time'] >= date_start,
        # OR df[wanted_keys]) TO RAISE KeyError, WHERE THE OLD PICKLE-BASED
        # READ PATH ALWAYS RETURNED A FULL-COLUMN DATAFRAME REGARDLESS OF
        # HOW MANY ROWS MATCHED A SUBSEQUENT DATE FILTER. DERIVE THE
        # EXPECTED COLUMNS FROM ANY AVAILABLE PARTITION FILE'S SCHEMA (NOT
        # FROM DATA, SINCE NONE MATCHED) SO THE EMPTY RESULT IS STILL
        # SHAPED CORRECTLY FOR CALLERS THAT EXPECT PARTICULAR COLUMNS TO
        # EXIST EVEN WHEN THERE ARE ZERO ROWS.
        try:
            empty_frame, _ = _read_table_safe(partition_files[0], columns=full_columns)
            empty_frame = empty_frame.iloc[0:0]
            if not date_col_requested:
                empty_frame = empty_frame.drop(columns=[date_column])
            return empty_frame
        except Exception:
            logger.warning("read_partitions_for_date_range: no data in range "
                "and could not derive column structure from an existing "
                "partition file either; returning a bare empty DataFrame.")
            return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)
