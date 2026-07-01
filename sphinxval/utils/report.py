"""SPHINX Markdown/HTML report generator.

Generates per-model validation reports in Markdown and HTML formats from
SPHINX output pkl files.  The public entry point is ``report()``.
"""

import base64
import datetime
import io
import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

import bs4
import markdown
import numpy as np
import pandas as pd
import PyPDF2 as pdf

from . import config
from . import make_index

# SUPPRESS CHAINED-ASSIGNMENT WARNINGS FROM PANDAS; THE ASSIGNMENTS IN
# build_info_events_table ARE INTENTIONAL (OPERATING ON AN EXPLICIT COPY)
pd.set_option('mode.chained_assignment', None)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# MODULE-LEVEL CONSTANTS
# -----------------------------------------------------------------------

# METRICS THAT LIVE IN [0, 1] — FORMATTED TO 2 DECIMAL PLACES
_ZERO_TO_ONE_METRICS = frozenset({
    'Percent Correct', 'Hit rate', 'False Alarm Rate',
    'Frequency of Misses', 'Probability of Correct Negatives',
    'Frequency of Hits', 'False Alarm Ratio', 'Detection Failure Ratio',
    'Frequency of Correct Negatives', 'Threat Score', 'Brier Score',
    'Pearson Correlation Coefficient (linear)',
    'Pearson Correlation Coefficient (log)',
})

# METRICS THAT ARE ALWAYS INTEGERS
_INT_METRICS = frozenset({
    'Hits (TP)', 'Misses (FN)', 'False Alarms (FP)', 'Correct Negatives (TN)',
})

# APPENDAGE VARIANTS SPHINX PRODUCES FOR EACH METRIC TYPE
_APPENDAGES = ['', '_First', '_Last', '_Mean', '_Max']

# SHARED METRICS DESCRIPTION STRINGS
_DESC_FLUX = (
    'Correlation coefficients and regression lines indicate association.<br>'
    'Metrics involving error indicate bias. Positive values indicate model '
    'overprediction and negative values indicate model underprediction.<br>'
    'Metrics involving absolute error or squared error indicate accuracy.\n'
)
_DESC_TIME = (
    'Metrics for Predicted Time - Observed Time are in hours.<br>'
    'Negative values indicate predicted time is earlier than observed.<br>'
    'Positive values indicate predicted time is later than observed.\n'
)


# -----------------------------------------------------------------------
# SECTION DEFINITION DATACLASS
# -----------------------------------------------------------------------

@dataclass(frozen=True)
class SectionDef:
    """Describes one validated quantity section in the SPHINX report."""
    # FILENAME PREFIX USED TO DETECT SELECTION FILES IN output_dir
    file_prefix: str
    # INTERNAL TAG USED IN OUTPUT FILENAMES (E.G. 'all_clear')
    tag: str
    # HUMAN-READABLE SECTION TITLE
    title: str
    # COLUMN NAME WHERE METRICS START IN THE METRICS PKL
    metric_label_start: str
    # DESCRIPTION TEXT SHOWN ABOVE THE METRICS TABLE
    metrics_description: str
    # KEY INTO validation_reference_flag_dict
    ref_flag: str
    # REFERENCE CSV FILENAMES (RELATIVE TO config.referencepath)
    ref_csv_metrics: str
    ref_csv_skills: Optional[str] = None
    ref_csv_plots: Optional[str] = None
    # COLUMNS TO SKIP IN THE METRICS TABLE
    skip_labels: List[str] = field(default_factory=list)
    # SECTION TYPE FLAGS
    is_all_clear: bool = False
    is_awt: bool = False
    # OVERRIDE FOR THE METRICS FILE TAG WHEN IT DIFFERS FROM self.tag
    metrics_tag_override: Optional[str] = None

    @property
    def metrics_tag(self):
        return self.metrics_tag_override or self.tag


_SECTION_DEFS = [
    SectionDef(
        file_prefix='all_clear_selections_',
        tag='all_clear', title='All Clear',
        metric_label_start='Mean Error (pred - obs)',
        metrics_description='',
        ref_flag='All Clear',
        ref_csv_metrics='validation_reference_sheet_contingency_metrics.csv',
        ref_csv_skills='validation_reference_sheet_contingency_skills.csv',
        ref_csv_plots='validation_reference_sheet_contingency_plots.csv',
        is_all_clear=True,
    ),
    SectionDef(
        file_prefix='awt_selections_',
        tag='awt', title='Advanced Warning Time',
        metric_label_start='Mean AWT for Predicted SEP All Clear to Observed SEP Threshold Crossing Time',
        metrics_description='N/A',
        ref_flag='AWT',
        ref_csv_metrics='validation_reference_sheet_awt_metrics.csv',
        is_awt=True,
    ),
    SectionDef(
        file_prefix='duration_selections_',
        tag='duration', title='Duration',
        metric_label_start='Mean Error (pred - obs)',
        metrics_description=(
            'Duration is calculated in hours.<br> Metrics involving error indicate bias. '
            'Positive values indicate model overprediction and negative values indicate '
            'model underprediction.<br>Metrics involving absolute error indicate accuracy.\n'
        ),
        ref_flag='Time',
        ref_csv_metrics='validation_reference_sheet_time_metrics.csv',
    ),
    SectionDef(
        file_prefix='peak_intensity_selections_',
        tag='peak_intensity', title='Peak Intensity (Onset Peak)',
        metric_label_start='Linear Regression Slope',
        metrics_description=_DESC_FLUX,
        ref_flag='Flux',
        ref_csv_metrics='validation_reference_sheet_flux_metrics.csv',
        ref_csv_plots='validation_reference_sheet_flux_plots.csv',
    ),
    SectionDef(
        file_prefix='peak_intensity_max_selections_',
        tag='peak_intensity_max', title='Peak Intensity Max (Max Flux)',
        metric_label_start='Linear Regression Slope',
        metrics_description=_DESC_FLUX,
        ref_flag='Flux',
        ref_csv_metrics='validation_reference_sheet_flux_metrics.csv',
        ref_csv_plots='validation_reference_sheet_flux_plots.csv',
    ),
    SectionDef(
        file_prefix='peak_intensity_time_selections_',
        tag='peak_intensity_time', title='Peak Intensity (Onset Peak) Time',
        metric_label_start='Mean Error (pred - obs)',
        metrics_description=_DESC_TIME,
        ref_flag='Time',
        ref_csv_metrics='validation_reference_sheet_time_metrics.csv',
    ),
    SectionDef(
        file_prefix='peak_intensity_max_time_selections_',
        tag='peak_intensity_max_time', title='Peak Intensity Max (Max Flux) Time',
        metric_label_start='Mean Error (pred - obs)',
        metrics_description=_DESC_TIME,
        ref_flag='Time',
        ref_csv_metrics='validation_reference_sheet_time_metrics.csv',
    ),
    SectionDef(
        file_prefix='threshold_crossing_time_selections_',
        tag='threshold_crossing_time', title='Threshold Crossing Time',
        metric_label_start='Mean Error (pred - obs)',
        metrics_description=_DESC_TIME,
        ref_flag='Time',
        ref_csv_metrics='validation_reference_sheet_time_metrics.csv',
        metrics_tag_override='threshold_crossing',
    ),
    SectionDef(
        file_prefix='fluence_selections_',
        tag='fluence', title='Fluence',
        metric_label_start='Linear Regression Slope',
        metrics_description=_DESC_FLUX,
        ref_flag='Flux',
        ref_csv_metrics='validation_reference_sheet_flux_metrics.csv',
        ref_csv_plots='validation_reference_sheet_flux_plots.csv',
    ),
    SectionDef(
        file_prefix='max_flux_in_pred_win_selections_',
        tag='max_flux_in_pred_win', title='Max Flux in Prediction Window',
        metric_label_start='Linear Regression Slope',
        metrics_description=_DESC_FLUX,
        ref_flag='Flux',
        ref_csv_metrics='validation_reference_sheet_flux_metrics.csv',
        ref_csv_plots='validation_reference_sheet_flux_plots.csv',
    ),
    SectionDef(
        file_prefix='probability_selections_',
        tag='probability', title='Probability',
        metric_label_start='Brier Score',
        metrics_description='',
        ref_flag='Probability',
        ref_csv_metrics='validation_reference_sheet_probability_metrics.csv',
        ref_csv_skills='validation_reference_sheet_probability_skills.csv',
        ref_csv_plots='validation_reference_sheet_probability_plots.csv',
    ),
    SectionDef(
        file_prefix='start_time_selections_',
        tag='start_time', title='Start Time',
        metric_label_start='Mean Error (pred - obs)',
        metrics_description=_DESC_TIME,
        ref_flag='Time',
        ref_csv_metrics='validation_reference_sheet_time_metrics.csv',
    ),
    SectionDef(
        file_prefix='end_time_selections_',
        tag='end_time', title='End Time',
        metric_label_start='Mean Error (pred - obs)',
        metrics_description=_DESC_TIME,
        ref_flag='Time',
        ref_csv_metrics='validation_reference_sheet_time_metrics.csv',
    ),
    SectionDef(
        file_prefix='time_profile_selections_',
        tag='time_profile', title='Time Profile',
        metric_label_start='Linear Regression Slope',
        metrics_description=(
            'Correlation plots are created for each predicted time profile and may be '
            'viewed in the output/plots directory.<br>Metrics are calculated from '
            'overlapping portions of predicted and observed time profiles, highlighted '
            'in red and orange in the Time Profile plots.<br>Metrics involving error '
            'indicate bias. Positive values indicate model overprediction and negative '
            'values indicate model underprediction.<br>Metrics involving absolute error '
            'or squared error indicate accuracy.\n'
        ),
        ref_flag='Flux',
        ref_csv_metrics='validation_reference_sheet_flux_metrics.csv',
        ref_csv_plots='validation_reference_sheet_flux_plots.csv',
        skip_labels=['Time Profile Selection Plot'],
    ),
]


# -----------------------------------------------------------------------
# CACHES — AVOID REDUNDANT DISK I/O ACROSS MODELS AND APPENDAGES
# -----------------------------------------------------------------------

# PKL CACHE: KEYED BY ABSOLUTE FILE PATH → LOADED DATAFRAME.
# CLEARED AT THE START OF EACH report() CALL.
_pkl_cache: dict = {}

# REFERENCE CSV CACHE: KEYED BY ABSOLUTE FILE PATH → RENDERED MARKDOWN.
# REFERENCE SHEETS NEVER CHANGE BETWEEN MODELS.
_ref_csv_cache: dict = {}


def _load_pkl(path: str) -> pd.DataFrame:
    """Load a selections file (.pkl or .csv), returning the cached result
    on subsequent calls. .csv files are loaded with pandas.read_csv()."""
    abs_path = os.path.abspath(path)
    if abs_path not in _pkl_cache:
        if abs_path.endswith('.csv'):
            _pkl_cache[abs_path] = pd.read_csv(abs_path)
        else:
            _pkl_cache[abs_path] = pd.read_pickle(abs_path)
    return _pkl_cache[abs_path]


def _ref_csv_markdown(path: str) -> str:
    """Load a reference CSV and render it as Markdown, caching the result."""
    abs_path = os.path.abspath(path)
    if abs_path not in _ref_csv_cache:
        data = pd.read_csv(abs_path, skiprows=1)
        _ref_csv_cache[abs_path] = '\n' + data.to_markdown(index=False) + '\n'
    return _ref_csv_cache[abs_path]


def _ref_path(filename: Optional[str]) -> Optional[str]:
    """Resolve a reference CSV filename to its full path, or None."""
    return os.path.join(config.referencepath, filename) if filename else None


# -----------------------------------------------------------------------
# FORMATTING HELPERS
# -----------------------------------------------------------------------

def _format_significant(value: float) -> str:
    """Format a float to 2-5 significant decimal places, stopping once a
    non-zero digit appears.  Falls back to scientific notation after 5
    decimal places."""
    for n in range(2, 6):
        formatted = f'{value:.{n}f}'
        if any(d in formatted for d in '123456789'):
            return formatted
    return formatted


def _formatter_for_metric(metric_name: str):
    """Return an appropriate single-argument formatting callable for a
    named metric."""
    if metric_name in _ZERO_TO_ONE_METRICS or 'Skill Score' in metric_name or 'Skill Statistic' in metric_name:
        return lambda v: f'{v:.2f}'
    if metric_name in _INT_METRICS:
        return lambda v: str(int(round(v)))
    return _format_significant


def _format_value(label: str, value) -> str:
    """Format a single metric value, falling back to str() for
    non-numeric values (e.g. date strings like 'Predicted SEP Events')."""
    if isinstance(value, str):
        return value
    if not isinstance(value, (int, float, np.integer, np.floating)):
        return str(value)
    return _formatter_for_metric(label)(value)


# -----------------------------------------------------------------------
# MARKDOWN TABLE BUILDERS
# -----------------------------------------------------------------------

def make_markdown_table(column_1: str, column_2: str, dataframe: pd.DataFrame, width: int = 50, uncertainties: dict = None) -> str:
    """Render a Markdown table from *dataframe*.

    *dataframe* must have one column; its index provides the row labels.
    If *uncertainties* is provided (a dict mapping label -> uncertainty
    value or None), a third "Uncertainty" column is added, showing '-'
    for metrics with no corresponding uncertainty value.
    """
    rows = list(dataframe.index)
    raw_values = list(dataframe.to_numpy().flatten())

    # CONVERT RATIOS TO PERCENTAGES WHERE APPROPRIATE
    for i, label in enumerate(rows):
        if label in config.in_percent:
            raw_values[i] = raw_values[i] * 100.0
            rows[i] = label + ' [%]'

    buf = io.StringIO()
    if uncertainties is not None:
        buf.write(f'| {column_1:<{width - 1}}| {column_2:<{width - 1}}| {"Uncertainty":<{width - 1}}|\n')
        buf.write(f'|:{"-" * (width - 1)}:|:{"-" * (width - 1)}:|:{"-" * (width - 1)}:|\n')
    else:
        buf.write(f'| {column_1:<{width - 1}}| {column_2:<{width - 1}}|\n')
        buf.write(f'|:{"-" * (width - 1)}:|:{"-" * (width - 1)}:|\n')

    for orig_label, label, value in zip(dataframe.index, rows, raw_values):
        safe_label = label.replace('|', r'\|')
        if value is None or (isinstance(value, float) and np.isnan(value)):
            fmt_value = 'NaN'
        else:
            fmt_value = _format_value(label, value)
        if uncertainties is not None:
            unc_value = uncertainties.get(orig_label)
            if unc_value is None or (isinstance(unc_value, float) and np.isnan(unc_value)):
                fmt_unc = '-'
            else:
                fmt_unc = _format_value(label, unc_value)
            buf.write(f'| {safe_label:<{width}}| {fmt_value:<{width}}| {fmt_unc:<{width}}|\n')
        else:
            buf.write(f'| {safe_label:<{width}}| {fmt_value:<{width}}|\n')
    return buf.getvalue()


def transpose_markdown(text: str) -> str:
    """Transpose a wide Markdown table into a two-column (Metric | Value) table."""
    lines = text.strip().split('\n')
    labels = [c.strip() for c in lines[0].split('|') if c.strip()]
    values = [c.strip() for c in lines[2].split('|') if c.strip()]
    buf = io.StringIO()
    buf.write('| Metric | Value |\n')
    buf.write(f'|:{"-" * 48}:|:{"-" * 48}:|\n')
    for label, value in zip(labels, values):
        buf.write(f'| {label} | {value} |\n')
    return buf.getvalue().rstrip('|\n') + '|\n'


# -----------------------------------------------------------------------
# COLLAPSIBLE HTML/MARKDOWN SEGMENTS
# -----------------------------------------------------------------------

def add_collapsible_segment(header: str, text: str) -> str:
    """Wrap *text* in a <details>/<summary> block."""
    return f'<details>\n<summary>{header}</summary>\n\n{text}\n</details>\n'


def add_collapsible_segment_start(header: str, text: str = '') -> str:
    return f'<details>\n<summary>{header}</summary>\n\n{text}\n<blockquote>\n\n'


def add_collapsible_segment_end() -> str:
    return '</blockquote>\n</details>\n\n'


# -----------------------------------------------------------------------
# SMALL HTML PRIMITIVES
# -----------------------------------------------------------------------

def add_text_color(text: str, color: str) -> str:
    return f'<span style="color:{color}">{text}</span>'


def add_index_link() -> str:
    return '<a href="index.html">&#8592; Other Reports</a>\n'


def add_title(model: str) -> str:
    return f'<h1>{model} Validation Report</h1>\n'


def add_script(text: str) -> str:
    return '''\
<script>
function openTab(evt, tabName) {
  var i, tabcontent, tablinks;
  tabcontent = document.getElementsByClassName("tabcontent");
  for (i = 0; i < tabcontent.length; i++) { tabcontent[i].style.display = "none"; }
  tablinks = document.getElementsByClassName("tablinks");
  for (i = 0; i < tablinks.length; i++) {
    tablinks[i].className = tablinks[i].className.replace(" active", "");
  }
  document.getElementById(tabName).style.display = "block";
  evt.currentTarget.className += " active";
}
</script>\n\n''' + text


def add_style(text: str) -> str:
    return '''\
<style>
table, th, td { border: 1px solid black; border-collapse: collapse; }
html * { font-size: 16px; line-height: 1.25; color: #000000; font-family: Arial, sans-serif; }
.tab { overflow: hidden; border: 1px solid #ccc; background-color: #f1f1f1; }
.tab button { background-color: inherit; float: left; border: none; outline: none;
              cursor: pointer; padding: 14px 16px; transition: 0.3s; }
.tab button:hover { background-color: #ddd; }
.tab button.active { background-color: #ccc; }
.tabcontent { display: none; padding: 6px 12px; border: 1px solid #ccc; border-top: none; }
.red { background-color: #fad5d2; }
.green { background-color: #89d99e; }
</style>\n\n''' + text


def get_html_report_preamble(model: str) -> str:
    return add_script('') + add_style('') + add_index_link() + add_title(model)


def add_tab(appendage: str, markdown_text: str, model: str) -> str:
    tab_id = appendage.replace('_', '') if appendage else 'All'
    display = 'style="display:block"' if not appendage else ''
    return (
        f'<div id="{tab_id}" class="tabcontent" {display}>\n'
        f'    <h3>{tab_id}</h3>\n'
        f'    {convert_markdown_to_html(markdown_text, model + "..." + tab_id, False)}\n'
        f'</div>\n'
    )


# -----------------------------------------------------------------------
# THRESHOLD / ENERGY-CHANNEL PARSING
# -----------------------------------------------------------------------

def _parse_energy_channel(raw: str):
    """Return a human-readable energy channel string and mismatch suffix."""
    if raw.count('MeV') > 1:
        # MISMATCH CASE: TWO CHANNELS JOINED BY '_'
        segments = []
        for part in raw.split('_'):
            emin = part.split('min.')[1].split('.max.')[0]
            emax = part.split('.max.')[1].split('.units.')[0]
            segments.append(f'> {emin} MeV' if float(emax) < 0 else f'{emin} < E < {emax} MeV')
        return ' , '.join(segments), '_mm'
    emin = raw.split('min.')[1].split('.max.')[0]
    emax = raw.split('.max.')[1].split('.units.')[0]
    label = f'> {emin} MeV' if float(emax) < 0 else f'{emin} < E < {emax} MeV'
    return label, ''


def build_threshold_string(data: pd.DataFrame, k: int):
    row = data.iloc[k]
    energy_threshold, mismatch_str = _parse_energy_channel(row['Energy Channel'])
    obs_threshold = row['Threshold'].split('.units.')[0].split('threshold.')[1] + ' pfu'
    pred_threshold = row['Prediction Threshold'].split('.units.')[0].split('threshold.')[1] + ' pfu'
    threshold_string = (
        f'* Energy Channel: {energy_threshold}\n'
        f'* Observation Threshold: {obs_threshold}\n'
        f'* Predictions Threshold: {pred_threshold}\n'
    )
    return threshold_string, energy_threshold, obs_threshold, pred_threshold, mismatch_str


# -----------------------------------------------------------------------
# CONTINGENCY / SKILL SCORE TABLES
# -----------------------------------------------------------------------

def build_contingency_table(yes_yes, yes_no, no_no, no_yes) -> str:
    return (
        '| |Observed Yes|Observed No|\n'
        '|----|:----:|:----:|\n'
        f'|Predicted Yes|{yes_yes}|{yes_no}|\n'
        f'|Predicted No|{no_yes}|{no_no}|\n'
    )


def build_skill_score_table(labels, values) -> str:
    """Build a Metric | Value | Uncertainty table from parallel labels and
    values lists, where some labels may be '{Metric} Uncertainty' siblings
    of a preceding metric label."""
    uncertainty_suffix = ' Uncertainty'
    pairs = list(zip(labels, values))
    uncertainties = {}
    main_pairs = []
    for label, value in pairs:
        if label.endswith(uncertainty_suffix):
            uncertainties[label[:-len(uncertainty_suffix)]] = value
        else:
            main_pairs.append((label, value))
    main_labels = [label for label, _ in main_pairs]
    main_values = [value for _, value in main_pairs]
    subset_df = pd.DataFrame(main_values, index=main_labels)
    return '\n' + make_markdown_table('Metric', 'Value', subset_df, uncertainties=uncertainties) + '\n'


# -----------------------------------------------------------------------
# SELECTIONS / INFO-EVENTS TABLE
# -----------------------------------------------------------------------

def append_subset_list(selections_filename: str, subset_list: list, include_after: str, exclusion_pattern: Optional[str] = None) -> list:
    """Extend *subset_list* with column names from the CSV counterpart of
    *selections_filename* that appear after *include_after*."""
    if selections_filename.endswith('.csv'):
        csv_path = selections_filename
    else:
        csv_path = selections_filename.replace('.pkl', '.csv')
    if not os.path.exists(csv_path):
        logger.warning('CSV counterpart not found: %s', csv_path)
        return subset_list
    try:
        header = pd.read_csv(csv_path, nrows=0)
        columns = list(header.columns)
        if columns and columns[0].startswith('Unnamed'):
            columns = columns[1:]
        include = False
        for col in columns:
            col = col.rstrip('\n').rstrip('\\')
            if include and (exclusion_pattern is None or exclusion_pattern not in col):
                subset_list.append(col)
            if col == include_after:
                include = True
    except Exception:
        logger.warning('Could not parse CSV header: %s', csv_path, exc_info=True)
    return subset_list


def build_info_events_table(filename: str, sphinx_dataframe: pd.DataFrame, subset_list: list, subset_replacement_dict: dict, selections_limit: int = 1000):
    data = _load_pkl(filename)
    subset = data[subset_list].copy()
    subset.insert(0, 'Observatory', '')
    selection_index = list(data.index)
    subset['Observatory'] = sphinx_dataframe.loc[selection_index, 'Observatory'].to_list()
    subset['Observed SEP Start Time'] = sphinx_dataframe.loc[selection_index, 'Observed SEP Start Time'].to_list()
    subset['Observed SEP End Time'] = sphinx_dataframe.loc[selection_index, 'Observed SEP End Time'].to_list()
    if len(subset) > selections_limit:
        subset = subset.iloc[:selections_limit]
        limit_message = (
            f'This list has been truncated to the first {selections_limit} entries. '
            f'See {filename} for full list.\n'
        )
    else:
        limit_message = ''
    output = '\n' + subset.to_markdown(index=False) + '\n'
    return output, len(data), limit_message, subset


def build_info_string_header(value, limit_message: str, selections_filename: str) -> str:
    return (
        f'Instruments and observed values used in validation.<br>'
        f'Extracted from: {selections_filename}<br>\n'
        f'N = {value}<br>\n'
        f'{limit_message}'
    )


# -----------------------------------------------------------------------
# PLOT HELPERS
# -----------------------------------------------------------------------

def get_plot_type(plot_string: str) -> str:
    if 'Time_Profile' in plot_string:
        return 'Time Profile'
    if 'Correlation' in plot_string:
        return 'Correlation'
    if 'ROC' in plot_string:
        return 'ROC Curve'
    return 'None'


def _plot_path(plot_string_: str, relative_path_plots: bool) -> str:
    if relative_path_plots:
        return os.path.relpath(plot_string_, os.path.basename(config.reportpath))
    return os.path.abspath(plot_string_)


def _append_plot(plot_string_: str, plot_string_list: list, plot_file_string_list: list, relative_path_plots: bool):
    if not plot_string_:
        plot_string_list.append('No image files found.\n\n')
        plot_file_string_list.append('')
    else:
        path = _plot_path(plot_string_, relative_path_plots)
        file_path = os.path.relpath(plot_string_, '.') if relative_path_plots else path
        plot_string_list.append(f'![]({path})\n\n')
        plot_file_string_list.append(file_path)
    return plot_string_list, plot_file_string_list


def build_plot_string_list(data: pd.DataFrame, current_index: int, relative_path_plots: bool):
    plot_string_list = []
    plot_file_string_list = []
    row = data.iloc[current_index]
    if 'Scatter Plot' in data.columns:
        plot_string_list, plot_file_string_list = _append_plot(
            row.get('Scatter Plot', ''), plot_string_list, plot_file_string_list, relative_path_plots)
    if 'ROC Curve Plot' in data.columns:
        plot_string_list, plot_file_string_list = _append_plot(
            row.get('ROC Curve Plot', ''), plot_string_list, plot_file_string_list, relative_path_plots)
    if 'Time Profile Selection Plot' in data.columns:
        raw = row.get('Time Profile Selection Plot', '')
        if raw:
            for part in raw.split(';'):
                path = _plot_path(part, relative_path_plots)
                file_path = os.path.relpath(part, '.') if relative_path_plots else path
                plot_string_list.append(f'![]({path})\n\n')
                plot_file_string_list.append(file_path)
    return plot_string_list, plot_file_string_list


def plot_subsection(plot_string_list: list, subset=None) -> str:
    """Build collapsible plot segments, labelling Time Profile plots with
    their observed event window when available."""
    buf = io.StringIO()
    plot_counter = 1
    last_plot_type = ''
    for plot_string in plot_string_list:
        plot_type = get_plot_type(plot_string)
        plot_counter = 1 if plot_type != last_plot_type else plot_counter + 1
        appendage = ''
        if plot_type == 'Time Profile':
            try:
                start = subset['Observed SEP Start Time'].iloc[0].isoformat()
                end = subset['Observed SEP End Time'].iloc[0].isoformat()
                appendage = f' for event observed {start} -- {end}'
            except (IndexError, AttributeError, TypeError):
                pass
        buf.write(add_collapsible_segment(f'Plot: {plot_type} {plot_counter}{appendage}', plot_string))
        last_plot_type = plot_type
    return buf.getvalue()


# -----------------------------------------------------------------------
# METRICS TABLE BUILDER
# -----------------------------------------------------------------------

def build_metrics_table(data: pd.DataFrame, current_index: int, metric_index_start: int, skip_label_list: list, relative_path_plots: bool):
    column_labels = list(data.columns)
    subset = data.iloc[current_index, metric_index_start:]
    subset_df = pd.DataFrame(subset, index=column_labels[metric_index_start:])

    # SEPARATE OUT "{Metric} Uncertainty" ROWS INTO A LOOKUP DICT, KEYED
    # BY THE BASE METRIC NAME (WITHOUT THE " Uncertainty" SUFFIX), AND
    # DROP THEM FROM THE MAIN METRIC ROWS SO THEY DON'T APPEAR TWICE.
    uncertainty_suffix = ' Uncertainty'
    uncertainties = {}
    uncertainty_labels = [label for label in subset_df.index if label.endswith(uncertainty_suffix)]
    for label in uncertainty_labels:
        base_label = label[:-len(uncertainty_suffix)]
        uncertainties[base_label] = subset_df.loc[label].iloc[0]
        subset_df = subset_df.drop(label, axis=0)

    for label in skip_label_list:
        if label in subset_df.index:
            subset_df = subset_df.drop(label, axis=0)

    metrics_table_string = '\n' + make_markdown_table('Metric', 'Value', subset_df, uncertainties=uncertainties) + '\n'
    plot_string_list, plot_file_string_list = build_plot_string_list(data, current_index, relative_path_plots)
    return metrics_table_string, plot_string_list, plot_file_string_list


# -----------------------------------------------------------------------
# SECTION BUILDERS
# -----------------------------------------------------------------------

def _build_selections_info(selections_filename: str, sphinx_dataframe: pd.DataFrame) -> tuple:
    """Load a selections pkl and build the info string and subset for a section row."""
    subset_list = ['Prediction Window Start', 'Prediction Window End']
    subset_list = append_subset_list(selections_filename, subset_list, 'Prediction Window End', 'Units')
    info_str_, n_events, limit_message, subset = build_info_events_table(
        selections_filename, sphinx_dataframe, subset_list, {})
    info_str = build_info_string_header(n_events, limit_message, selections_filename) + info_str_
    return info_str, n_events, subset


def _resolve_selections_filename(output_dir: str, basename: str) -> str:
    """Return the path to a selections file, preferring .pkl over .csv.

    *basename* should be the filename WITHOUT extension (e.g.
    'all_clear_selections_MAG4_...'). Tries .pkl first; if that doesn't
    exist, falls back to .csv. Returns the .pkl path regardless if
    neither exists (so the caller gets a consistent missing-file error).
    """
    pkl_path = os.path.join(output_dir, basename + '.pkl')
    if os.path.exists(pkl_path):
        return pkl_path
    csv_path = os.path.join(output_dir, basename + '.csv')
    if os.path.exists(csv_path):
        return csv_path
    return pkl_path  # RETURN pkl PATH SO MISSING-FILE ERRORS ARE CONSISTENT


def build_section_content(sdef: SectionDef, filename: str, model: str, sphinx_dataframe: pd.DataFrame, relative_path_plots: bool, output_dir: str, appendage: str = '') -> str:
    """Build the collapsible Markdown content for a single validated-quantity section.

    Handles all three section types (all-clear, AWT, and standard metrics)
    via SectionDef flags, eliminating the need for three nearly-identical functions.
    """
    data = _load_pkl(filename)
    data = data[data.Model == model]
    if data.empty:
        return ''

    column_labels = list(data.columns)
    buf = io.StringIO()
    buf.write(add_collapsible_segment_start(f'{sdef.title} Metrics'))
    n_total = 0

    for i in range(len(data)):
        threshold_string, energy_threshold, obs_threshold, pred_threshold, mismatch_str = build_threshold_string(data, i)
        energy_channel = data.iloc[i]['Energy Channel']
        obs_thresh_val = obs_threshold.rstrip(' pfu')
        pred_thresh_val = pred_threshold.rstrip(' pfu')
        buf.write(add_collapsible_segment_start(energy_threshold))
        buf.write(add_collapsible_segment('Thresholds Applied', threshold_string))

        if sdef.is_all_clear:
            # ALL-CLEAR: CONTINGENCY TABLE + SKILL SCORES
            hits = data.iloc[i]["All Clear 'True Positives' (Hits)"]
            false_alarms = data.iloc[i]["All Clear 'False Positives' (False Alarms)"]
            correct_negatives = data.iloc[i]["All Clear 'True Negatives' (Correct Negatives)"]
            misses = data.iloc[i]["All Clear 'False Negatives' (Misses)"]
            contingency_values = [hits, false_alarms, correct_negatives, misses]
            selections_filename = _resolve_selections_filename(
                output_dir,
                f'all_clear_selections_{model}_{energy_channel}'
                f'_threshold_{obs_thresh_val}{mismatch_str}{appendage}'
            )
            info_str, n_events, subset = _build_selections_info(
                selections_filename, sphinx_dataframe)
            n_total += n_events
            buf.write(add_collapsible_segment('Validation Info', info_str))
            buf.write(add_collapsible_segment('Contingency Table', build_contingency_table(*contingency_values)))
            skill_score_start_index = 9
            skill_score_labels = list(column_labels[skill_score_start_index:])
            skill_score_values = data.iloc[i, skill_score_start_index:]
            buf.write(add_collapsible_segment('Skill Scores Table', build_skill_score_table(skill_score_labels, skill_score_values)))

        elif sdef.is_awt:
            # AWT: ONE INFO BLOCK PER AWT STRING VARIANT, THEN METRICS
            awt_string_list = [
                'Predicted SEP All Clear',
                'Predicted SEP Peak Intensity (Onset Peak)',
                'Predicted SEP Peak Intensity Max (Max Flux)',
            ]
            for awt_string in awt_string_list:
                selections_filename = _resolve_selections_filename(
                    output_dir,
                    f'{sdef.tag}_selections_{model}_{energy_channel}'
                    f'_threshold_{obs_thresh_val}{mismatch_str}_{awt_string}{appendage}'
                )
                if os.path.exists(selections_filename):
                    info_str, n_events, _ = _build_selections_info(selections_filename, sphinx_dataframe)
                    n_total += n_events
                    buf.write(add_collapsible_segment(f'Validation Info - {awt_string}', info_str))
            metric_index_start = column_labels.index(sdef.metric_label_start)
            metrics_str, plot_string_list, _ = build_metrics_table(
                data, i, metric_index_start, sdef.skip_labels, relative_path_plots)
            buf.write(add_collapsible_segment('Metrics', sdef.metrics_description + '\n' + metrics_str))
            buf.write(plot_subsection(plot_string_list))

        else:
            # STANDARD: INFO + METRICS + PLOTS
            selections_filename = _resolve_selections_filename(
                output_dir,
                f'{sdef.tag}_selections_{model}_{energy_channel}'
                f'_threshold_{obs_thresh_val}{mismatch_str}{appendage}'
            )
            info_str, n_events, subset = _build_selections_info(selections_filename, sphinx_dataframe)
            n_total += n_events
            buf.write(add_collapsible_segment('Validation Info', info_str))
            metric_index_start = column_labels.index(sdef.metric_label_start)
            metrics_str, plot_string_list, _ = build_metrics_table(
                data, i, metric_index_start, sdef.skip_labels, relative_path_plots)
            buf.write(add_collapsible_segment('Metrics', sdef.metrics_description + metrics_str))
            buf.write(plot_subsection(plot_string_list, subset))

        buf.write(add_collapsible_segment_end())

    buf.write(add_collapsible_segment_end())
    return buf.getvalue() if n_total > 0 else ''


# -----------------------------------------------------------------------
# VALIDATION REFERENCE SHEET
# -----------------------------------------------------------------------

def build_validation_reference_section(filename1: str, filename2: Optional[str], filename3: Optional[str] = None) -> str:
    buf = io.StringIO()
    for label, filename in [('Metrics', filename1), ('Skill Scores', filename2), ('Plots', filename3)]:
        if filename:
            buf.write(add_collapsible_segment(label, _ref_csv_markdown(filename)))
    return buf.getvalue()


def construct_validation_reference_sheet(vr_subtext: str, vr_flag_dict: dict, sdef: SectionDef) -> tuple:
    """Add the validation reference sheet for *sdef* to *vr_subtext*, if not already added."""
    if not vr_flag_dict.get(sdef.ref_flag, False):
        return vr_subtext, vr_flag_dict
    buf = io.StringIO()
    buf.write(add_collapsible_segment_start(sdef.ref_flag))
    if sdef.ref_flag == 'Time':
        plot_path = os.path.abspath(os.path.join(config.referencepath, 'AWT_image'))
        buf.write(f'![]({plot_path})\n\n')
    buf.write(build_validation_reference_section(
        _ref_path(sdef.ref_csv_metrics),
        _ref_path(sdef.ref_csv_skills),
        _ref_path(sdef.ref_csv_plots),
    ))
    buf.write(add_collapsible_segment_end())
    vr_flag_dict[sdef.ref_flag] = False
    return vr_subtext + buf.getvalue(), vr_flag_dict


# -----------------------------------------------------------------------
# MARKDOWN → HTML CONVERSION
# -----------------------------------------------------------------------

def get_image_string(original_string: str) -> str:
    """Extract the image path from a Markdown image string ``![](...)``.
    Handles paths that themselves contain parentheses."""
    if original_string.count('(') > 1 or original_string.count(')') > 1:
        last_close = len(original_string) - original_string[::-1].index(')') - 1
        return original_string.split('![](')[1][:last_close]
    return original_string.split('![](')[1].split(')')[0]


def convert_tables_html(lines: list) -> list:
    """Convert Markdown table blocks within *lines* to HTML."""
    out = []
    table_buf = []
    for line in lines:
        if line.startswith('|'):
            table_buf.append(line)
        else:
            if table_buf:
                out.append(markdown.markdown(
                    '\n'.join(table_buf),
                    extensions=['markdown.extensions.tables']
                ))
                table_buf = []
            out.append(line)
    if table_buf:
        out.append(markdown.markdown(
            '\n'.join(table_buf),
            extensions=['markdown.extensions.tables']
        ))
    return out


def convert_plots_html(lines: list) -> list:
    """Replace Markdown image references with <embed> tags for PDF plots."""
    out = []
    for line in lines:
        if line.startswith('!'):
            path = get_image_string(line)
            abs_path = os.path.abspath(os.path.join(config.reportpath, path))
            try:
                reader = pdf.PdfReader(abs_path)
                box = reader.pages[0].mediabox
                out.append(f'<embed src="{path}" alt="" height="{float(box.height) + 60}" width="{box.width}">')
            except Exception:
                logger.warning('Could not read PDF dimensions: %s', abs_path, exc_info=True)
                out.append(f'<embed src="{path}" alt="">')
        elif line:
            out.append(line)
    return out


def convert_bullets_html(lines: list) -> list:
    """Convert Markdown bullet lines to <li> elements."""
    return [
        '<li>' + line.split('*', 1)[1] + '</li>' if line and '*' in line else line
        for line in lines
    ]


def convert_markdown_to_html(text: str, model: str, validation_reference: bool = False) -> str:
    if not validation_reference:
        logger.info('Generating HTML report...%s', model)
    lines = text.split('\n')
    lines = convert_tables_html(lines)
    lines = convert_plots_html(lines)
    lines = convert_bullets_html(lines)
    return markdown.markdown('\n'.join(lines))


def convert_pdf_to_base64(pdf_path: str) -> str:
    with open(pdf_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def embed_pdf_files_in_html(html_content: str, output_html_path: str) -> str:
    soup = bs4.BeautifulSoup(html_content, 'html.parser')
    html_dir = os.path.abspath(config.reportpath)
    for embed in soup.find_all('embed'):
        src = embed.get('src', '')
        if src.endswith('.pdf'):
            pdf_path = os.path.normpath(os.path.join(html_dir, src))
            try:
                embed['src'] = f'data:application/pdf;base64,{convert_pdf_to_base64(pdf_path)}'
            except FileNotFoundError:
                logger.warning('PDF not found for embedding: %s', pdf_path)
    return str(soup)


# -----------------------------------------------------------------------
# GIT INFO BLOCK
# -----------------------------------------------------------------------

def _build_git_info_text() -> str:
    buf = io.StringIO()
    buf.write(f'Date of Report: {datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S UTC")}<br>')
    buf.write('Report generated by SPHINX<br>')
    buf.write(f'This code may be publicly accessed at: [{config.git_repo_url}]({config.git_repo_url})<br>')
    sha_url = os.path.join(config.git_repo_url, 'tree', config.git_commit_sha)
    buf.write(f'Specific git commit SHA: [{config.git_commit_sha}]({sha_url})<br>')
    if config.git_is_dirty:
        dirty_buf = io.StringIO()
        dirty_buf.write('&nbsp;&nbsp;&nbsp;&nbsp;The sphinxval code has changed since the commit listed above.<br>')
        if config.git_changed_files:
            dirty_buf.write('&nbsp;&nbsp;&nbsp;&nbsp;Changes were found in the following files:<br>')
            for fname in config.git_changed_files:
                dirty_buf.write(f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{fname}<br>')
        buf.write(add_collapsible_segment('Dirty Git Repository (details)', dirty_buf.getvalue()) + '<br>')
    return buf.getvalue()


# -----------------------------------------------------------------------
# MAIN ENTRY POINT
# -----------------------------------------------------------------------

def report(output_dir: Optional[str], relative_path_plots: bool, sphinx_dataframe: Optional[pd.DataFrame] = None) -> None:
    """Generate per-model Markdown and HTML validation reports.

    Parameters
    ----------
    output_dir : str or None
        Path to the directory containing SPHINX output pkl files.
        If None, defaults to ``config.outpath/pkl``.
    relative_path_plots : bool
        If True, embed plot paths as relative paths in reports.
    sphinx_dataframe : pandas.DataFrame or None
        The SPHINX evaluated dataframe.  If None, read from
        ``output_dir/SPHINX_evaluated.pkl``.
    """
    pkl_dir = os.path.join(config.outpath, 'pkl')
    csv_dir = os.path.join(config.outpath, 'csv')
    if output_dir is None:
        if os.path.isdir(pkl_dir):
            output_dir = pkl_dir
        elif os.path.isdir(csv_dir):
            output_dir = csv_dir
        else:
            output_dir = pkl_dir  # DEFAULT EVEN IF NOT YET CREATED
    os.makedirs(config.reportpath, exist_ok=True)

    # CLEAR CACHES AT THE START OF EACH CALL SO STALE DATA IS NEVER USED
    # IF report() IS CALLED MULTIPLE TIMES IN THE SAME PROCESS
    _pkl_cache.clear()
    _ref_csv_cache.clear()

    if sphinx_dataframe is None:
        pkl_path = os.path.join(output_dir, 'SPHINX_evaluated.pkl')
        logger.info('No SPHINX dataframe supplied; reading from %s', pkl_path)
        sphinx_dataframe = pd.read_pickle(pkl_path)

    try:
        files = sorted(f for f in os.listdir(output_dir) if f != 'desktop.ini')
    except FileNotFoundError:
        logger.error('Output directory not found: %s', output_dir)
        return

    models = sorted(set(sphinx_dataframe['Model']))
    git_info_text = _build_git_info_text()

    # PRE-INDEX FILES FOR O(1) SECTION-PRESENCE LOOKUPS.
    # BUILD A SET OF (tag, model, appendage) TUPLES FROM THE PKL FILENAMES
    # IN output_dir SO THE INNER LOOPS DON'T NEED TO SCAN THE FULL LIST.
    present_index: set = set()
    for fname in files:
        for sdef in _SECTION_DEFS:
            for appendage in _APPENDAGES:
                stem = fname.rstrip('.pkl')
                if (sdef.file_prefix in fname) and \
                        (appendage in fname) and \
                        (stem.endswith(appendage) or appendage == ''):
                    after_prefix = fname.split(sdef.file_prefix, 1)[1]
                    for model in models:
                        if after_prefix.startswith(model):
                            present_index.add((sdef.tag, model, appendage))

    for model in models:
        markdown_texts = {}
        appendage_set_list = []
        html_text = get_html_report_preamble(model)
        validation_reference_subtext_html = ''
        validation_reference_flag_dict = {
            'All Clear': True, 'AWT': True, 'Duration': True,
            'Flux': True, 'Time': True, 'Probability': True,
        }

        for appendage in _APPENDAGES:
            present = {tag for (tag, m, app) in present_index
                       if m == model and app == appendage}
            if not present:
                continue

            validation_text = 'This model was validated for the following quantities.\n\n'
            validation_reference_subtext = ''
            markdown_text = ''
            report_exists = False

            for sdef in _SECTION_DEFS:
                if sdef.tag not in present:
                    continue
                section_filename = os.path.join(
                    output_dir, f'{sdef.metrics_tag}_metrics{appendage}.pkl')
                if not os.path.exists(section_filename):
                    continue

                section_content = build_section_content(
                    sdef, section_filename, model, sphinx_dataframe,
                    relative_path_plots, output_dir, appendage=appendage)
                if not section_content:
                    continue

                validation_text += f'* {sdef.title}\n'
                markdown_text += section_content
                report_exists = True
                appendage_set_list.append(appendage)

                validation_reference_subtext, validation_reference_flag_dict = (
                    construct_validation_reference_sheet(
                        validation_reference_subtext, validation_reference_flag_dict, sdef))

            validation_reference_subtext_html += validation_reference_subtext

            if report_exists:
                full_markdown = (
                    git_info_text
                    + add_collapsible_segment('Validated Quantities', validation_text)
                    + markdown_text
                )
                md_filename = os.path.join(config.reportpath, f'{model}_report{appendage}.md')
                with open(md_filename, 'w') as f:
                    f.write(full_markdown)
                markdown_texts[appendage] = full_markdown

        # BUILD HTML TABS
        appendage_set_list = sorted(set(appendage_set_list))
        tab_bar = io.StringIO()
        tab_bar.write('<div class="tab">\n')
        tab_bar.write("    <button class=\"tablinks\" onclick=\"openTab(event, 'All')\">All</button>\n")
        for app in appendage_set_list:
            if app:
                tab_id = app.replace('_', '')
                tab_bar.write(f'    <button class="tablinks" onclick="openTab(event, \'{tab_id}\')">{tab_id}</button>\n')
        tab_bar.write('</div>\n')
        html_text += tab_bar.getvalue()
        for app in appendage_set_list:
            if app in markdown_texts:
                html_text += add_tab(app, markdown_texts[app], model)

        # ADD VALIDATION REFERENCE SHEET
        html_text += (
            add_collapsible_segment_start('Validation Reference Sheet')
            + validation_reference_subtext_html
            + add_collapsible_segment_end()
        )

        html_filename = os.path.join(config.reportpath, f'{model}_report.html')
        html_text = embed_pdf_files_in_html(html_text, html_filename)
        with open(html_filename, 'w', encoding='utf-8') as f:
            f.write(html_text)
        logger.info('    Complete')

    # WRITE INDEX PAGE
    html_index = make_index.make_index(
        config.reportpath, banner_text='SPHINX Validation Report Repository')
    with open(os.path.join(config.reportpath, 'index.html'), 'w') as f:
        f.write(html_index)
