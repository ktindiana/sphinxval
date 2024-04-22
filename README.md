# sphinxval
SPHINX validation code for solar energetic particle models

This SPHINX code is not associated with sphinx-doc, the automated documentation building code.

SPHINX is developed via community challenges through the SHINE, ISWAT, ESWW, and SEPVAL conferences and in support of the SEP Scoreboards.

Reminder to Windows users: change your system PYTHONPATH environment variable such that your `.../sphinxval/` directory is included.

## Run SPHINX
The exectuables for SPHINX are in the bin directory. 

Input `ModelList`: Text list of model forecast jsons (full or relative path to sphinxval required, typically in a subdirectory in sphinxval/model/)

Input `DataList`: Text list of observation jsons, prepared with fetchsep (full or relative path to sphinxval required, typically in a subdirectory in sphinxval/data/)

Input `TopDirectory`: For models that produce time profiles in txt files, their location is stored in SPHINX by searching through the directories on the user computer. TopDirectory needs to be general enough to find all txt files needed for the validation, but specific enough to avoid searching through unnecessary directories or directories that may contain copies of the same files. 

Forecast files, observations files, and lists of these files can be stored anywhere on the user computer (or accessible drive) as long as full paths are provided to SPHINX.

On a Mac, the run command is:

`python3 bin/sphinx.py --ModelList model/forecasts.list --DataList lists/observations.list --TopDirectory model/`

## Output Files
Results and supporting output files will be written to the directories:

`output/csv` - files containing all observation to forecast matching information and metrics in csv format

`output/pkl` - files containing all observation to forecast matching information and metrics in pkl format (pandas Dataframes)

`output/plots` - validation plots

`reports` md and html files containing a report or summary of the metrics results (easier to read than the individual csv files). Recommend viewing html files in a browser other than Safari.
