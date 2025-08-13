<img src="https://github.com/lukestegeman/sphinxval/blob/a9692f89426f35f9bcdcf38f2cfa851aed75ce80/logo/sphinx-logo-official.png?raw=true" width="250" align="right"/>

### sphinxval
SPHINX validation code for solar energetic particle models

> [!warning]  
>  This SPHINX code is not associated with sphinx-doc, the automated documentation building code.

SPHINX is developed via community challenges through the SHINE, ISWAT, ESWW, and SEPVAL conferences and in support of the SEP Scoreboards.

> [!note]
> Reminder to Windows users: change your system PYTHONPATH environment variable such that your `../sphinxval/` directory is included.


## Run SPHINX
The executables for SPHINX are in the `bin` directory. 

`ModelList`: __Required__. Text list of model forecast jsons (full or relative path to sphinxval required, typically in a subdirectory in sphinxval/model/)

`DataList`: __Required__. Text list of observation jsons prepared with [fetchsep](https://github.com/ktindiana/fetchsep) (full or relative path to sphinxval required, typically in a subdirectory in sphinxval/data/)

`TopDirectory`: _Optional_. For models that produce time profiles in txt files, if the forecast jsons and time profile txt files are in different directories, then SPHINX searches through the directories on the user computer to find the location of the txt files starting at TopDirectory. TopDirectory needs to be general enough to find all txt files needed for the validation, but specific enough to avoid searching through unnecessary directories or directories that may contain copies of the same files. 

Forecast files, observations files, and lists of these files can be stored anywhere on the user computer (or accessible drive) as long as full paths are provided to SPHINX.

The run command is

`python3 bin/sphinx.py --ModelList model/forecasts.list --DataList lists/observations.list`

`python3 bin/sphinx.py --ModelList model/forecasts.list --DataList lists/observations.list --TopDirectory model/MyModel`

## Output Files
Results and supporting output files are written to the directories:

`output/csv` - files containing all observation to forecast matching information and metrics in csv format

`output/pkl` - files containing all observation to forecast matching information and metrics in pkl format (pandas Dataframes)

`output/plots` - validation plots

`reports` md and html files containing a report or summary of the metrics results (easier to read than the individual csv files). Recommend viewing html files in a browser other than Safari.

## Run SPHINX Unit Tests (for developers)
To run the full suite of SPHINX unit tests, execute `python3 -m unittest` from the base directory.

To run the full suite of SPHINX unit tests with verbose output, execute `python3 -m unittest --verbose` from the base directory.

To run the full suite of SPHINX unit tests with concise output (without extraneous print statements), execute `python3 -m unittest -b` from the base directory.

To run a specific SPHINX unit test (for example, the `TestMatchAllForecasts.test_match_all_forecasts_1` test), execute `python3 -m unittest tests.test_match.TestMatchAllForecasts.test_match_all_forecasts_1` from the base directory.

