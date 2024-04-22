import sphinxval.sphinx
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--ModelList", type=str, default='', \
        help=("List containing filenames of prediction json files."))
parser.add_argument("--DataList", type=str, default="",
        help=("List containing filenames of observation json files."))
parser.add_argument("--TopDirectory", default=None,
        help=("If json files and time profile txt files not in the same directory, "
            "specify the base path in which the model json and time profile txt files "
            "might be found. Optional. "))
parser.add_argument("--resume",
        help=("Specify resume to add forecasts to an existing "
            "dataframe and recalculate metrics."), action="store_true")
parser.add_argument("--DataFrame", type=str, default="",
        help=("Filename of existing Pandas DataFrame containing the results of a previous SPHINX run in pkl format. "
            "Must specify if choose resume."))
parser.add_argument("--RelativePathPlots", type=bool, default=True, \
        help=("Generate reports with relative paths for plots"))


#parser.add_argument("--OneModel", type=str, default="",
#        help=("If want to validate only one model in the list, specify "
#            "model name here exactly as it appears in \"short_name\" "
#            "in the json files."))

#parser.add_argument("--showplot", help="Flag to display plots (do not recommend)", \
#            action="store_true")
#parser.add_argument("--PrintToScreen", help="Flag to print metrics " \
#                    "dictionaries to screen.", action="store_true")
#parser.add_argument("--WriteReport", help="Flag to write out a pdf " \
#                    "report of the validation results.",action="store_true")
#parser.add_argument("--SaveMetrics", help=("Flag to write out a json file "
#                    "containing the validation results."),action="store_true")

#parser.add_argument("--ReportOnly", help=("Do not run SPHINX; simply generate reports from pre-existing SPHINX output. "), action="store_true")



args = parser.parse_args()
model_list = args.ModelList
data_list = args.DataList
top = args.TopDirectory
resume = args.resume
df_pkl = args.DataFrame
relative_path_plots = args.RelativePathPlots

#report_only = args.ReportOnly


#show_plot = args.showplot
#PrintToScreen = args.PrintToScreen
#write_report = args.WriteReport
#SaveMetrics = args.SaveMetrics
#onemodel = args.OneModel


sphinxval.sphinx.validate(data_list, model_list, top, resume, df_pkl)
sphinxval.sphinx.report.report(None, relative_path_plots)
