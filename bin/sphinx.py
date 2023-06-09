import sphinxval.sphinx
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--ModelList", type=str, default='', \
        help=("List containing filenames of prediction json files."))
parser.add_argument("--DataList", type=str, default="",
        help=("List containing filenames of observation json files."))


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


args = parser.parse_args()
model_list = args.ModelList
data_list = args.DataList

#show_plot = args.showplot
#PrintToScreen = args.PrintToScreen
#write_report = args.WriteReport
#SaveMetrics = args.SaveMetrics
#onemodel = args.OneModel

sphinxval.sphinx.validate(data_list, model_list)
