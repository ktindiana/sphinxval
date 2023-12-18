#!/usr/bin/env python

import os.path
import argparse
import json
import traceback
import sphinxval.utils.validation_json_handler as vjson
import datetime

description = """
Updates a series of CCMC Scoreboard REleASE forecast jsons for
validation. The prediction window is extended to account for
uncertainty in the model timing.
"""

epilog = """
If jsons are given both as an argument list and as a file containing a
list (e.g. both --ModelList and --model), then the argument list is
appended to end of the file list.
"""

parser = argparse.ArgumentParser(description=description, epilog=epilog)
parser.add_argument("--ModelList",
        help="File contining a list of prediction json files.")
parser.add_argument("--model", nargs="*",  default=[],
                    help="Command-line list of prediction jsons files")
parser.add_argument("--dontstop", action="store_true", default=False,
                    help="Don't stop checking jsons on ERROR")

args = parser.parse_args()

# Prepare lists of json files
if args.ModelList:
    model_list = vjson.read_list_of_jsons(args.ModelList)
else:
    model_list = []
model_list.extend(args.model)


def maybe_stop():
    if not args.dontstop:
        exit()

def print_json(json_fname, json_pretty):
    print()
    print(f"=== {json_fname} ===")
    print(json_pretty)
    print()



def update_prediction_window(json_in):
    """ Change REleASE's prediction window to
        start = last_data_time + 20
        end = last_data_time + 100
        
        This takes into the account in the uncertainty in the
        magnetic connectivity, acknowledged by the model as it
        produces 30 and 90 minute forecasts as well. An additional
        10 minutes of uncertainty extends the prediction window
        on either side as they report in Posner (2007) an uncertainty
        of about 10 minutes on timing values.
    """
    #REleASE uses electron intensity for forecasts
    last_data_time = None
    triggers = json_in['sep_forecast_submission']['triggers']
    for trig in triggers:
        try:
            last_data_time = trig['particle_intensity']['last_data_time']
        except:
            continue
    
    if last_data_time == None:
        print("Could not update prediction window. No last data time.")
        return json_in

    
    last_data_time = vjson.zulu_to_time(last_data_time)
    new_start = last_data_time + datetime.timedelta(minutes=20)
    new_end = last_data_time + datetime.timedelta(minutes=100)
    
    new_start = vjson.make_ccmc_zulu_time(new_start)
    new_end = vjson.make_ccmc_zulu_time(new_end)
    
    #REleASE makes forecasts for two blocks
    n = len(json_in['sep_forecast_submission']['forecasts'])
    for i in range(n):
        json_in['sep_forecast_submission']['forecasts'][i]['prediction_window']['start_time'] = new_start
        json_in['sep_forecast_submission']['forecasts'][i]['prediction_window']['end_time'] = new_end

    return json_in




def update_jsons(json_list):

    for json_fname in json_list:
        # Check if the file exists
        if not os.path.isfile(json_fname):
            print("ERROR", json_fname, "File not found")
            maybe_stop()

        # Try to read the file into a json object
        try:
            with open(json_fname,'r') as f:
                
                json_in = json.load(f)
                
                json_in = update_prediction_window(json_in)

            vjson.write_json(json_in,json_fname)
    
        except Exception as e:
            print("ERROR", json_fname, "Could not update json:", e)
            print()
            print(f"=== {json_fname} ===")
            with open(json_fname, 'r') as f:
                  print(f.read())
            maybe_stop()
            continue

        # Report success
        print("OK, json updated", json_fname)

        
    
if model_list:
    print("=== Updating forecast jsons ===")
    update_jsons(model_list)
