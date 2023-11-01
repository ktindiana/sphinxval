#!/usr/bin/env python

import os.path
import argparse
import json
import traceback
import sphinxval.utils.validation_json_handler as vjson

description = """
Checks a series of CCMC Scoreboard forecast jsons or SPHINX
observation jsons for compatibility with the SPHINX validation tool.
For each json file given to the input (see arguments), prints OK for
jsons passing all tests, and ERROR for jsons that have a problem,
followed by a printout of the file/json.
"""

epilog = """
If jsons are given both as an argument list and as a file containing a
list (e.g. both --ModelList and --model), then the argument list is
appended to end of the file list.
"""

parser = argparse.ArgumentParser(description=description, epilog=epilog)
parser.add_argument("--ModelList",
        help="File contining a list of prediction json files.")
parser.add_argument("--DataList",
        help="File containing a list of observation json files.")
parser.add_argument("--model", nargs="*",  default=[],
                    help="Command-line list of prediction jsons files")
parser.add_argument("--data", nargs="*", default=[],
                    help="Command-line list of observation  jsons files")
parser.add_argument("--dontstop", action="store_true", default=False,
                    help="Don't stop checking jsons on ERROR")
parser.add_argument("--print", action="store_true", default=False,
                    help="Print each json even when OK")
args = parser.parse_args()

# Prepare lists of json files
if args.ModelList:
    model_list = vjson.read_list_of_jsons(args.ModelList)
else:
    model_list = []
model_list.extend(args.model)

if args.DataList:
    data_list = vjson.read_list_of_jsons(args.DataList)
else:
    data_list = []
data_list.extend(args.data)

def maybe_stop():
    if not args.dontstop:
        exit()

def print_json(json_fname, json_pretty):
    print()
    print(f"=== {json_fname} ===")
    print(json_pretty)
    print()
    
def check_jsons(kind, json_list):
    if kind == 'observation':
        object_from_json = vjson.observation_object_from_json        
    elif kind == 'forecast':
        object_from_json = vjson.forecast_object_from_json        

    for json_fname in json_list:
        # Check if the file exists
        if not os.path.isfile(json_fname):
            print("ERROR", json_fname, "File not found")
            maybe_stop()

        # Try to read the file into a json object
        try:
            json_obj = vjson.read_json_list([json_fname], verbose=False)[0]
            json_pretty = json.dumps(json_obj, indent=4) # Save pretty format; see below
        except Exception as e:
            print("ERROR", json_fname, "Failed to read json:", e)
            print()
            print(f"=== {json_fname} ===")
            with open(json_fname, 'r') as f:
                  print(f.read())
            maybe_stop()
            continue

        # Try to extract energy channels from the json object
        try:
            all_energy_channels = vjson.identify_all_energy_channels([json_obj], kind)
        except Exception as e:
            print("ERROR", json_fname, "Failed to extract energy channel:", e)
            print_json(json_fname, json_pretty)
            maybe_stop()
            continue

        # Try to build object for each of the energy channels
        errored = False
        for channel in all_energy_channels:
            try:
                sphinx_obj = object_from_json(json_obj, channel)
            except Exception as e:
                print("ERROR", json_fname, f"Failed to load object (channel {channel}):", e)
                print(traceback.format_exc())
                print_json(json_fname, json_pretty)
                maybe_stop()
                errored = True
                break

            # Check if trigger timestamps are valid
            if kind == 'forecast':
                is_valid = sphinx_obj.valid_forecast(verbose=True)
                if not is_valid:
                    print("ERROR", json_fname, f"valid_forecast() = {is_valid}")
                    print_json(json_fname, json_pretty)
                    maybe_stop()
                    errored = True
                    break

        if errored:
            continue

        # Report success
        print("OK", json_fname)
        if args.print:
            # Note: printing a saved formatted string from above because
            # one of the functions above changes the object type of json_obj
            # TODO: is that an unwanted side-effect?
            print_json(json_fname, json_pretty)
        
if data_list:
    print("=== Checking observation jsons ===")
    check_jsons('observation', data_list)
    
if model_list:
    print("=== Checking forecast jsons ===")
    check_jsons('forecast', model_list)
