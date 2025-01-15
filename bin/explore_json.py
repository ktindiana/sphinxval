#!/usr/bin/env python

import os.path
import argparse
import json
import traceback
import sys
import sphinxval.utils.validation_json_handler as vjson
import sphinxval.utils.object_handler as objh
import pandas as pd

description = """
Pull information from a series of CCMC Scoreboard forecast 
jsons or SPHINX observation jsons with a value input by the user.
For each json file given to the input (see arguments), extracts
information specified by user via set flags.
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
parser.add_argument("--sep_list", action="store_true", default=False,
                    help="Create a list of SEP events contained within the jsons.")
parser.add_argument("--continuous", action="store_true", default=False,
                    help="Check if the json files cover a continuous time range.")


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

sep_list = args.sep_list
continuous = args.continuous


def maybe_stop():
    if not args.dontstop:
        exit()

def print_json(json_fname, json_pretty):
    print()
    print(f"=== {json_fname} ===")
    print(json_pretty)
    print()



def get_short_name(json_in, kind):
        
    #Set new issue time
    if kind == 'observation':
        short_name = json_in['sep_observation_submission']['observatory']['short_name']
  
    elif kind == 'forecast':
        short_name = json_in['sep_forecast_submission']['model']['short_name']

    return short_name


def get_keys(kind):
    if kind == 'observation':
        key1 = 'sep_observation_submission'
        key2 = 'observations'
        key_win = 'observation_window'
    elif kind == 'forecast':
        key1 = 'sep_forecast_submission'
        key2 = 'forecasts'
        key_win = 'prediction_window'
    else:
        raise Exception(f"kind={kind} invalid.  Must be 'observation' or 'forecast'")

    return key1, key2, key_win


def extract_values(json_fname, json_in, kind, dict):
    """ Extract SEP start and end time (if present) from
        json. Save in dict organized by energy channel,
        threshold, and short_name.
    """
    energy_channels = vjson.identify_all_energy_channels_per_json(json_in, kind)
    short_name = get_short_name(json_in, kind)
    key1, key2, key_win = get_keys(kind)
    
    for energy_channel in energy_channels:
        ek = objh.energy_channel_to_key(energy_channel)

        if ek not in dict['windows'].keys():
            dict['windows'].update({ek:{'window_start': [], 'window_end': [],
                'short_name': [], 'source': []}})
        if ek not in dict['sep'].keys():
            dict['sep'].update({ek:{'start_time': [], 'end_time': [],
                'threshold_start': [], 'threshold_end': [], 'threshold_units':[],
                'short_name': [], 'source': []}})
 
        #The energy channel block in forecasts/observations
        dataD = vjson.extract_block(json_in[key1][key2], energy_channel)
        
        if key_win in dataD.keys():
            dict['windows'][ek]['window_start'].append(dataD[key_win]['start_time'])
            dict['windows'][ek]['window_end'].append(dataD[key_win]['end_time'])
            dict['windows'][ek]['short_name'].append(short_name)
            dict['windows'][ek]['source'].append(json_fname)
        
        if "event_lengths" in dataD.keys():
            events = dataD['event_lengths']
            for ev in events:
                dict['sep'][ek]['start_time'].append(ev['start_time'])
                dict['sep'][ek]['end_time'].append(ev['end_time'])
                dict['sep'][ek]['threshold_start'].append(ev['threshold_start'])
                dict['sep'][ek]['threshold_end'].append(ev['threshold_end'])
                dict['sep'][ek]['threshold_units'].append(ev['threshold_units'])
                dict['sep'][ek]['short_name'].append(short_name)
                dict['sep'][ek]['source'].append(json_fname)

    return dict




def explore_jsons(kind, json_list, sep_list, continuous):

    dict = {'windows': {}, 'sep': {}} #gather json information

    if sep_list:
        print("Creating list of SEP events from JSONs.")
    
    if continuous:
        print("Checking that JSONs cover a continous timeframe.")

    if not sep_list and not continuous:
        sys.exit("Please specify an action by selecting the --sep_list or --continuous flag. Use explore_json.py --help for more information.")

    for json_fname in json_list:
        # Check if the file exists
        if not os.path.isfile(json_fname):
            print("ERROR", json_fname, "File not found")
            maybe_stop()

        # Try to read the file into a json object
        try:
            with open(json_fname) as f:
                json_in = json.load(f)
                dict = extract_values(json_fname, json_in, kind, dict)

        except Exception as e:
            print("ERROR", json_fname, "Could not access info in json:", e)
            print()
            print(f"=== {json_fname} ===")
            with open(json_fname, 'r') as f:
                  print(f.read())
            maybe_stop()
            continue



        # Report success
        print("OK, json info collected", json_fname)
    
    for ek in dict['windows'].keys():
        df_win = pd.DataFrame(dict['windows'][ek])
        print(df_win)
        df_win.to_csv(f"../test_windows_{ek}.csv")


    for ek in dict['sep']:
        df_sep = pd.DataFrame(dict['sep'][ek])
        print(df_sep)
        df_sep.to_csv(f"../test_sep_{ek}.csv")

        
if data_list:
    print("=== Exploring observation jsons ===")
    explore_jsons('observation', data_list, sep_list, continuous)
    
if model_list:
    print("=== Exploring forecast jsons ===")
    explore_jsons('forecast', model_list, sep_list, continuous)
