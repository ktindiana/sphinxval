#!/usr/bin/env python

import os.path
import argparse
import json
import traceback
import sys
import sphinxval.utils.validation_json_handler as vjson

description = """
Updates a series of CCMC Scoreboard forecast jsons or SPHINX
observation jsons with a value input by the user.
For each json file given to the input (see arguments), puts the
new value in the specified field. Will update all jsons in the
list with the same value.
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
parser.add_argument("--issue_time", type=str, default="",
                    help="New issue time as string in zulu format: YYYY-MM-DDTHH:MM:SSZ")
parser.add_argument("--short_name", type=str, default="",
                    help="New short name.")
parser.add_argument("--append_short_name", type=str, default="",
                    help="String to add to end of current short name. Include a space or underscore or other separator.")
parser.add_argument("--add_pi_trigger", type=str, default="",
                    help="Add particle_intensity trigger block. "
                    "Specify \"observatory=observatory;time=YYYY-MM-DD HH:MM:SS\" where time is the last_data_time of the measurements. Set time=issue to use the issue time as the last_data_time.")

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

issue_time = args.issue_time
short_name = args.short_name
app_short_name = args.append_short_name
add_pi_trigger = args.add_pi_trigger

def maybe_stop():
    if not args.dontstop:
        exit()

def print_json(json_fname, json_pretty):
    print()
    print(f"=== {json_fname} ===")
    print(json_pretty)
    print()



def update_issue_time(kind, json_in, issue_time):
    
    time = vjson.zulu_to_time(issue_time)
    if time == None or time == 0:
        sys.exit("ERROR: Check that the issue time is in the correct format: YYY-MM-DDTHH:MM:SSZ")
        
    #Set new issue time
    if kind == 'observation':
        json_in['sep_observation_submission']['issue_time'] = issue_time
    elif kind == 'forecast':
        json_in['sep_forecast_submission']['issue_time'] = issue_time

    return json_in



def update_short_name(kind, json_in, short_name):
        
    #Set new issue time
    if kind == 'observation':
        json_in['sep_observation_submission']['observatory']['short_name'] = short_name
    elif kind == 'forecast':
        json_in['sep_forecast_submission']['model']['short_name'] = short_name

    return json_in


def append_short_name(kind, json_in, app_short_name):
        
    #Set new issue time
    if kind == 'observation':
        short_name = json_in['sep_observation_submission']['observatory']['short_name']
        json_in['sep_observation_submission']['observatory']['short_name'] = short_name + app_short_name
    elif kind == 'forecast':
        short_name = json_in['sep_forecast_submission']['model']['short_name']
        json_in['sep_forecast_submission']['model']['short_name'] = short_name + app_short_name

    return json_in



def add_pi_trigger_block(kind, json_in, add_pi_trigger):
    """ Add a particle_intensity trigger block and set the
        last_data_time to time.
        
        add_pi_trigger (str) in format:
            "observatory=observatory;time=time"
        
        observatory (str) specifies which observatory provided
        the particle intensities.
        
        Set time to "issue" in order to use the issue time of
        the forecast as the last_data_time.
        
    """
    observatory = ""
    time = ""
    try:
        trigger = add_pi_trigger.strip().split(";")
        observatory = trigger[0].split("=")[1]
        time = trigger[1].split("=")[1]
    except:
        sys.exit(f"Please specify add_pi_trigger in the correct format (see help). You specified {add_pi_trigger}.")
    
    if kind == "forecast":
        key = "sep_forecast_submission"
    if kind == "observations":
        key = "sep_observation_submission"
        
    if time == "issue":
        time = json_in[key]['issue_time']
    
    #Try to add to existing trigger block
    try:
        json_in[key]['triggers'].append( {"particle_intensity":{"observatory":observatory, "last_data_time":time}})
    #If doesn't exist, make trigger block
    except:
        json_in[key].update({"triggers": [{"particle_intensity":{"observatory":observatory, "last_data_time":time}}]})

    return json_in


def update_jsons(kind, json_list, issue_time, short_name, app_short_name):

    if issue_time != "":
        print("JSONs will be updated with new issue time: " + issue_time)
        
    if short_name != "":
        print("JSONs will be updated with new short name: " + short_name)

    if app_short_name != "":
        print("JSON short names will be appended with: " + short_name)

    for json_fname in json_list:
        # Check if the file exists
        if not os.path.isfile(json_fname):
            print("ERROR", json_fname, "File not found")
            maybe_stop()

        # Try to read the file into a json object
        try:
            with open(json_fname) as f:
                json_in = json.load(f)
                
                if issue_time != "":
                    json_in = update_issue_time(kind, json_in, issue_time)

                if short_name != "":
                    json_in = update_short_name(kind, json_in, short_name)
                    
                if app_short_name != "":
                    json_in = append_short_name(kind, json_in, app_short_name)

                if add_pi_trigger != "":
                    add_pi_trigger_block(kind, json_in, add_pi_trigger)

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

        
if data_list:
    print("=== Updating observation jsons ===")
    update_jsons('observation', data_list, issue_time, short_name, app_short_name)
    
if model_list:
    print("=== Updating forecast jsons ===")
    update_jsons('forecast', model_list, issue_time, short_name, app_short_name)
