### SPHINX MARKDOWN GENERATOR -- please ignore the mess for now

import pandas as pd
import numpy as np
pd.set_option('mode.chained_assignment', None)
import datetime
import os
import pickle

import markdown
import PyPDF2 as pdf
import glob

from . import config 

def replace_backslash_with_forward_slash(input_string):
    replaced_string = input_string.replace('\\', '/')
    return replaced_string

def formatting_function(value):
    condition = True
    num_floats = 2
    while condition:
        formatter_prefix = '{:.' + str(num_floats) + 'f' + '}'
        formatted_value = formatter_prefix.format(value)
        not_present = True
        for digit in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
            if digit in formatted_value:
                not_present = False
        if not_present:
            num_floats += 1
        else:
            condition = False
        if num_floats >= 5:
            condition = False
    return formatted_value
    
def transpose_markdown(text):
    table = text.split('\n')
    output = '| Metric | Value |\n'
    output += '|:' + '-' * 48 + ':|:' + '-' * 48 + ':|\n'
    labels = table[0].split('|')
    values = table[2].split('|')
    for i in range(1, len(labels)):
        output += '|' + labels[i] + '|' + values[i] + '|\n'
    output = output.rstrip('|||\n') + '|\n'
    output = format_markdown_table(output)
    return output

'''
def drop_empty_rows(df, pivot_column_name):
    pivot_column_index = list(df.columns).index(pivot_column_name)
    df = df.dropna(subset=df.columns[pivot_column_index+1:], how='all')
    df = df.reset_index(drop=True)
    return df
'''
 
def make_markdown_table(column_1, column_2, dataframe, width=50):
    # INPUT MUST BE TWO COLUMN MATRIX; FIRST COLUMN LABELS, SECOND COLUMN DATA
    zero_to_one_metric_name = ['Percent Correct', 'Hit rate', 
                               'False Alarm Rate', 'Frequency of Misses', 
                               'Probability of Correct Negatives', 
                               'Frequency of Hits', 'False Alarm Ratio', 
                               'Detection Failure Ratio', 
                               'Frequency of Correct Negatives', 
                               'Threat Score', 'Brier Score', 
                               'Pearson Correlation Coefficient (linear)', 
                               'Pearson Correlation Coefficient (log)']    
    int_metric_name = ['Hits (TP)', 'Misses (FN)', 
                       'False Alarms (FP)', 'Correct Negatives (TN)']                   
    rows = dataframe.index.to_list()
    numbers = list(dataframe.to_numpy())
    
    # CHANGE RATIOS TO PERCENTAGES
    '''
    for i in range(0, len(rows)):
         if 'percent' in rows[i].lower():
            numbers[i] *= 100.0
    '''
     
    # CLEAN UP numbers
    formatted_numbers = []
    for i in range(0, len(rows)):
        if rows[i] in zero_to_one_metric_name:
            formatter = '{:.2f}'.format
        elif rows[i] in int_metric_name:
            formatter = '{:02d}'.format
        else:
            formatter = formatting_function
        if numbers[i][0] is None:
            formatter = lambda x : 'NaN'
        formatted_numbers.append(formatter(numbers[i][0]))    
    table = '| ' + column_1 + ' ' * (width - 1 - len(column_1)) + ' | ' + column_2 + ' ' * (width - 1 - len(column_2)) + ' |\n'
    table += '|:' + (width - 1) * '-' + ':|:' + (width - 1) * '-' + ':|\n'
    for i in range(0, len(rows)):
        rows[i] = rows[i].replace('|', '\|')
        table += '| ' + rows[i] + ' ' * (width - len(rows[i])) + '| ' + formatted_numbers[i] + ' ' * (width - len(formatted_numbers[i])) + '|\n'
    return table
    
def format_markdown_table(text, width=50):
    zero_to_one_metric_name = ['Percent Correct', 'Hit rate', 
                               'False Alarm Rate', 'Frequency of Misses', 
                               'Probability of Correct Negatives', 
                               'Frequency of Hits', 'False Alarm Ratio', 
                               'Detection Failure Ratio', 
                               'Frequency of Correct Negatives', 
                               'Threat Score', 'Brier Score', 
                               'Pearson Correlation Coefficient (linear)', 
                               'Pearson Correlation Coefficient (log)']    
    int_metric_name = ['Hits (TP)', 'Misses (FN)', 
                       'False Alarms (FP)', 'Correct Negatives (TN)']
    table = text.split('\n')
    table_string = ''
    for i in range(0, len(table)):
        cells = table[i].split('|')[1:-1]
        for j in range(0, len(cells)):
            try:
                value = float(cells[j])
            except ValueError:
                value = cells[j].strip(' ')
            formatter = None
            if (type(value) == str):    
                cells[j] = value
                if ('Skill Score' in value) or ('Skill Statistic' in value):
                    formatter = formatting_function
                elif (value in zero_to_one_metric_name):
                    formatter = formatting_function
                elif value in int_metric_name:
                    formatter = lambda x : str(int(round(x, 0)))
                else:
                    formatter = None
            else:
                if formatter is None:
                    if abs(value) >= 1000.0:
                        formatter = '{:.4e}'.format
                    else:
                        formatter = '{:.2f}'.format
                cells[j] = formatter(value)

        for j in range(0, len(cells)):
            if len(cells[j]) < width:
                counter = 0
                while len(cells[j]) < width:
                    if j != 0:
                        if counter % 2 == 0:    
                            cells[j] = ' ' + cells[j]
                        else:
                            cells[j] = cells[j] + ' '
                    else:
                        cells[j] = ' ' + cells[j]
                    counter += 1
        line = '|'
        for j in range(0, len(cells)):
            line += cells[j] + '|'
        line += '\n'
        table_string += line
    return table_string

def add_collapsible_segment(header, text):
    markdown = '<details>\n'
    markdown += '<summary>' + header + '</summary>\n\n'
    markdown += text + '\n'
    markdown += '</details>\n'
    return markdown

def add_collapsible_segment_start(header, text):
    markdown = '<details>\n'
    markdown += '<summary>' + header + '</summary>\n\n'
    markdown += text + '\n'
    markdown += '<blockquote>\n\n'
    return markdown

def add_collapsible_segment_end():
    markdown = '</blockquote>\n'
    markdown += '</details>\n\n'
    return markdown

def add_collapsible_segment_nest(header_list, text_list, depth=0):
    result = []
    for i in range(0, len(header_list)):
        if isinstance(header_list[i], list):
            sublist_result = add_collapsible_segment_nest(header_list[i], 
                                                          text_list[i], 
                                                          depth + 1)
            result.append(sublist_result)
        else:
            result.append((depth, header_list[i], text_list[i]))
    return result

def build_info_string_header(value, limit_message):
    info_string = 'Instruments and observed values used in validation.<br>'
    info_string += 'N = ' + str(value) + '<br>\n'
    info_string += limit_message
    return info_string

def define_colors():
    text = '<style>\n'
    text += '    .red {\n'
    text += '            background-color: #fad5d2;\n'
    text += '        }\n'
    text += '    .green {\n'
    text += '           background-color: #89d99e;\n'
    text += '        }\n'
    text += '</style>\n'
    return text

def build_contingency_table(yes_yes, yes_no, no_no, no_yes, text_color='black'):
    table_string = ''
    table_string += '| |Observed Yes|Observed No|\n'
    table_string += '|----|:----:|:----:|\n'
    table_string += '|Predicted Yes|' + str(yes_yes) + '|' + str(yes_no) + '|\n'
    table_string += '|Predicted No|' + str(no_yes) + '|' + str(no_no) + '|\n'
    return table_string

def add_text_color(text, color):
    new_text = '<span style="' + 'color:' + color + '">' + text + '</span>'
    return new_text


def build_skill_score_table(labels, values):
    table = pd.DataFrame(data=[values], columns=labels)
    table_string = '\n' + transpose_markdown(table.to_markdown(index=False)) + '\n'
    return table_string

def build_info_events_table(filename, sphinx_dataframe, subset_list, subset_replacement_dict, selections_limit=1000):
    data = pd.read_pickle(filename)
    subset = data[subset_list]
    subset.insert(0, 'Observatory', 'dummy')
    selection_index = list(data.index)
    subset['Observatory'] = sphinx_dataframe.loc[selection_index, 'Observatory'].to_list()
    subset = subset.rename(columns=subset_replacement_dict)
    if len(subset) > selections_limit:
        subset = subset.iloc[:selections_limit]
        limit_message = 'This list has been truncated to the first ' + str(selections_limit) + ' entries. See ' + filename + ' for full list.\n'
    else:
        limit_message = ''
    output = '\n' + subset.to_markdown(index=False) + '\n'
    n_events = len(data)
    return output, n_events, limit_message

def build_threshold_string(data, k):
    energy_threshold_data = data.iloc[k]['Energy Channel']
    if energy_threshold_data.count('MeV') > 1:
        energy_threshold_values = energy_threshold_data.split('_')
        energy_threshold = ''
        mismatch_allowed_string = '_mm'
        for i in range(0, len(energy_threshold_values)):
            energy_threshold_min_split = energy_threshold_values[i].split('min.')[1]
            energy_threshold_max_split = energy_threshold_values[i].split('.max.')[1]
            energy_threshold_min = energy_threshold_min_split.split('.max.')[0]
            energy_threshold_max = energy_threshold_max_split.split('.units.')[0]
            if float(energy_threshold_max) < 0:
                energy_threshold += '> ' + energy_threshold_min + ' MeV , '
            else:
                energy_threshold += energy_threshold_min + ' < E < ' + energy_threshold_max + ' MeV , '
        energy_threshold = energy_threshold.rstrip(' , ')        
    else:
        energy_threshold = '> ' + energy_threshold_data.split('.')[1] + ' MeV'
        energy_threshold_min_split = energy_threshold_data.split('min.')[1]
        energy_threshold_max_split = energy_threshold_data.split('.max.')[1]
        energy_threshold_min = energy_threshold_min_split.split('.max.')[0]
        energy_threshold_max = energy_threshold_max_split.split('.units.')[0]
        if float(energy_threshold_max) < 0:
            energy_threshold = '> ' + energy_threshold_min + ' MeV'
        else:
            energy_threshold = energy_threshold_min + ' < E < ' + energy_threshold_max + ' MeV'
        mismatch_allowed_string = ''
    obs_threshold = data.iloc[k]['Threshold'].split('.units.')[0].split('threshold.')[1] + ' pfu'
    pred_threshold = data.iloc[k]['Prediction Threshold'].split('.units.')[0].split('threshold.')[1] + ' pfu'
    threshold_string = '* Energy Channel: ' + energy_threshold + '\n'
    threshold_string += '* Observation Threshold: ' + obs_threshold + '\n'
    threshold_string += '* Predictions Threshold: ' + pred_threshold + '\n'
    return threshold_string, energy_threshold, obs_threshold, pred_threshold, mismatch_allowed_string

def build_all_clear_skill_scores_section(filename, model, sphinx_dataframe, appendage=''):
    data = pd.read_pickle(filename)
    data = data[data.Model == model]
    column_labels = data.columns
    text = ''
    number_rows = data.shape[0]
    if number_rows > 0:
        text += add_collapsible_segment_start('All Clear Skill Scores', '')
    skill_score_start_index = 9
    skill_score_table_labels = list(column_labels[skill_score_start_index:])
    for i in range(0, number_rows):
        threshold_string, energy_threshold, obs_threshold, pred_threshold, mismatch_allowed_string = build_threshold_string(data, i)
        hits = data.iloc[i]["All Clear 'True Positives' (Hits)"]
        false_alarms = data.iloc[i]["All Clear 'False Positives' (False Alarms)"]
        correct_negatives = data.iloc[i]["All Clear 'True Negatives' (Correct Negatives)"]
        misses = data.iloc[i]["All Clear 'False Negatives' (Misses)"]
        contingency_table_values = [hits, false_alarms, correct_negatives, misses]
        contingency_table_string = build_contingency_table(*contingency_table_values)
        selections_filename = output_dir__ + 'all_clear_selections_' + model + '_' + data.iloc[i]['Energy Channel'] + '_threshold_' + obs_threshold.rstrip(' pfu') + mismatch_allowed_string + appendage + '.pkl'
        subset_list = ['Prediction Window Start', 'Prediction Window End']
        subset_list = append_subset_list(selections_filename, subset_list, 'Prediction Window End', 'Units')
        info_string_, n_events, limit_message = build_info_events_table(selections_filename, sphinx_dataframe, subset_list, {})
        info_string = build_info_string_header(sum(contingency_table_values), limit_message)
        info_string += info_string_
        skill_score_table_values = data.iloc[i, skill_score_start_index:]
        skill_score_table_string = build_skill_score_table(skill_score_table_labels, skill_score_table_values)
        text += add_collapsible_segment_start(energy_threshold, '')
        text += add_collapsible_segment('Thresholds Applied', threshold_string)
        text += add_collapsible_segment('Validation Info', info_string)
        text += add_collapsible_segment('Contingency Table', contingency_table_string)
        text += add_collapsible_segment('Skill Scores Table', skill_score_table_string)
        text += add_collapsible_segment_end()
    text += add_collapsible_segment_end()
    return text

def build_metrics_table(data, current_index, metric_index_start, skip_label_list):
    metrics_table_string = ''
    column_labels = list(data.columns)
    subset = data.iloc[current_index, metric_index_start:]
    subset = pd.DataFrame(subset, index=column_labels[metric_index_start:])
    for i in range(0, len(skip_label_list)):
        subset = subset.drop(skip_label_list[i], axis=0)
    column_1 = 'Metric'
    column_2 = 'Value'
    metrics_table_string += '\n' + make_markdown_table(column_1, column_2, subset) + '\n'
    plot_string_list, plot_file_string_list = build_plot_string_list(data, current_index)
    return metrics_table_string, plot_string_list, plot_file_string_list
    
def build_plot_string_list(data, current_index):
    plot_string_list = []
    plot_file_string_list = []
    if ('Scatter Plot' in data.columns):    
        plot_string_ = data.iloc[current_index]['Scatter Plot']
        if plot_string_ == '':
            plot_string = 'No image files found.\n\n'
            plot_file_string = ''
            plot_string_list.append(plot_string)
            plot_file_string_list.append(plot_file_string)
        else:    
            if relative_path_plots__:
                plot_string = os.path.relpath(plot_string_, 'reports/')
                plot_file_string = os.path.relpath(plot_string_, '.')
            else:
                plot_string = os.path.abspath(plot_string_)
                plot_file_string = plot_string + ''
            plot_string = replace_backslash_with_forward_slash(plot_string) + '.pdf'
            plot_file_string = replace_backslash_with_forward_slash(plot_file_string) + '.pdf'
            plot_string_list.append('![](' +  plot_string + ')\n\n')
            plot_file_string_list.append(plot_file_string)
    
    if ('ROC Curve Plot' in data.columns):
        plot_string_ = data.iloc[current_index]['ROC Curve Plot']
        if plot_string_ == '':
            plot_string = 'No image files found.\n\n'
            plot_file_string = ''
            plot_string_list.append(plot_string)
            plot_file_string_list.append(plot_file_string)
        else:    
            if relative_path_plots__:
                plot_string = os.path.relpath(plot_string_, 'reports/')
                plot_file_string = os.path.relpath(plot_string_, '.')
            else:
                plot_string = os.path.abspath(plot_string_)
                plot_file_string = plot_string + ''
            plot_string = replace_backslash_with_forward_slash(plot_string) + '.pdf'
            plot_file_string = replace_backslash_with_forward_slash(plot_file_string) + '.pdf'
            plot_string_list.append('![](' +  plot_string + ')\n\n')
            plot_file_string_list.append(plot_file_string)
 
    if 'Time Profile Selection Plot' in data.columns:
        time_profile_plot_string = data.iloc[current_index]['Time Profile Selection Plot']
        time_profile_plot_string_list = time_profile_plot_string.split(';')
        if time_profile_plot_string == '':
            pass
        else:
            for i in range(0, len(time_profile_plot_string_list)):
                if relative_path_plots__:
                    plot_string = os.path.relpath(time_profile_plot_string_list[i], 'reports/')
                    plot_file_string = os.path.relpath(time_profile_plot_string_list[i], '.')
                else:
                    plot_string = os.path.abspath(time_profile_plot_string_list[i])
                    plot_file_string = plot_string + ''
                plot_string = replace_backslash_with_forward_slash(plot_string)
                plot_file_string = replace_backslash_with_forward_slash(plot_file_string)
                plot_string_list.append('![](' +  plot_string + ')\n\n')
                plot_file_string_list.append(plot_file_string)
    return plot_string_list, plot_file_string_list

def append_subset_list(selections_filename, subset_list, include_after, exclusion_pattern=None):
    # HACKY
    a = open(selections_filename.replace('pkl', 'csv'), 'r')
    read = a.readlines()
    a.close()
    columns = read[0].lstrip(',').split(',')
    for i in range(0, len(columns)):
        columns[i] = columns[i].rstrip('\n').rstrip('\\')
    include = False
    for i in range(0, len(columns)):
        if include:
            if exclusion_pattern is None:
                subset_list.append(columns[i])
            else:
                if not (exclusion_pattern in columns[i]):
                    subset_list.append(columns[i])
        if columns[i] == include_after:
            include = True
    return subset_list

def get_awt_filename(data, i, output_dir, section_tag, model, obs_threshold, pred_threshold, mismatch_allowed_string, awt_string, appendage=''):
    selections_filename = output_dir + section_tag + '_selections_' + model + '_' + data.iloc[i]['Energy Channel'] + '_threshold_' + obs_threshold.rstrip(' pfu') + mismatch_allowed_string + '_' + awt_string + appendage + '.pkl'
    return selections_filename

def build_section_awt(filename, model, sphinx_dataframe, metric_label_start, section_title, section_tag, metrics_description_string, skip_label_list=[], rename_dict={}, appendage=''):
    data = pd.read_pickle(filename)
    data = data[data.Model == model]
    column_labels = data.columns    
    metric_index_start = list(column_labels).index(metric_label_start)
    text = ''
    number_rows = data.shape[0]
    awt_index = 0
    if number_rows > 0:
        text += add_collapsible_segment_start(section_title + ' Metrics', '')

    awt_string_list = ['Predicted SEP All Clear', 'Predicted SEP Peak Intensity (Onset Peak)', 'Predicted SEP Peak Intensity Max (Max Flux)']
    for i in range(0, number_rows):
        threshold_string, energy_threshold, obs_threshold, pred_threshold, mismatch_allowed_string = build_threshold_string(data, i)
        metrics_string = metrics_description_string + '\n' 
        metrics_string_, plot_string_list, plot_file_string_list = build_metrics_table(data, i, metric_index_start, skip_label_list)
        metrics_string += metrics_string_
        text += add_collapsible_segment_start(energy_threshold, '')
        text += add_collapsible_segment('Thresholds Applied', threshold_string)
        for j in range(0, len(awt_string_list)):
            awt_string = awt_string_list[j]
            selections_filename = get_awt_filename(data, i, output_dir__, section_tag, model, obs_threshold, pred_threshold, mismatch_allowed_string, awt_string, appendage)
            subset_list = ['Prediction Window Start', 'Prediction Window End']
            if os.path.exists(selections_filename):            
                subset_list = append_subset_list(selections_filename, subset_list, 'Prediction Window End', 'Units')
                info_string_, n_events, limit_message = build_info_events_table(selections_filename, sphinx_dataframe, subset_list, rename_dict)
                info_string = build_info_string_header(n_events, limit_message)
                info_string += info_string_
                text += add_collapsible_segment('Validation Info - ' + awt_string, info_string)
        text += add_collapsible_segment('Metrics', metrics_string)
        plot_counter = 1
        last_plot_type = ''
        for j in range(0, len(plot_string_list)):
            plot_type = get_plot_type(plot_string_list[j])
            if plot_type != last_plot_type:
                plot_counter = 1
            else:
                plot_counter += 1
            text += add_collapsible_segment('Plot: ' + plot_type + ' ' + str(plot_counter), plot_string_list[j])
            last_plot_type = plot_type + ''
            
        text += add_collapsible_segment_end()
        
    text += add_collapsible_segment_end()
    return text

def build_section(filename, model, sphinx_dataframe, metric_label_start, section_title, section_tag, metrics_description_string, skip_label_list=[], rename_dict={}, appendage=''):
    data = pd.read_pickle(filename)
    data = data[data.Model == model]
    column_labels = data.columns    
    metric_index_start = list(column_labels).index(metric_label_start)
    text = ''
    number_rows = data.shape[0]
    awt_index = 0
    if number_rows > 0:
        text += add_collapsible_segment_start(section_title + ' Metrics', '')
    for i in range(0, number_rows):
        threshold_string, energy_threshold, obs_threshold, pred_threshold, mismatch_allowed_string = build_threshold_string(data, i)
        selections_filename = output_dir__ + section_tag + '_selections_' + model + '_' + data.iloc[i]['Energy Channel'] + '_threshold_' + obs_threshold.rstrip(' pfu') + mismatch_allowed_string + appendage + '.pkl'
        subset_list = ['Prediction Window Start', 'Prediction Window End']
        subset_list = append_subset_list(selections_filename, subset_list, 'Prediction Window End', 'Units')
        info_string_, n_events, limit_message = build_info_events_table(selections_filename, sphinx_dataframe, subset_list, rename_dict)
        info_string = build_info_string_header(n_events, limit_message)
        info_string += info_string_
        metrics_string = metrics_description_string + '' 
        metrics_string_, plot_string_list, plot_file_string_list = build_metrics_table(data, i, metric_index_start, skip_label_list)
        metrics_string += metrics_string_
        text += add_collapsible_segment_start(energy_threshold, '')
        text += add_collapsible_segment('Thresholds Applied', threshold_string)
        text += add_collapsible_segment('Validation Info', info_string)
        text += add_collapsible_segment('Metrics', metrics_string)
        plot_counter = 1
        last_plot_type = ''
        for j in range(0, len(plot_string_list)):
            plot_type = get_plot_type(plot_string_list[j])
            if plot_type != last_plot_type:
                plot_counter = 1
            else:
                plot_counter += 1
            text += add_collapsible_segment('Plot: ' + plot_type + ' ' + str(plot_counter), plot_string_list[j])
            last_plot_type = plot_type + ''
        text += add_collapsible_segment_end()
    text += add_collapsible_segment_end()
    return text
    
def build_validation_reference_section(text, filename1, filename2, filename3=None):
    data = pd.read_csv(filename1, skiprows=1)
    table = '\n' + data.to_markdown(index=False) + '\n'
    text += add_collapsible_segment('Metrics', table)
    if filename2:
        data = pd.read_csv(filename2, skiprows=1)
        table = '\n' + data.to_markdown(index=False) + '\n'
        text += add_collapsible_segment('Skill Scores', table)
    if filename3:
        data = pd.read_csv(filename3, skiprows=1)
        table = '\n' + data.to_markdown(index=False) + '\n'
        text += add_collapsible_segment('Plots', table)
    # text += add_collapsible_segment_end()
    return text

def construct_validation_reference_sheet(vr_subtext, vr_flag_dict, vr_flag, vr_filename_1, vr_filename_2=None, vr_filename_3=None):
    if vr_flag_dict[vr_flag]:
        vr_subtext += add_collapsible_segment_start(vr_flag, '')

        # This block adds the AWT image to the reference section. Currently has the flag for Time but change to AWT for when AWT is 
        # actually being calculated (it was not for my testing)
        if vr_flag == 'Time':
            plot_string_ = "./reference/AWT_image"
            plot_string = os.path.abspath(plot_string_)
            plot_file_string = plot_string + ''
            plot_string = replace_backslash_with_forward_slash(plot_string) + '.pdf'
            vr_subtext += '![](' +  plot_string + ')\n\n'
            
        vr_subtext += build_validation_reference_section('', vr_filename_1, vr_filename_2, vr_filename_3)
        vr_subtext += add_collapsible_segment_end()
        vr_flag_dict[vr_flag] = False
    return vr_subtext, vr_flag_dict


### CONVERT MARKDOWN TO HTML 
def get_image_string(original_string):
    # COUNT NUMBER OF PARENTHESES
    left_parentheses = original_string.count('(')
    right_parentheses = original_string.count(')')
    if (left_parentheses > 1) or (right_parentheses > 1):
        # FIND LAST RIGHT PARENTHESES
        last_right_parentheses_index = -(original_string[::-1].index(')') + 1)
        result = original_string[:last_right_parentheses_index]
        result = result.split('![](')[1]
    else:
        left = original_string.split('![](')[1]
        result = left.split(')')[0]
    return result
    
def convert_tables_html(text):
    new_text = ''
    outside_table = True
    first = False
    for i in range(0, len(text)):
        line = text[i]
        if line == '' and outside_table: # NOT IN A TABLE
            table = ''
            outside_table = True
        if len(line) > 0:
            if line[0] == '|':
                outside_table = False
                table += line + '\n'
            else:
                outside_table = True
                first = True
        if first:
            if 'table' in list(locals().keys()):
                table_html = markdown.markdown(table, extensions=['markdown.extensions.tables'])
                new_text += table_html + '\n'
                del table
            first = False    
        if outside_table:
            new_text += line + '\n'
    new_text = new_text.split('\n')
    return new_text

def convert_plots_html(text):
    new_text = ''
    for i in range(0, len(text)):
        line = text[i]
        if line == '':
            pass
        else:
            if line[0] == '!':
                image_filename_plot_path = get_image_string(line)
                # CHECK IMAGE DIMENSIONS
                image_filename = os.path.abspath(image_filename_plot_path)
                image_filename = replace_backslash_with_forward_slash(image_filename)
                image_filename = image_filename.replace('output/plots/', 'clayton_sphinxval/output/plots/')
                reader = pdf.PdfReader(image_filename)
                box = reader.pages[0].mediabox
                width = str(box.width)
                height = str(box.height + 60)
                new_text += '<embed src="' + image_filename_plot_path + '" alt="" height="' + height + '" width="' + width + '">'  
            else:
                new_text += line + '\n'
    new_text = new_text.split('\n')
    return new_text

def convert_bullets_html(text):
    new_text = ''
    for i in range(0, len(text)):
        line = text[i]
        if line == '':
            pass
        else:
            if '*' in line:
                line_text = line.split('*')[1]
                new_text += '<li>' + line_text + '</li>'
            else:
                new_text += line + '\n'
    new_text = new_text.split('\n')
    return new_text

def add_space_between_sections(text):
    # CHECK FOR BREAKS
    text = ''.join(text)
    text.replace('<br>', '')
    # ADD SPACE BETWEEN SECTIONS
    for i in range(0, len(text)):
        if text[i] == '</details>' or text[i] == '</details>\n':
            text[i] = '<br></details>'
    return text

def add_script(text):
    text = '''<script>
function openTab(evt, tabName) {
  // Declare all variables
  var i, tabcontent, tablinks;

  // Get all elements with class="tabcontent" and hide them
  tabcontent = document.getElementsByClassName("tabcontent");
  for (i = 0; i < tabcontent.length; i++) {
    tabcontent[i].style.display = "none";
  }

  // Get all elements with class="tablinks" and remove the class "active"
  tablinks = document.getElementsByClassName("tablinks");
  for (i = 0; i < tablinks.length; i++) {
    tablinks[i].className = tablinks[i].className.replace(" active", "");
  }

  // Show the current tab, and add an "active" class to the button that opened the tab
  document.getElementById(tabName).style.display = "block";
  evt.currentTarget.className += " active";
}
</script>\n\n''' + text
    return text

def add_style(text):
    text = '''<style>
table, th, td 
{
    border: 1px solid black;
    border-collapse: collapse;
}
html * 
{
    font-size: 16px;
    line-height: 1.25;
    color: #000000;
    font-family: Arial, sans-serif;
}
 
/* Style the tab */
.tab {
    overflow: hidden;
    border: 1px solid #ccc;
    background-color: #f1f1f1;
}

/* Style the buttons that are used to open the tab content */
.tab button {
    background-color: inherit;
    float: left;
    border: none;
    outline: none;
    cursor: pointer;
    padding: 14px 16px;
    transition: 0.3s;
}

/* Change background color of buttons on hover */
.tab button:hover {
    background-color: #ddd;
}

/* Create an active/current tablink class */
.tab button.active {
    background-color: #ccc;
}

/* Style the tab content */
.tabcontent {
    display: none;
    padding: 6px 12px;
    border: 1px solid #ccc;
    border-top: none;
}
	
.red {
    background-color: #fad5d2;
}

.green {
    background-color: #89d99e;
}
</style>\n\n''' + text
    return text

def add_title(model):
    return '<h1>' + model + ' Validation Report</h1>\n'

def add_tab(appendage, markdown_text, model):
    if appendage == '':
        appendage = 'All'
        default_string = 'style="display:block"'
    else:
        default_string = ''
    text = '<div id="' + appendage + '" class="tabcontent"' + default_string + '>\n'
    text += '    <h3>' + appendage + '</h3>\n'
    text += '    ' + convert_markdown_to_html(markdown_text, model + '...' + appendage, False) + '\n'
    text += '</div>\n'
    return text

def convert_markdown_to_html(text, model, validation_reference=False):
    
    if validation_reference:
        None
    else:
        print('Generating HTML report...' + model + '...' + str(datetime.datetime.now()))
    text = text.split('\n')
    
    # REPLACE TABLES
    text = convert_tables_html(text)
    
    # REPLACE IMAGES
    text = convert_plots_html(text)
        
    # REPLACE BULLETS
    text = convert_bullets_html(text)
    
    # ADD SPACE
    text = add_space_between_sections(text)
    
    text_final = ''
    for i in range(0, len(text)):
        text_final += text[i]
        
    # FINALIZE
    html = markdown.markdown(text_final)        
    return html
      
def get_html_report_preamble(model):
    text = add_script('')
    text += add_style('')
    text += add_title(model)
    return text

def get_plot_type(plot_string):
    if 'Time_Profile' in plot_string:
        plot_type = 'Time Profile'
    elif 'Correlation' in plot_string:
        plot_type = 'Correlation'
    elif 'ROC' in plot_string:
        plot_type = 'ROC Curve'
    else:
        plot_type = 'None'
    return plot_type
    
# FINAL RESULT
def report(output_dir, relative_path_plots): ### ADD OPTIONAL ARGUMENT HERE
    global output_dir__
    global relative_path_plots__

    # get all model metrics
    # analyze the output directory
    if output_dir is None:
        output_dir__ = config.outpath + '/pkl/'
    else:
        output_dir__ = output_dir   

    relative_path_plots__ = relative_path_plots

    files = os.listdir(output_dir__)
    if 'desktop.ini' in files:
        files.remove('desktop.ini')
    files.sort()
    
    # obtain sphinx dataframe
    sphinx_dataframe = pd.read_pickle(output_dir__ + 'SPHINX_dataframe.pkl')

    # grab all models
    models = list(set(sphinx_dataframe['Model']))
    models.sort()
    
    # define appendages (First, Last, Mean, Max, ...)
    appendages = ['', '_First', '_Last', '_Mean', '_Max']
    for i in range(0, len(models)):
        model = models[i]
        markdown_texts = {}
        appendage_set_list = []
        html_text = get_html_report_preamble(model)
        validation_reference_subtext_html = ''
        # define on/off flags for validation reference tables
        validation_reference_flag_dict = {'All Clear' : True,
                                          'AWT' : True,
                                          'Duration' : True,
                                          'Flux' : True,
                                          'Time' : True,
                                          'Probability' : True,
                                         }
        for j in range(0, len(appendages)):
        
            # check which sections to include
            all_clear = False
            awt = False
            duration = False
            peak_intensity = False
            peak_intensity_max = False
            peak_intensity_time = False
            peak_intensity_max_time = False
            threshold_crossing = False
            fluence = False
            max_flux_in_pred_win = False
            probability = False
            start_time = False
            duration = False
            end_time = False
            time_profile = False
            
            validation_reference_subtext = ''
            for k in range(0, len(files)):
                if appendages[j] in files[k]:
                    file_no_extension = files[k].rstrip('.pkl')
                    if file_no_extension[:-len(appendages[j])] == appendages[j]:
                        if ('all_clear_selections_' + model) in files[k]:
                            all_clear = True
                            continue
                        if ('awt_selections_' + model) in files[k]:
                            awt = True
                            continue
                        if ('duration_selections_' + model) in files[k]:
                            duration = True
                            continue
                        if ('peak_intensity_selections_' + model) in files[k]:
                            peak_intensity = True
                            continue
                        if ('peak_intensity_max_selections_' + model) in files[k]:
                            peak_intensity_max = True
                            continue
                        if ('peak_intensity_time_selections_' + model) in files[k]:
                            peak_intensity_time = True
                            continue
                        if ('peak_intensity_max_time_selections_' + model) in files[k]:
                            peak_intensity_max_time = True
                            continue
                        if ('threshold_crossing_time_selections_' + model) in files[k]:
                            threshold_crossing = True
                            continue       
                        if ('fluence_selections_' + model) in files[k]:
                            fluence = True
                            continue
                        if ('max_flux_in_pred_win_selections_' + model) in files[k]:
                            max_flux_in_pred_win = True
                            continue
                        if ('probability_selections_' + model) in files[k]:
                            probability = True
                            continue
                        if ('start_time_selections_' + model) in files[k]:
                            start_time = True
                            continue
                        if ('end_time_selections_' + model) in files[k]:
                            end_time = True
                            continue
                        if ('time_profile_selections_' + model) in files[k]:
                            time_profile = True
                            continue

            # preamble -- define colors and font and whatnot
            info_header = 'Report Information'
            info_text = 'Date of Report: ' + datetime.datetime.today().strftime('%Y-%m-%d' + 't' + '%H:%M:%S') + '<br>'
            info_text += 'Report generated by SPHINX<br>'
            info_text += 'This code may be publicly accessed at: ' + '[https://github.com/ktindiana/sphinxval](https://github.com/ktindiana/sphinxval)\n'
            if appendages[j] == '':
                title = model + ' Validation Report'
            else:
                title = model + ' ' + appendages[j].lstrip('_') + ' Validation Report'
            # info_text = '# ' + title + '\n\n' + define_colors() + add_collapsible_segment(info_header, info_text)
            
            validation_header = 'Validated Quantities'
            validation_text = 'This model was validated for the following quantities.\n\n'
            
            section_filename = ''
            markdown_text = ''
            report_exists = False
            if all_clear:
                ### build the all clear skill scores
                all_clear_filename = output_dir__ + 'all_clear_metrics' + appendages[j] + '.pkl'
                if os.path.exists(all_clear_filename):
                    validation_text += '* All Clear\n'
                    report_exists = True
                    appendage_set_list.append(appendages[j])
                    markdown_text += build_all_clear_skill_scores_section(all_clear_filename, model, sphinx_dataframe, appendage=appendages[j])
                validation_reference_subtext_string, validation_reference_flag_dict = construct_validation_reference_sheet(validation_reference_subtext, validation_reference_flag_dict, 'All Clear', 
                                                                                                                           config.referencepath + '/validation_reference_sheet_contingency_metrics.csv',
                                                                                                                           config.referencepath + '/validation_reference_sheet_contingency_skills.csv',
                                                                                                                           config.referencepath + '/validation_reference_sheet_contingency_plots.csv')
                validation_reference_subtext = validation_reference_subtext_string

            if awt:
                ### build the advanced warning time (AWT) metrics
                metric_label_start = 'Mean AWT for Predicted SEP All Clear to Observed SEP Threshold Crossing Time'
                section_title = 'Advanced Warning Time'
                section_tag = 'awt'
                metrics_description_string = 'N/A'
                section_filename = output_dir__ + section_tag + '_metrics' + appendages[j] + '.pkl'
                if os.path.exists(section_filename):
                    validation_text += '* ' + section_title + '\n'
                    report_exists = True
                    appendage_set_list.append(appendages[j])
                    markdown_text += build_section_awt(section_filename, model, sphinx_dataframe, metric_label_start, section_title, section_tag, metrics_description_string, appendage=appendages[j])
                validation_reference_subtext_string, validation_reference_flag_dict = construct_validation_reference_sheet(validation_reference_subtext, validation_reference_flag_dict, 'AWT', 
                                                                                                                           config.referencepath + '/validation_reference_sheet_awt_metrics.csv',
                                                                                                                           None,
                                                                                                                           None)
                validation_reference_subtext += validation_reference_subtext_string
                
            if peak_intensity:
                ### build the peak intensity metrics
                metric_label_start = 'Linear Regression Slope'
                section_title = 'Peak Intensity (Onset Peak)'
                section_tag = 'peak_intensity'
                metrics_description_string = "Correlation coefficients and regression lines indicate association.<br>Metrics involving error indicate bias. Positive values indicate model overprediction and negative values indicate model underprediction.<br>Metrics involving absolute error or squared error indicate accuracy.\n"
                section_filename = output_dir__ + section_tag + '_metrics' + appendages[j] + '.pkl'
                if os.path.exists(section_filename):
                    validation_text += '* ' + section_title + '\n'
                    report_exists = True
                    appendage_set_list.append(appendages[j])
                    markdown_text += build_section(section_filename, model, sphinx_dataframe, metric_label_start, section_title, section_tag, metrics_description_string, appendage=appendages[j])
                validation_reference_subtext_string, validation_reference_flag_dict = construct_validation_reference_sheet(validation_reference_subtext, validation_reference_flag_dict, 'Flux', 
                                                                                                                           config.referencepath + '/validation_reference_sheet_flux_metrics.csv',
                                                                                                                           None,
                                                                                                                           config.referencepath + '/validation_reference_sheet_flux_plots.csv')
                validation_reference_subtext = validation_reference_subtext_string
                
            if peak_intensity_max:
                ### build the peak intensity max metrics
                metric_label_start = 'Linear Regression Slope'
                section_title = 'Peak Intensity Max (Max Flux)'
                section_tag = 'peak_intensity_max'
                metrics_description_string = "Correlation coefficients and regression lines indicate association.<br>Metrics involving error indicate bias. Positive values indicate model overprediction and negative values indicate model underprediction.<br>Metrics involving absolute error or squared error indicate accuracy.\n"
                section_filename = output_dir__ + section_tag + '_metrics' + appendages[j] + '.pkl'
                if os.path.exists(section_filename):
                    validation_text += '* ' + section_title + '\n'
                    report_exists = True
                    appendage_set_list.append(appendages[j])
                    markdown_text += build_section(section_filename, model, sphinx_dataframe, metric_label_start, section_title, section_tag, metrics_description_string, appendage=appendages[j])
                validation_reference_subtext_string, validation_reference_flag_dict = construct_validation_reference_sheet(validation_reference_subtext, validation_reference_flag_dict, 'Flux', 
                                                                                                                           config.referencepath + '/validation_reference_sheet_flux_metrics.csv',
                                                                                                                           None,
                                                                                                                           config.referencepath + '/validation_reference_sheet_flux_plots.csv')
                validation_reference_subtext = validation_reference_subtext_string
                
            if peak_intensity_time:
                ### build the peak intensity time metrics
                metric_label_start = 'Mean Error (pred - obs)'
                section_title = 'Peak Intensity (Onset Peak) Time'
                section_tag = 'peak_intensity_time'
                metrics_description_string = "Metrics for Predicted Time - Observed Time are in hours.<br>Negative values indicate predicted time is earlier than observed.<br>Positive values indicate predicted time is later than observed.\n"
                section_filename = output_dir__ + section_tag + '_metrics' + appendages[j] + '.pkl'
                if os.path.exists(section_filename):
                    validation_text += '* ' + section_title + '\n'
                    report_exists = True
                    appendage_set_list.append(appendages[j])
                    markdown_text += build_section(section_filename, model, sphinx_dataframe, metric_label_start, section_title, section_tag, metrics_description_string, appendage=appendages[j])
                validation_reference_subtext_string, validation_reference_flag_dict = construct_validation_reference_sheet(validation_reference_subtext, validation_reference_flag_dict, 'Time', 
                                                                                                                           config.referencepath + '/validation_reference_sheet_time_metrics.csv',
                                                                                                                           None,
                                                                                                                           None)
                validation_reference_subtext = validation_reference_subtext_string
           
            if peak_intensity_max_time:
                ### build the peak intensity max time metrics
                metric_label_start = 'Mean Error (pred - obs)'
                section_title = 'Peak Intensity Max (Max Flux) Time'
                section_tag = 'peak_intensity_max_time'
                metrics_description_string = "Metrics for Predicted Time - Observed Time are in hours.<br>Negative values indicate predicted time is earlier than observed.<br>Positive values indicate predicted time is later than observed.\n"
                section_filename = output_dir__ + section_tag + '_metrics' + appendages[j] + '.pkl'
                if os.path.exists(section_filename):
                    validation_text += '* ' + section_title + '\n'
                    report_exists = True
                    appendage_set_list.append(appendages[j])
                    markdown_text += build_section(section_filename, model, sphinx_dataframe, metric_label_start, section_title, section_tag, metrics_description_string, appendage=appendages[j])
                validation_reference_subtext_string, validation_reference_flag_dict = construct_validation_reference_sheet(validation_reference_subtext, validation_reference_flag_dict, 'Time', 
                                                                                                                           config.referencepath + '/validation_reference_sheet_time_metrics.csv',
                                                                                                                           None,
                                                                                                                           None)
                validation_reference_subtext = validation_reference_subtext_string
            
            if threshold_crossing:
                ### build the threshold crossing metrics
                metric_label_start = 'Mean Error (pred - obs)'
                section_title = 'Threshold Crossing Time'
                section_tag = 'threshold_crossing_time'
                alt_section_tag = 'threshold_crossing'
                metrics_description_string = "Metrics for Predicted Time - Observed Time are in hours.<br>Negative values indicate predicted time is earlier than observed.<br>Positive values indicate predicted time is later than observed.\n"
                section_filename = output_dir__ + alt_section_tag + '_metrics' + appendages[j] + '.pkl'
                if os.path.exists(section_filename):
                    validation_text += '* ' + section_title + '\n'
                    report_exists = True
                    appendage_set_list.append(appendages[j])
                    markdown_text += build_section(section_filename, model, sphinx_dataframe, metric_label_start, section_title, section_tag, metrics_description_string, appendage=appendages[j])
                validation_reference_subtext_string, validation_reference_flag_dict = construct_validation_reference_sheet(validation_reference_subtext, validation_reference_flag_dict, 'Time', 
                                                                                                                           config.referencepath + '/validation_reference_sheet_time_metrics.csv',
                                                                                                                           None,
                                                                                                                           None)
                validation_reference_subtext = validation_reference_subtext_string
            
            if fluence:
                ### build the fluence metrics
                metric_label_start = 'Linear Regression Slope'
                section_title = 'Fluence'
                section_tag = 'fluence'
                metrics_description_string = "Correlation coefficients and regression lines indicate association.<br>Metrics involving error indicate bias. Positive values indicate model overprediction and negative values indicate model underprediction.<br>Metrics involving absolute error or squared error indicate accuracy.\n"
                section_filename = output_dir__ + section_tag + '_metrics' + appendages[j] + '.pkl'
                if os.path.exists(section_filename):
                    validation_text += '* ' + section_title + '\n'
                    report_exists = True
                    appendage_set_list.append(appendages[j])
                    markdown_text += build_section(section_filename, model, sphinx_dataframe, metric_label_start, section_title, section_tag, metrics_description_string, appendage=appendages[j])
                validation_reference_subtext_string, validation_reference_flag_dict = construct_validation_reference_sheet(validation_reference_subtext, validation_reference_flag_dict, 'Flux', 
                                                                                                                           config.referencepath + '/validation_reference_sheet_flux_metrics.csv',
                                                                                                                           None,
                                                                                                                           config.referencepath + '/validation_reference_sheet_flux_plots.csv')
                validation_reference_subtext = validation_reference_subtext_string
            
            if max_flux_in_pred_win:
                ### build the maximum flux in prediction window metrics
                metric_label_start = 'Linear Regression Slope'
                section_title = 'Max Flux in Prediction Window'
                section_tag = 'max_flux_in_pred_win'
                metrics_description_string = "Correlation coefficients and regression lines indicate association.<br>Metrics involving error indicate bias. Positive values indicate model overprediction and negative values indicate model underprediction.<br>Metrics involving absolute error or squared error indicate accuracy.\n"
                section_filename = output_dir__ + section_tag + '_metrics' + appendages[j] + '.pkl'
                if os.path.exists(section_filename):
                    validation_text += '* ' + section_title + '\n'
                    report_exists = True
                    appendage_set_list.append(appendages[j])
                    markdown_text += build_section(section_filename, model, sphinx_dataframe, metric_label_start, section_title, section_tag, metrics_description_string, appendage=appendages[j])
                validation_reference_subtext_string, validation_reference_flag_dict = construct_validation_reference_sheet(validation_reference_subtext, validation_reference_flag_dict, 'Flux', 
                                                                                                                           config.referencepath + '/validation_reference_sheet_flux_metrics.csv',
                                                                                                                           None,
                                                                                                                           config.referencepath + '/validation_reference_sheet_flux_plots.csv')
                validation_reference_subtext = validation_reference_subtext_string
                
            
            if probability:    
                ### build the probability metrics
                metric_label_start = 'Brier Score'
                section_title = 'Probability'
                section_tag = 'probability'
                metrics_description_string = ""
                section_filename = output_dir__ + section_tag + '_metrics' + appendages[j] + '.pkl'
                if os.path.exists(section_filename):
                    validation_text += '* ' + section_title + '\n'
                    report_exists = True
                    appendage_set_list.append(appendages[j])
                    markdown_text += build_section(section_filename, model, sphinx_dataframe, metric_label_start, section_title, section_tag, metrics_description_string, appendage=appendages[j])
                validation_reference_subtext_string, validation_reference_flag_dict = construct_validation_reference_sheet(validation_reference_subtext, validation_reference_flag_dict, 'Probability', 
                                                                                                                           config.referencepath + '/validation_reference_sheet_probability_metrics.csv',
                                                                                                                           config.referencepath + '/validation_reference_sheet_probability_skills.csv',
                                                                                                                           config.referencepath + '/validation_reference_sheet_probability_plots.csv')
                validation_reference_subtext = validation_reference_subtext_string
                
            if start_time:
                ### build the start time metrics
                metric_label_start = 'Mean Error (pred - obs)'
                section_title = 'Start Time'
                section_tag = 'start_time'
                metrics_description_string = "Metrics for Predicted Time - Observed Time are in hours.<br>Negative values indicate predicted time is earlier than observed.<br>Positive values indicate predicted time is later than observed.\n"
                section_filename = output_dir__ + section_tag + '_metrics' + appendages[j] + '.pkl'
                if os.path.exists(section_filename):
                    validation_text += '* ' + section_title + '\n'
                    report_exists = True
                    appendage_set_list.append(appendages[j])
                    markdown_text += build_section(section_filename, model, sphinx_dataframe, metric_label_start, section_title, section_tag, metrics_description_string, appendage=appendages[j])
                validation_reference_subtext_string, validation_reference_flag_dict = construct_validation_reference_sheet(validation_reference_subtext, validation_reference_flag_dict, 'Time', 
                                                                                                                           config.referencepath + '/validation_reference_sheet_time_metrics.csv',
                                                                                                                           None,
                                                                                                                           None)
                validation_reference_subtext = validation_reference_subtext_string
            
            if duration:
                ### build the duration metrics
                metric_label_start = 'Mean Error (pred - obs)'
                section_title = 'Duration'
                section_tag = 'duration'
                metrics_description_string = "Duration is calculated in hours.<br> Metrics involving error indicate bias. Positive values indicate model overprediction and negative values indicate model underprediction.<br>Metrics involving absolute error indicate accuracy.\n"
                section_filename = output_dir__ + section_tag + '_metrics' + appendages[j] + '.pkl'
                if os.path.exists(section_filename):
                    validation_text += '* ' + section_title + '\n'
                    report_exists = True
                    appendage_set_list.append(appendages[j])
                    markdown_text += build_section(section_filename, model, sphinx_dataframe, metric_label_start, section_title, section_tag, metrics_description_string, appendage=appendages[j])
                validation_reference_subtext_string, validation_reference_flag_dict = construct_validation_reference_sheet(validation_reference_subtext, validation_reference_flag_dict, 'Duration', 
                                                                                                                           config.referencepath + '/validation_reference_sheet_time_metrics.csv',
                                                                                                                           None,
                                                                                                                           None)
                validation_reference_subtext = validation_reference_subtext_string
              
            if end_time:
                ### build the end time metrics
                metric_label_start = 'Mean Error (pred - obs)'
                section_title = 'End Time'
                section_tag = 'end_time'
                metrics_description_string = "Metrics for Predicted Time - Observed Time are in hours.<br>Negative values indicate predicted time is earlier than observed.<br>Positive values indicate predicted time is later than observed.\n"
                section_filename = output_dir__ + section_tag + '_metrics' + appendages[j] + '.pkl'
                if os.path.exists(section_filename):
                    validation_text += '* ' + section_title + '\n'
                    report_exists = True
                    appendage_set_list.append(appendages[j])
                    markdown_text += build_section(section_filename, model, sphinx_dataframe, metric_label_start, section_title, section_tag, metrics_description_string, appendage=appendages[j])
                validation_reference_subtext_string, validation_reference_flag_dict = construct_validation_reference_sheet(validation_reference_subtext, validation_reference_flag_dict, 'Time', 
                                                                                                                           config.referencepath + '/validation_reference_sheet_time_metrics.csv',
                                                                                                                           None,
                                                                                                                           None)
                validation_reference_subtext = validation_reference_subtext_string
                    
            if time_profile:
                ### build the time profile metrics
                metric_label_start = 'Linear Regression Slope'
                section_title = 'Time Profile'
                section_tag = 'time_profile'            
                metrics_description_string = "Correlation plots are created fro each predicted time profile and may be viewed in the output/plots directory.<br>Metrics are calculated from overlapping portions of predicted and observed time profiles, highlighted in red and orange in the Time Profile plots.<br>Metrics involving error indicate bias. Positive values indicate model overprediction and negative values indicate model underprediction.<br>Metrics involving absolute error or squared error indicate accuracy.\n"
                skip_label_list = ['Time Profile Selection Plot']
                section_filename = output_dir__ + section_tag + '_metrics' + appendages[j] + '.pkl'
                if os.path.exists(section_filename):
                    validation_text += '* ' + section_title + '\n'
                    report_exists = True
                    appendage_set_list.append(appendages[j])
                    markdown_text += build_section(section_filename, model, sphinx_dataframe, metric_label_start, section_title, section_tag, metrics_description_string, skip_label_list=skip_label_list, appendage=appendages[j])        
                validation_reference_subtext_string, validation_reference_flag_dict = construct_validation_reference_sheet(validation_reference_subtext, validation_reference_flag_dict, 'Flux', 
                                                                                                                           config.referencepath + '/validation_reference_sheet_flux_metrics.csv',
                                                                                                                           None,
                                                                                                                           config.referencepath + '/validation_reference_sheet_flux_plots.csv')
                validation_reference_subtext = validation_reference_subtext_string
            
            ### BUILD THE VALIDATION REFERENCE SHEET AND FINALIZE
            validation_text = add_collapsible_segment(validation_header, validation_text)
            markdown_text = info_text + validation_text + markdown_text
            markdown_filename = config.reportpath + '/' + model + '_report' + appendages[j] + '.md'
            appendage_set_list = list(set(appendage_set_list))

            validation_reference_text = add_collapsible_segment_start('Validation Reference Sheet', '')
            validation_reference_text += validation_reference_subtext
            validation_reference_text += add_collapsible_segment_end()
            validation_reference_subtext_html += validation_reference_subtext
            
            if report_exists:
                a = open(markdown_filename, 'w')
                a.write(markdown_text + validation_reference_text)
                a.close()
                markdown_texts[appendages[j]] = markdown_text
        
        
        html_text += '<div class="tab">\n'
        html_text += '    <button class="tablinks" onclick="openTab(event, \'All\')">All</button>\n'
        for j in range(0, len(appendage_set_list)):
            if appendage_set_list[j] == '':
                None
            else:
                html_text += '    <button class="tablinks" onclick="openTab(event, \'' + appendage_set_list[j].replace('_', '') + '\')">' + appendage_set_list[j].replace('_', '') + '</button>\n'
        html_text += '</div>\n'
        
        
        for j in range(0, len(appendage_set_list)):
            html_text += add_tab(appendage_set_list[j].replace('_', ''), markdown_texts[appendage_set_list[j]], model)
        
        
        validation_reference_text_html = add_collapsible_segment_start('Validation Reference Sheet', '')
        validation_reference_text_html += validation_reference_subtext_html
        validation_reference_text_html += add_collapsible_segment_end()
        html_text += convert_markdown_to_html(validation_reference_text_html, model)
        html_filename = config.reportpath + '/' + model + '_report.html'
        a = open(html_filename, 'w')
        a.write(html_text)
        a.close()
        print('    Complete')
        
        
        
            
                
        
