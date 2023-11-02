### SPHINX MARKDOWN GENERATOR -- please ignore the mess for now

import pandas as pd
import numpy as np
pd.set_option('mode.chained_assignment', None)
import datetime
import os
import pickle

import markdown
import PyPDF2 as pdf

from . import config 

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
    markdown += text
    markdown += '</details>\n'
    return markdown

def add_collapsible_segment_start(header, text):
    markdown = '<details>\n'
    markdown += '<summary>' + header + '</summary>\n\n'
    markdown += text
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

def build_info_events_table(filename, sphinx_dataframe, subset_list, subset_replacement_dict):
    data = pd.read_pickle(filename)
    subset = data[subset_list]
    subset.insert(0, 'Observatory', 'dummy')
    selection_index = list(data.index)
    subset['Observatory'] = sphinx_dataframe.loc[selection_index, 'Observatory'].to_list()
    subset = subset.rename(columns=subset_replacement_dict)
    output = '\n' + subset.to_markdown(index=False) + '\n'
    n_events = len(data)
    return output, n_events

def build_threshold_string(data, i):
    energy_threshold_data = data.iloc[i]['Energy Channel']
    print(energy_threshold_data)
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
    print(mismatch_allowed_string)
    obs_threshold = data.iloc[i]['Threshold'].split('.units.')[0].split('threshold.')[1] + ' pfu'
    if float(obs_threshold.split(' pfu')[0]) < 1:
        obs_threshold = '0'
    pred_threshold = data.iloc[i]['Prediction Threshold'].split('.units.')[0].split('threshold.')[1] + ' pfu'
    threshold_string = '* Energy Channel: ' + energy_threshold + '\n'
    threshold_string += '* Observation Threshold: ' + obs_threshold + '\n'
    threshold_string += '* Predictions Threshold: ' + pred_threshold + '\n'
    return threshold_string, energy_threshold, obs_threshold, pred_threshold, mismatch_allowed_string

def build_all_clear_skill_scores_section(filename, model, sphinx_dataframe):
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
        print(mismatch_allowed_string)
        hits = data.iloc[i]["All Clear 'True Positives' (Hits)"]
        false_alarms = data.iloc[i]["All Clear 'False Positives' (False Alarms)"]
        correct_negatives = data.iloc[i]["All Clear 'True Negatives' (Correct Negatives)"]
        misses = data.iloc[i]["All Clear 'False Negatives' (Misses)"]
        contingency_table_values = [hits, false_alarms, correct_negatives, misses]
        contingency_table_string = build_contingency_table(*contingency_table_values)
        info_string = 'Instruments and SEP events used in validation<br>'
        info_string += 'N = ' + str(sum(contingency_table_values)) + '<br>'
        info_string += '...\n' # need to complete
        
        selections_filename = output_dir__ + 'all_clear_selections_' + model + '_' + data.iloc[i]['Energy Channel'] + '_threshold_' + obs_threshold.rstrip(' pfu') + mismatch_allowed_string + '.pkl'
        info_string_, n_events = build_info_events_table(selections_filename, sphinx_dataframe, [], {})
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

def build_info_events_table_peak_intensity(filename, sphinx_dataframe):
    data = pd.read_pickle(filename)
    subset = data[['Prediction Window Start', 'Prediction Window End', 'Observed SEP Peak Intensity (Onset Peak)', 'Predicted SEP Peak Intensity (Onset Peak)']]
    subset.insert(0, 'Observatory', 'dummy')
    selection_index = list(data.index)
    subset['Observatory'] = sphinx_dataframe.loc[selection_index, 'Observatory'].to_list()
    subset = subset.rename(columns={'Observed SEP Peak Intensity (Onset Peak)' : 'Observations', 'Predicted SEP Peak Intensity (Onset Peak)' : 'Predictions'})
    subset = subset
    output = '\n' + subset.to_markdown(index=False) + '\n'
    n_events = len(data)
    return output, n_events

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
    plot_string = ''
    plot_string_list = []
    for i in range(0, len(column_labels)):
        plot_index = None
        if type(column_labels[i]) == str:
            if ('plot' in column_labels[i]) or ('plot' in column_labels[i].lower()):
                plot_index = i * 1
                
            if (not (plot_index is None)):
                plot_path = data.iloc[current_index][column_labels[plot_index]]
                if plot_path == '':
                    plot_string = ''
                else:
                    output_directory = config.outpath
                    if '.' in output_directory:
                        if output_directory[0] == '.':
                            output_directory = output_directory[1:]
                    plot_path = output_directory + '/plots/' + plot_path.split('plots')[1] 
                    full_path = os.getcwd().replace('\\', '/') + plot_path
                    full_path = full_path.replace('//', '/') + '.pdf'
                    if os.path.exists(full_path) and plot_path != '':
                        plot_string = '![](' + full_path + ')\n\n'
                        plot_string_list.append(plot_string)
                    else:
                        plot_string = ''
    if plot_string == '':
        if 'plot_path' in locals():
            if plot_path == '':
                None
            else:
                print('Not displaying ' + plot_path)
        plot_string = 'No image files found.\n\n'
        plot_string_list.append(plot_string)
    return metrics_table_string, plot_string_list

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

def build_section(filename, model, sphinx_dataframe, metric_label_start, section_title, section_tag, metrics_description_string, skip_label_list=[], rename_dict={}):
    data = pd.read_pickle(filename)
    data = data[data.Model == model]
    column_labels = data.columns    
    metric_index_start = list(column_labels).index(metric_label_start)
    text = ''
    number_rows = data.shape[0]
    if number_rows > 0:
        text += add_collapsible_segment_start(section_title + ' Metrics', '')
    for i in range(0, number_rows):
        threshold_string, energy_threshold, obs_threshold, pred_threshold, mismatch_allowed_string = build_threshold_string(data, i)
        print(mismatch_allowed_string)
        selections_filename = output_dir__ + section_tag + '_selections_' + model + '_' + data.iloc[i]['Energy Channel'] + '_threshold_' + obs_threshold.rstrip(' pfu') + mismatch_allowed_string + '.pkl'
        subset_list = ['Prediction Window Start', 'Prediction Window End']
        subset_list = append_subset_list(selections_filename, subset_list, 'Prediction Window End', 'Units')
        info_string_, n_events = build_info_events_table(selections_filename, sphinx_dataframe, subset_list, rename_dict)
        info_string = 'Instruments and SEP events used in validation<br>'
        info_string += 'N = ' + str(n_events) + '<br>'
        info_string += '...\n' # need to complete
        info_string += info_string_
        metrics_string = metrics_description_string + '' 
        metrics_string_, plot_string_list = build_metrics_table(data, i, metric_index_start, skip_label_list)
        metrics_string += metrics_string_
        text += add_collapsible_segment_start(energy_threshold, '')
        text += add_collapsible_segment('Thresholds Applied', threshold_string)
        text += add_collapsible_segment('Validation Info', info_string)
        text += add_collapsible_segment('Metrics', metrics_string)
        for j in range(0, len(plot_string_list)):
            plot_type = get_plot_type(plot_string_list[j])
            text += add_collapsible_segment('Plot: ' + plot_type, plot_string_list[j])
        text += add_collapsible_segment_end()
    text += add_collapsible_segment_end()
    return text
    
def build_validation_reference_section(filename1, filename2):
    text = ''
    text += add_collapsible_segment_start('Validation Reference', '')
    data = pd.read_csv(filename1, skiprows=1)
    table = '\n' + data.to_markdown(index=False) + '\n'
    text += add_collapsible_segment('Metrics', table)
    data = pd.read_csv(filename2, skiprows=1)
    table = '\n' + data.to_markdown(index=False) + '\n'
    text += add_collapsible_segment('Skill Scores', table)
    text += add_collapsible_segment_end()
    return text



### CONVERT MARKDOWN TO HTML 
def get_image_string(original_string):
    left = original_string.split('![](')[1]
    result = left.split(')')[0]
    result = result.replace('.png', '.pdf')
    return result

def convert_tables_html(text):
    new_text = ''
    outside_table = True
    for i in range(0, len(text)):
        line = text[i]
        if line[0] == '|' and outside_table:
            outside_table = False
            table = ''            
        if line[0] != '|':
            if (not outside_table):    
                if 'table' in locals():
                    table_html = markdown.markdown(table, extensions=['markdown.extensions.tables'])
                    new_text += table_html + '\n'
            outside_table = True
        if (not outside_table):
            table += line   
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
                image_filename = get_image_string(line)
                # CHECK IMAGE DIMENSIONS
                reader = pdf.PdfReader(image_filename)
                box = reader.pages[0].mediabox
                width = str(box.width)
                height = str(box.height + 60)
                new_text += '<embed src="' + image_filename + '" alt="" height="' + height + '" width="' + width + '">'  
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
              </style>''' + text
    return text

def convert_markdown_to_html(filename, size_limit=10**20):
    
    # CHECK HOW BIG THIS FILE IS
    if os.path.getsize(filename) > size_limit:
        print('This Markdown report is too big to convert to HTML: ', filename)
    else:
        print('Generating HTML report: ', filename.replace('.md', '.html'))
        # CONVERT MARKDOWN TO HTML
        a = open(filename, 'r')
        text = a.readlines()
        a.close()
    
        # REPLACE TABLES
        text = convert_tables_html(text)
    
        # REPLACE IMAGES
        text = convert_plots_html(text)
    
        # REPLACE BULLETS
        text = convert_bullets_html(text)
    
        # ADD SPACE
        text = add_space_between_sections(text)

        # ADD STYLE
        text = add_style(text)

        # CONDENSE INTO ONE STRING
        text_final = ''
        for i in range(0, len(text)):
            text_final += text[i]
    
        html = markdown.markdown(text_final)
        a = open(filename.replace('.md', '.html'), 'w')
        a.write(html)
        a.close()
        print('    Complete')

def get_plot_type(plot_string):
    if 'Time_Profile' in plot_string:
        plot_type = 'Time Profile'
    elif 'Correlation' in plot_string:
        plot_type = 'Correlation'
    else:
        plot_type = 'None'
    return plot_type
    
# FINAL RESULT
def report(output_dir):
    global output_dir__

    # get all model metrics
    # analyze the output directory
    if output_dir is None:
        output_dir__ = config.outpath + '/pkl/'
    else:
        output_dir__ = output_dir   

    files = os.listdir(output_dir__)
    if 'desktop.ini' in files:
        files.remove('desktop.ini')
    files.sort()
    
    # obtain sphinx dataframe
    a = open(output_dir__ + 'sphinx_dataframe.pkl', 'rb')
    sphinx_dataframe = pickle.load(a)
    a.close()

    # grab all models
    models = list(set(sphinx_dataframe['Model']))
    models.sort()
            
    
    for i in range(0, len(models)):
        model = models[i]
   
        # check which sections to include
        all_clear = False
        peak_intensity = False
        peak_intensity_max = False
        peak_intensity_time = False
        threshold_crossing = False
        fluence = False
        max_flux_in_pred_win = False
        probability = False
        start_time = False
        duration = False
        end_time = False
        time_profile = False
        
        for j in range(0, len(files)):
            if ('all_clear_selections_' + model) in files[j]:
                all_clear = True
                continue
            if ('peak_intensity_selections_' + model) in files[j]:
                peak_intensity = True
                continue
            if ('peak_intensity_max_selections_' + model) in files[j]:
                peak_intensity_max = True
                continue
            if ('peak_intensity_max_time_selections_' + model) in files[j]:
                peak_intensity_time = True
                continue
            if ('threshold_crossing_time_selections_' + model) in files[j]:
                threshold_crossing = True
                continue       
            if ('fluence_selections_' + model) in files[j]:
                fluence = True
                continue
            if ('max_flux_in_pred_win_selections_' + model) in files[j]:
                max_flux_in_pred_win = True
                continue
            if ('probability_selections_' + model) in files[j]:
                probability = True
                continue
            if ('start_time_selections_' + model) in files[j]:
                start_time = True
                continue
            if ('duration_selections_' + model) in files[j]:
                duration = True
                continue
            if ('end_time_selections_' + model) in files[j]:
                end_time = True
                continue
            if ('time_profile_selections_' + model) in files[j]:
                time_profile = True
                continue
 
        # preamble -- define colors and font and whatnot
        info_header = 'Report Information'
        info_text = 'Date of Report: ' + datetime.datetime.today().strftime('%Y-%m-%d' + 't' + '%H:%M:%S') + '<br>'
        info_text += 'Report generated by sep-validation > validation.py<br>'
        info_text += 'This code may be publicly accessed at: ' + '[https://github.com/ktindiana/sphinxval](https://github.com/ktindiana/sphinxval)\n'
        title = model + ' Validation Report'
        info_text = '# ' + title + '\n\n' + define_colors() + add_collapsible_segment(info_header, info_text)
        
        validation_header = 'Validated Quantities'
        validation_text = 'This model was validated for the following quantities. If the model does not make predictions for any of these quantities, they will not be included in the report.\n\n'
        
        markdown_text = ''
        if all_clear:
            ### build the all clear skill scores
            all_clear_filename = output_dir__ + 'all_clear_metrics.pkl'
            validation_text += '* All Clear\n'
            markdown_text += build_all_clear_skill_scores_section(all_clear_filename, model, sphinx_dataframe)

        if peak_intensity:
            ### build the peak intensity metrics
            metric_label_start = 'Linear Regression Slope'
            section_title = 'Peak Intensity'
            section_tag = 'peak_intensity'
            metrics_description_string = "Metrics for $log_{10}$(model) - $log_{10}$(Observations).<br>Positive values indicate model overprediction.<br>Negative values indicate model underprediction.<br>r_lin and r_log indicate the pearson's correlation coefficient calculated using values or $log_{10}$(values), respectively."
            section_filename = output_dir__ + section_tag + '_metrics.pkl'
            validation_text += '* ' + section_title + '\n'
            markdown_text += build_section(section_filename, model, sphinx_dataframe, metric_label_start, section_title, section_tag, metrics_description_string)
            
        if peak_intensity_max:
            ### build the peak intensity max metrics
            metric_label_start = 'Linear Regression Slope'
            section_title = 'Peak Intensity Max'
            section_tag = 'peak_intensity_max'
            metrics_description_string = "Metrics for $log_{10}$(model) - $log_{10}$(Observations).<br>Positive values indicate model overprediction.<br>Negative values indicate model underprediction.<br>r_lin and r_log indicate the pearson's correlation coefficient calculated using values or $log_{10}$(values), respectively."
            section_filename = output_dir__ + section_tag + '_metrics.pkl'
            validation_text += '* ' + section_title + '\n'
            markdown_text += build_section(section_filename, model, sphinx_dataframe, metric_label_start, section_title, section_tag, metrics_description_string)
            
        if peak_intensity_time:
            ### build the peak intensity time metrics
            metric_label_start = 'Mean Error (pred - obs)'
            section_title = 'Peak Intensity Time'
            section_tag = 'peak_intensity_time'
            metrics_description_string = "Metrics for Observed Time - Predicted Time are in hours.<br>Negative values indicate predicted time is later than observed.<br>Positive values indicate predicted time is earlier than observed.\n"
            section_filename = output_dir__ + section_tag + '_metrics.pkl'
            validation_text += '* ' + section_title + '\n'
            markdown_text += build_section(section_filename, model, sphinx_dataframe, metric_label_start, section_title, section_tag, metrics_description_string)
        
        if threshold_crossing:
            ### build the threshold crossing metrics
            metric_label_start = 'Mean Error (pred - obs)'
            section_title = 'Threshold Crossing Time'
            section_tag = 'threshold_crossing_time'
            alt_section_tag = 'threshold_crossing'
            metrics_description_string = "Metrics for Observed Time - Predicted Time are in hours.<br>Negative values indicate predicted time is later than observed.<br>Positive values indicate predicted time is earlier than observed.\n"
            section_filename = output_dir__ + alt_section_tag + '_metrics.pkl'
            validation_text += '* ' + section_title + '\n'
            markdown_text += build_section(section_filename, model, sphinx_dataframe, metric_label_start, section_title, section_tag, metrics_description_string)
        

        if fluence:
            ### build the fluence metrics
            metric_label_start = 'Linear Regression Slope'
            section_title = 'Fluence'
            section_tag = 'fluence'
            metrics_description_string = "Metrics for $log_{10}$(model) - $log_{10}$(Observations).<br>Positive values indicate model overprediction.<br>Negative values indicate model underprediction.<br>r_lin and r_log indicate the pearson's correlation coefficient calculated using values or $log_{10}$(values), respectively."
            section_filename = output_dir__ + section_tag + '_metrics.pkl'
            validation_text += '* ' + section_title + '\n'
            markdown_text += build_section(section_filename, model, sphinx_dataframe, metric_label_start, section_title, section_tag, metrics_description_string)
        
        
        if max_flux_in_pred_win:
            ### build the maximum flux in prediction window metrics
            metric_label_start = 'Linear Regression Slope'
            section_title = 'Max Flux in Prediction Window'
            section_tag = 'max_flux_in_pred_win'
            metrics_description_string = "Metrics for $log_{10}$(model) - $log_{10}$(Observations).<br>Positive values indicate model overprediction.<br>Negative values indicate model underprediction.<br>r_lin and r_log indicate the pearson's correlation coefficient calculated using values or $log_{10}$(values), respectively."
            section_filename = output_dir__ + section_tag + '_metrics.pkl'
            validation_text += '* ' + section_title + '\n'
            markdown_text += build_section(section_filename, model, sphinx_dataframe, metric_label_start, section_title, section_tag, metrics_description_string)
        

        if probability:    
            ### build the probability metrics
            metric_label_start = 'Brier Score'
            section_title = 'Probability'
            section_tag = 'probability'
            metrics_description_string = "Metrics for $log_{10}$(Model) - $log_{10}$(Observations).<br>Positive values indicate model overprediction.<br>Negative values indicate model underprediction.<br>r_lin and r_log indicate the pearson's correlation coefficient calculated using values or $log_{10}$(values), respectively."
            section_filename = output_dir__ + section_tag + '_metrics.pkl'
            validation_text += '* ' + section_title + '\n'
            markdown_text += build_section(section_filename, model, sphinx_dataframe, metric_label_start, section_title, section_tag, metrics_description_string)
        
        
        if start_time:
            ### build the start time metrics
            metric_label_start = 'Mean Error (pred - obs)'
            section_title = 'Start Time'
            section_tag = 'start_time'
            metrics_description_string = "Metrics for Observed Time - Predicted Time are in hours.<br>Negative values indicate predicted time is later than observed.<br>Positive values indicate predicted time is earlier than observed.\n"
            section_filename = output_dir__ + section_tag + '_metrics.pkl'
            validation_text += '* ' + section_title + '\n'
            markdown_text += build_section(section_filename, model, sphinx_dataframe, metric_label_start, section_title, section_tag, metrics_description_string)
        
        
        if duration:
            ### build the duration metrics
            metric_label_start = 'Mean Error (pred - obs)'
            section_title = 'Duration'
            section_tag = 'duration'
            metrics_description_string = "Metrics for Observed Time - Predicted Time are in hours.<br>Negative values indicate predicted time is later than observed.<br>Positive values indicate predicted time is earlier than observed.\n"
            section_filename = output_dir__ + section_tag + '_metrics.pkl'
            validation_text += '* ' + section_title + '\n'
            markdown_text += build_section(section_filename, model, sphinx_dataframe, metric_label_start, section_title, section_tag, metrics_description_string)
        
            
        if end_time:
            ### build the end time metrics
            metric_label_start = 'Mean Error (pred - obs)'
            section_title = 'End Time'
            section_tag = 'end_time'
            metrics_description_string = "Metrics for Observed Time - Predicted Time are in hours.<br>Negative values indicate predicted time is later than observed.<br>Positive values indicate predicted time is earlier than observed.\n"
            section_filename = output_dir__ + section_tag + '_metrics.pkl'
            validation_text += '* ' + section_title + '\n'
            markdown_text += build_section(section_filename, model, sphinx_dataframe, metric_label_start, section_title, section_tag, metrics_description_string)
        
        
        if time_profile:
            ### build the time profile metrics
            metric_label_start = 'Linear Regression Slope'
            section_title = 'Time Profile'
            section_tag = 'time_profile'
            metrics_description_string = "Metrics for Observed Time - Predicted Time are in hours.<br>Negative values indicate predicted time is later than observed.<br>Positive values indicate predicted time is earlier than observed.\n"
            section_filename = output_dir__ + section_tag + '_metrics.pkl'
            validation_text += '* ' + section_title + '\n'
            skip_label_list = ['Time Profile Selection Plot']
            markdown_text += build_section(section_filename, model, sphinx_dataframe, metric_label_start, section_title, section_tag, metrics_description_string, skip_label_list=skip_label_list)
        
            
        
        ### BUILD THE VALIDATION REFERENCE
        vr_filename1 = config.referencepath + '/validation_reference_sheet_1.csv'
        vr_filename2 = config.referencepath + '/validation_reference_sheet_2.csv'
        markdown_text += build_validation_reference_section(vr_filename1, vr_filename2)
        
        # FINALIZE
        validation_text = add_collapsible_segment(validation_header, validation_text)
        markdown_text = info_text + validation_text + markdown_text
        markdown_filename = config.reportpath + '/' + model + '_report.md'
        a = open(markdown_filename, 'w')
        a.write(markdown_text)
        a.close()

        convert_markdown_to_html(markdown_filename)




