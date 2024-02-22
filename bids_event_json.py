import csv
import json
import os
import glob

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def create_json_structure(reader):
    columns = next(reader)
    
    # Initialize a dictionary with an empty set for each column
    unique_values = {column: set() for column in columns}
    
    # Read through the TSV file to populate unique_values
    for row in reader:
        for column, value in zip(columns, row):
            if not is_number(value):
                unique_values[column].add(value)
    
    # Define the structure of your json files
    json_structure = {
        column: {
            "LongName": column,
            "Description": "",
            "Units": "",
            "Levels": {value: "" for value in unique_values[column]} if unique_values[column] else None
        }
        for column in columns
    }
    return json_structure

def write_json_file(tsv_file, json_structure, output_directory):
    # Create a single json file for the tsv file
    task = os.path.basename(tsv_file).split('_')[1]

    json_filename = f'task-{task}_events.json'
    with open(os.path.join(output_directory, json_filename), 'w') as jsonfile:
        json.dump(json_structure, jsonfile, indent=4)

# Define the name of your tsv file
tsv_files = glob.glob('/oak/stanford/groups/russpold/data/uh2/aim1/behavioral_data/event_files_sharing/s192_*_events.tsv')
for tsv_file in tsv_files:

    # Define the output directory
    output_directory = '/oak/stanford/groups/russpold/data/uh2/aim1/analysis_code/'

    # Read the tsv file
    with open(tsv_file, 'r') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        
        # Generate json_structure
        json_structure = create_json_structure(reader)

        # Create the output directory if it does not exist
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # Write json files
        write_json_file(tsv_file, json_structure, output_directory)
