import os
from sybil import Serie, Sybil
import csv
import pandas as pd
import argparse
from tqdm import tqdm
import traceback
import time

"""
This script is used to run inference on a set of 3d .mha files and save the results to a csv file.
"""

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run inference on a set of .mha files and save the results to a CSV file.')
    parser.add_argument('--parent-directory', type=str, required=True, help='Path to the parent directory containing .mha files')
    parser.add_argument('--mha-3d', type=bool, required=False, default=True, help = 'When inputting .mha files: True if the files are 3D, False for .mha slices ')
    parser.add_argument('--overview-csv', type=str, required=False, help='Path to the CSV file containing seriesinstenceuid')
    parser.add_argument('--output-csv', type=str, required=True, help='Path to the output CSV file')
    parser.add_argument('--log-file',type=str, required=True, help='Path to the output log file')
    return parser.parse_args()

args = parse_arguments()

ParentDirectory = args.parent_directory
myoverview = args.overview_csv
csvoutput = args.output_csv
log_file = args.log_file
mha3d = args.mha_3d

start_time_model = time.time()
def get_subfolder_paths(parent_folder):
    """Return a list of all subfolder paths in a parent folder.
    Parameters:
    parent_folder (str): The path to the parent folder.
    Returns:s
    List[str]: A list of paths for all subfolders in the parent folder.
    """
    subfolder_names = os.listdir(parent_folder)
    subfolder_paths = [os.path.join(parent_folder, name) for name in subfolder_names if os.path.isdir(os.path.join(parent_folder, name))]
    return subfolder_paths

def get_mha_paths_from_csv(csv_file_path,path_to_mha):
    """Return a list of all mha file paths extracted from a CSV file.
    
    Parameters:
    csv_file_path (str): The path to the CSV file.
    path_to_mha (str): The path to the folder containing the scans

    Returns:
    List[str]: A list of paths for all mha files extracted from the CSV file.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)
    series_instance_ids = df['SeriesInstanceUID']+".mha"
    subfolder_paths = [os.path.join(path_to_mha, str(series_instance_id)) for series_instance_id in series_instance_ids]
    return subfolder_paths

def get_mha_filepaths(folder_path):
    """Parse through a folder of .mha files of CT slices and return a list of all file paths.
    Parameters:
    folder_path (str): The path to the folder containing the .mha files.
    Returns:
    List[str]: A list of file paths for all .mha files in the folder.
    """
    mha_filepaths = []
    for root, _, files in os.walk(folder_path):
        for file in sorted(files):  # Sort the files list
            if file.endswith(".mha") or file.endswith(".dcm"):
                mha_filepaths.append(os.path.join(root, file))
    return mha_filepaths


# Initialize an empty dictionary
data_dict = {}

def collectscores(seriesuid, scores):
    global data_dict

    keys = ['SeriesInstanceUID', 'year1', 'year2', 'year3', 'year4', 'year5', 'year6']
    values = [
        os.path.basename(seriesuid).removesuffix('.mha'),
        scores[0][0],
        scores[0][1],
        scores[0][2],
        scores[0][3],
        scores[0][4],
        scores[0][5]
    ]
    # Use the seriesuid as the key for this entry in the data_dict
    entry_key = os.path.basename(seriesuid)
    data_dict[entry_key] = dict(zip(keys, values))

    return data_dict

def save_data_as_csv(data_dict, output_filename):

    with open(output_filename, "w", newline='') as csvfile:
        pass

    with open(output_filename, "w", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)

        writer.writeheader()

        for series_key, series_value in data_dict.items():
            writer.writerow(series_value)

def log_error(error_message, subfolder):
    with open(log_file, "a") as log:
        log.write("Error in subfolder '{}': {}\n".format(subfolder, error_message))
        traceback.print_exc(file=log)

if myoverview:
    scans = get_mha_paths_from_csv(myoverview,ParentDirectory)
elif mha3d:
    scans = get_mha_filepaths(ParentDirectory)
else:
    subfolders = get_subfolder_paths(ParentDirectory)
    scans = [get_mha_filepaths(i) for i in subfolders]

headers = ["SeriesInstanceUID", "year1", "year2", "year3", "year4", "year5", "year6"]

model = Sybil("sybil_ensemble")

for scan in tqdm(scans):
    try:
        serie = Serie([scan], mha3d=mha3d)
        scores = model.predict([serie])
        data_dict = collectscores(scan, scores[0])
        save_data_as_csv(data_dict, csvoutput)
    except Exception as e:
        log_error(str(e), scan)
end_time_model = time.time()
print(f"Time taken: {end_time_model - start_time_model} seconds")



