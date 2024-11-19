#!/usr/bin/env python

__doc__ = """
Simple example script showing how to use the Sybil library locally to predict risk scores for a set of DICOM files.
"""

import numpy as np
# from sybil import visualize_attentions
from sybil import *
import os
import csv
# from sybil import visualize_attentions
# from examples.utils import get_demo_data


def get_dcm_filepaths(folder_path):
    # from Fennie van der Graaf
    """Parse through a folder of .dcm files and return a list of all file paths.

    Parameters:
    folder_path (str): The path to the folder containing the .dcm files.

    Returns:
    List[str]: A list of file paths for all .dcm files in the folder.
    """
    dcm_filepaths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".dcm"):
                dcm_filepaths.append(os.path.join(root, file))
    return dcm_filepaths

def get_subfolder_paths(parent_folder):
    """Return a list of all subfolder paths in a parent folder.

    Parameters:
    parent_folder (str): The path to the parent folder.

    Returns:
    List[str]: A list of paths for all subfolders in the parent folder.
    """
    subfolder_names = os.listdir(parent_folder)
    subfolder_paths = [os.path.join(parent_folder, name) for name in subfolder_names if os.path.isdir(os.path.join(parent_folder, name))]
    return subfolder_paths

headers_scores = ["seriesuid name", "year1", "year2", "year3", "year4", "year5", "year6"]

def save_data_as_csv(data_dict, output_filename):

    # First, create an empty CSV file
    with open(output_filename, "w", newline='') as csvfile:
        pass

    # Open a CSV file in write mode
    with open(output_filename, "w", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers_scores)

        # Write the header row
        writer.writeheader()

        # Iterate over the data dictionary and write each entry to the CSV file
        for series_key, series_value in data_dict.items():
            writer.writerow(series_value)

data_dict = {}

def collectscores(seriesuid, scores):
    global data_dict

    keys = ['seriesuid name', 'year1', 'year2', 'year3', 'year4', 'year5', 'year6']
    values = [
        os.path.basename(seriesuid),
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

def main():
    # Load a trained model
    print("Loading model")
    model = Sybil("sybil_ensemble")

    #Paths
    ParentDirectory = r"/data/bodyct/experiments/kristina_sybil/temp/test_data"
    csvoutput = r"/data/bodyct/experiments/kristina_sybil/temp/1.csv"
    logfile = None

    subfolders = get_subfolder_paths(ParentDirectory) #list of subfolders
    
    for subfolder in subfolders[0:1]: #[0:100]
        dicom_files = get_dcm_filepaths(subfolder) #list of string paths to dcm files

        # Get risk scores, print them, rm later
        serie = Serie(dicom_files)
        # print(serie.get_volume.input_dicts)
        # print(serie._meta[1])
        # serie.get_volume()
        # model._predict
        
        print(f"Processing {len(dicom_files)} DICOM files")
        # prediction = model.predict([serie], return_attentions=False)
        # scores = prediction.scores
        # print("Risk scores for ",os.path.basename(os.path.normpath(subfolder)),":", scores)

        # saving scores in csv file
        # folder_type = os.path.basename(os.path.normpath(ParentDirectory)) 
        # data_dict = collectscores(subfolder, scores)
        # save_data_as_csv(data_dict, csvoutput)

        print("done")

        # Visualize attention maps

        # output_dir = "sybil_attention_output"

        # print(f"Writing attention images to {output_dir}")
        # series_with_attention = visualize_attentions(
        #     serie,
        #     attentions=prediction.attentions,
        #     save_directory=output_dir,
        #     gain=3,
        # )
        # print(f"Finished writing attention images to {output_dir}")

if __name__ == "__main__":
    main()
