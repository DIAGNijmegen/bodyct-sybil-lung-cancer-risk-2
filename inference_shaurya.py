import os
from sybil import Serie, Sybil
import time
import json
import pandas as pd
import numpy as np
import csv
import traceback
from sybil import visualize_attentions

EXPERIMENT_DIR = r"/data/bodyct/experiments/lung-malignancy-fairness-shaurya"

# Name of CSV file with series instance UIDs to extract
csv_input = rf"{EXPERIMENT_DIR}/nlst/sybil_fn_brock_top25.csv"
INFERENCE_DIR = rf"{EXPERIMENT_DIR}/nlst/sybil_attentions"

os.makedirs(INFERENCE_DIR, exist_ok=True)
ParentDirectory = rf"{EXPERIMENT_DIR}/nlst/DICOM_files"

# Name of the output csv file
csvoutput = rf"{INFERENCE_DIR}/inference.csv"
csvoutput2 = rf"{INFERENCE_DIR}/inference2.csv"
csv_error = rf"{INFERENCE_DIR}/error_ids.csv"
csvattentionoutput = rf"{INFERENCE_DIR}/output_attention_scores.csv"
log_file = rf"{INFERENCE_DIR}/errorlog.txt"


start_time_model = time.time()
# Load a trained model
model = Sybil("sybil_ensemble")


def get_series_instance_uids(csv_input, n=None):
    df = pd.read_csv(csv_input)
    # df = df[(df["Thijmen_mean"].isna()) & (df["InSybilTrain"] == False)]
    ids = pd.unique(df["SeriesInstanceUID"]).tolist()
    if n is not None:
        ids = ids[0:n]
    return ids


def get_subfolder_paths(parent_folder, id_list=None):
    """Return a list of all subfolder paths in a parent folder.

    Parameters:
    parent_folder (str): The path to the parent folder.

    Returns:
    List[str]: A list of paths for all subfolders in the parent folder.
    """
    subfolder_names = os.listdir(parent_folder)
    if id_list is not None:
        subfolder_names = list(set(list(subfolder_names)).intersection(set(id_list)))

    subfolder_paths = [
        os.path.join(parent_folder, name)
        for name in subfolder_names
        if os.path.isdir(os.path.join(parent_folder, name))
    ]
    return subfolder_paths


def get_dcm_filepaths(folder_path):
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


seriesids = get_series_instance_uids(csv_input)
subfolders = get_subfolder_paths(ParentDirectory, seriesids)
print(f"Examining {len(subfolders)} subfolders")

# Initialize an empty dictionary
data_dict = {}


def collectscores(seriesuid, scores):
    global data_dict

    keys = ["SeriesInstanceUID", "year1", "year2", "year3", "year4", "year5", "year6"]
    values = [
        os.path.basename(seriesuid),
        scores[0][0][0],
        scores[0][0][1],
        scores[0][0][2],
        scores[0][0][3],
        scores[0][0][4],
        scores[0][0][5],
    ]

    # Use the seriesuid as the key for this entry in the data_dict
    entry_key = os.path.basename(seriesuid)
    data_dict[entry_key] = dict(zip(keys, values))
    return data_dict


def save_data_as_csv(data_dict, output_filename):
    # First, create an empty CSV file
    with open(output_filename, "w", newline="") as csvfile:
        pass

    # Open a CSV file in write mode
    with open(output_filename, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers_scores)

        # Write the header row
        writer.writeheader()

        # Iterate over the data dictionary and write each entry to the CSV file
        for series_key, series_value in data_dict.items():
            writer.writerow(series_value)


def log_error(error_message, subfolder):
    with open(log_file, "a") as log:
        log.write("Error in subfolder '{}': {}\n\n".format(subfolder, error_message))
        traceback.print_exc(file=log)


headers_scores = [
    "SeriesInstanceUID",
    "year1",
    "year2",
    "year3",
    "year4",
    "year5",
    "year6",
]

with open(csvoutput, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=headers_scores)
    writer.writeheader()

with open(csv_error, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["SeriesInstanceUID"])
    writer.writeheader()

# New headers for attention scores CSV
headers_attention_scores = ["series instance uid", "attention_score"]

# Open the CSV file for attention scores
with open(csvattentionoutput, "w", newline="") as attention_csvfile:
    attention_writer = csv.writer(attention_csvfile)
    attention_writer.writerow(headers_attention_scores)

for i, subfolder in enumerate(subfolders):
    try:
        # print(f"{i+1} / {len(subfolders)}: examining {subfolder} ...")
        dcm_filepaths = get_dcm_filepaths(subfolder)
        serie = Serie(dcm_filepaths)
        scores = model.predict([serie], return_attentions=True)
        attentions = scores.attentions

        # Save attention scores to the attention CSV file
        with open(csvattentionoutput, "a", newline="") as attention_csvfile:
            attention_writer = csv.writer(attention_csvfile)
            for attention_score in attentions:
                attention_writer.writerow(
                    [os.path.basename(subfolder), attention_score]
                )

        folder_type = os.path.basename(os.path.normpath(ParentDirectory))
        data_dict = collectscores(subfolder, scores)

        with open(csvoutput, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers_scores)
            writer.writerow(data_dict[os.path.basename(subfolder)])

        # save_data_as_csv(data_dict, csvoutput)

        series_with_attention = visualize_attentions(
            serie,
            attentions=attentions,
            save_directory=rf"{INFERENCE_DIR}/attention_overlays/",
            gain=3,
            series_uids=str(os.path.basename(subfolder)),
        )

    except Exception as e:
        log_error(str(e), subfolder)
        seriesuid = os.path.basename(subfolder)
        with open(csv_error, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["SeriesInstanceUID"])
            writer.writerow({"SeriesInstanceUID": seriesuid})
        continue

end_time_model = time.time()
print(f"Time taken for inference: {end_time_model - start_time_model} seconds")

# def split_gif_to_frames(gif_path, output_folder):
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     with Image.open(gif_path) as gif:
#         frame_count = gif.n_frames
#         print(f"Total frames: {frame_count}")

#         for frame in range(frame_count):
#             gif.seek(frame)
#             frame_filename = os.path.join(output_folder, f"frame_{frame:03d}.png")
#             gif.save(frame_filename, format="PNG")


# print(f"Splitting gifs into frames")
# os.makedirs(rf"{EXPERIMENT_DIR}/nlst/sybil_attentions_black/attention_imgs")
# gif_subfolders = os.listdir(
#     rf"{EXPERIMENT_DIR}/nlst/sybil_attentions_black/attention_gifs"
# )

# for i, sub in enumerate(gif_subfolders):
#     gif_path = (
#         f"{EXPERIMENT_DIR}/nlst/sybil_attentions_black/attention_gifs/{sub}/{sub}.gif"
#     )
#     output_folder = f"{EXPERIMENT_DIR}/nlst/sybil_attentions_black/attention_imgs/{sub}"
#     print(f"{i+1} / {len(gif_subfolders)}: converting {sub} ...")
#     split_gif_to_frames(gif_path, output_folder)

# print("Splitting gifs into frames complete!")

save_data_as_csv(data_dict, csvoutput2)
