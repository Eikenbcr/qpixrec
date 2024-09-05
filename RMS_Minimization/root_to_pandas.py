#!/usr/bin/env python

# -----------------------------------------------------------------------------
# root_to_pandas.py
#
# Constructs pandas DataFrames from a ROOT file (qpixrtd output)
# * Author: Carter Eikenbary
# * Creation date: 10 August 2023
#
# Usage: python /path/to/root_to_pandas.py /path/to/file.root  /path/to/dataframe/output/ 
# Notes: HPRC users must load foss/2020b and source qpix-setup before running this script
# -----------------------------------------------------------------------------

import pandas as pd
import numpy as np
import uproot
import sys

input_path = sys.argv[1]
output_path = sys.argv[2]

root_tree = 'event_tree'

file = uproot.open(input_path + ':' + root_tree)

g4_df = file.arrays(["hit_start_x","hit_end_x","hit_start_y","hit_end_y","hit_start_z","hit_end_z","hit_start_t","hit_end_t","hit_energy_deposit", "hit_track_id"], library="pd")
g4_df.rename(columns={"hit_start_x": "xi", "hit_end_x": "xf"}, inplace=True)
g4_df.rename(columns={"hit_start_y": "yi", "hit_end_y": "yf"}, inplace=True)
g4_df.rename(columns={"hit_start_z": "zi", "hit_end_z": "zf"}, inplace=True)
g4_df.rename(columns={"hit_start_t": "ti", "hit_end_t": "tf"}, inplace=True)
g4_df.rename(columns={"hit_energy_deposit": "E", "hit_track_id": "ParticleID"}, inplace=True)
g4_df.reset_index("subentry", inplace=True)
g4_df.reset_index("entry", inplace=True)
g4_df.rename(columns={"entry": "event", "subentry": "TrackID"}, inplace=True)
g4_df.to_pickle(output_path + 'g4_df.pkl')
print("g4_df built in " + output_path + "g4_df.pkl")

# Load data into DataFrames
pixel_df = file.arrays(["pixel_x", "pixel_y"], library="pd")
pixel_df.reset_index("subentry", inplace=True)
pixel_df.reset_index("entry", inplace=True)
pixel_df.rename(columns={"entry": "event", "subentry": "PixelID"}, inplace=True)

resettime_df = file.arrays(["pixel_reset", "pixel_tslr"], library="pd")
resettime_df.reset_index(drop=False, inplace=True)
resettime_df.rename(columns={"index": "event"}, inplace=True)

# Before merging, set the 'PixelID' column to integers in pixel_df
pixel_df["PixelID"] = pixel_df["PixelID"].astype(int)

# Create an empty list to store the data
data_list = []

# Iterate through the rows and elements in the 'pixel_reset' column
for i, row in resettime_df.iterrows():
    event = row['event']
    relevant_rows = pixel_df[pixel_df['event'] == event]
    for j, (pixel_reset, pixel_tslr) in enumerate(zip(row['pixel_reset'], row['pixel_tslr'])):
        if j < len(relevant_rows):
            pixel_x = relevant_rows.iloc[j]['pixel_x']
            pixel_y = relevant_rows.iloc[j]['pixel_y']

            data_list.append({'event': event, 'PixelID': j, 'pixel_x': pixel_x, 'pixel_y': pixel_y, 'reset_time': pixel_reset, 'TSLR': pixel_tslr})

# Create the DataFrame from the list of dictionaries
resets_df = pd.DataFrame(data_list)
resets_df["nResets"] = resets_df['reset_time'].apply(len)
resets_df["mean_TOA"] = resets_df['reset_time'].apply(lambda x: pd.to_numeric(x).mean())
resets_df["RMS"] = resets_df['reset_time'].apply(lambda x: np.std(x, ddof=1))

column_order = ['event', 'PixelID', 'pixel_x', 'pixel_y', 'nResets', 'mean_TOA', 'RMS', 'reset_time', 'TSLR']
resets_df = resets_df[column_order]
main_df = resets_df.copy().reset_index(drop = True)
main_df.to_pickle(output_path + 'main_df.pkl')
print("main_df built in " + output_path + "main_df.pkl")

main_subdf = main_df.copy()
main_subdf.to_pickle(output_path + 'main_subdf.pkl')
print("main_subdf built in " + output_path + "main_subdf.pkl")
