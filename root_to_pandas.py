#!/usr/bin/env python

# -----------------------------------------------------------------------------
# root_to_pandas.py
#
# Constructs pandas DataFrames from a ROOT file (qpixrtd output)
# * Author: Carter Eikenbary
# * Creation date: 2 December 2024
#
# Usage: python /path/to/root_to_pandas.py /path/to/file.root  /path/to/dataframe/output/
# Notes: HPRC users must load foss/2022b and source qpix-setup before running this script
# -----------------------------------------------------------------------------

import pandas as pd
import numpy as np
import uproot
import sys

input_path = sys.argv[1]
output_path = sys.argv[2]

root_tree = 'event_tree'

file = uproot.open(input_path + ':' + root_tree)

##########################
# Loading Geant4 data into dataframe
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

# Loading RTD data into dataframe
rtd_df = file.arrays(["pixel_x", "pixel_y", "pixel_initial_z_mean", "pixel_initial_z_std", "pixel_n_electrons"], library="pd")
rtd_df.reset_index("subentry", inplace=True)
rtd_df.reset_index("entry", inplace=True)
rtd_df.rename(columns={"entry": "event", "subentry": "PixelID"}, inplace=True)

resettime_df = file.arrays(["pixel_reset", "pixel_tslr", "pixel_reset_truth_track_id", "pixel_reset_truth_weight"], library="pd")
resettime_df.reset_index(drop=False, inplace=True)
resettime_df.rename(columns={"index": "event"}, inplace=True)

# Before merging, set the 'PixelID' column to integers in rtd_df
rtd_df["PixelID"] = rtd_df["PixelID"].astype(int)

# Create an empty list to store the data
data_list = []

# Iterate through the rows and elements in the 'pixel_reset' column
for i, row in resettime_df.iterrows():
    event = row['event']
    relevant_rows = rtd_df[rtd_df['event'] == event]
    for j, (pixel_reset, pixel_tslr, pixel_reset_truth_track_id, pixel_reset_truth_weight) in enumerate(zip(row['pixel_reset'], row['pixel_tslr'], row['pixel_reset_truth_track_id'], row['pixel_reset_truth_weight'])):
        if j < len(relevant_rows):
            pixel_x = relevant_rows.iloc[j]['pixel_x']
            pixel_y = relevant_rows.iloc[j]['pixel_y']

            pixel_initial_z_mean = relevant_rows.iloc[j]['pixel_initial_z_mean']
            pixel_initial_z_std = relevant_rows.iloc[j]['pixel_initial_z_std']
            pixel_n_electrons = relevant_rows.iloc[j]['pixel_n_electrons']

            data_list.append({'event': event, 'PixelID': j, 'pixel_x': pixel_x, 'pixel_y': pixel_y, 'reset_time': pixel_reset, 'TSLR': pixel_tslr, 'trackIDs': pixel_reset_truth_track_id, 'track_w': pixel_reset_truth_weight, 'pixel_initial_z_mean': pixel_initial_z_mean, 'pixel_initial_z_std': pixel_initial_z_std, 'pixel_n_electrons': pixel_n_electrons})

# Create the DataFrame from the list of dictionaries
rtd_df = pd.DataFrame(data_list)
rtd_df["nResets"] = rtd_df['reset_time'].apply(len)

column_order = ['event', 'PixelID', 'pixel_x', 'pixel_y', 'nResets', 'reset_time', 'TSLR', 'trackIDs', 'track_w', 'pixel_initial_z_mean', 'pixel_initial_z_std', 'pixel_n_electrons']

rtd_df = rtd_df[column_order]
rtd_df = rtd_df.reset_index(drop = True)
rtd_df.to_pickle(output_path + 'rtd_df.pkl')
print("rtd_df built in " + output_path + "rtd_df.pkl")
