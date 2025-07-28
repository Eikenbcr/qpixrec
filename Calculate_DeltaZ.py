#!/usr/bin/env python

# -----------------------------------------------------------------------------
# Calculate_DeltaZ.py
#
# Measures the DeltaZ between measured Z above single-hit pixels and the true Z of hits above the pixel
# * Author: Carter Eikenbary
# * Creation date: 25 July 2025
#
# Usage: python /path/to/Calculate_DeltaZ.py /path/to/qpixrec/output/ event_low event_high
# Notes: HPRC users must load foss/2022b and source qpix-setup before running this script
# -----------------------------------------------------------------------------
import sys
import os
import numpy as np
import pandas as pd
import pickle
import argparse

from scipy import stats

from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

dfoutput_dir = sys.argv[1]
t0_hitmaker_dir = sys.argv[2]
deltaZ_dir = sys.argv[3] 

def std_exp(mean):
    return expected_const * np.sqrt(mean)

def std_difference(mean, std):
    return (std - std_exp(mean))

def single_cdf(x, a, b, c):
    return a * stats.norm.cdf(x, loc=b, scale=c)

def single_cdf_nostd(x, a, b):
    return a * stats.norm.cdf(x, loc=b, scale=std_exp(b))

def double_cdf_nostd(x, a, b, c, d):
    return a * stats.norm.cdf(x, loc=b, scale=std_exp(b)) + c * stats.norm.cdf(x, loc=d, scale=std_exp(d))

def triple_cdf_nostd(x, a, b, c, d, e, f):
    return a * stats.norm.cdf(x, loc=b, scale=std_exp(b)) + c * stats.norm.cdf(x, loc=d, scale=std_exp(d)) + e * stats.norm.cdf(x, loc=f, scale=std_exp(f))

def single_gaussian(x, amplitude, mean, sigma):
    return amplitude * np.exp(-0.5 * ((x - mean) / sigma)**2) 

diff_L = 6.8223 #cm**2/s
diff_T = 13.1586 #cm**2/s
elec_vel = 164800 #cm**2/s
expected_const = np.sqrt(2*diff_L/elec_vel**2)

#Read in rtd data
rtd_df = pd.read_pickle(dfoutput_dir + "rtd_df.pkl").reset_index(drop = True)
g4_df = pd.read_pickle(dfoutput_dir + "g4_df.pkl").reset_index(drop = True)
total_events = int(max(rtd_df.event) + 1)

singlehit_df = pd.read_pickle(t0_hitmaker_dir + "singlehit_df.pkl").reset_index(drop = True)

g4_Zs = []

for n in range(total_events):
    g4_Zs.append(np.median(g4_df[(g4_df.event == n) & (g4_df.ParticleID == 1)]['zi']))
    
g4_Zs = np.array(g4_Zs)

############
############

# Pre-index rtd_df for fast lookup
rtd_df_indexed = rtd_df.set_index(['event', 'PixelID'])

# Pre-group g4_df by event
g4_by_event = dict(tuple(g4_df.groupby('event')))

# Precompute pixel_range_ext per event
pixel_range_ext_dict = {
    event: np.sqrt(1 * diff_T * g4_Zs[event] / (elec_vel**3)) * elec_vel
    for event in singlehit_df['event'].unique()
}

# Filter to target events
singlehit_subdf = singlehit_df

# Initialize output containers
event_arr = []
pixelid_arr = []
num_hits_arr = []
delta_Z_arr = []

# Main loop
for row in tqdm(singlehit_subdf.itertuples(index=False), total=len(singlehit_subdf), desc="Processing hits"):
    event_num = int(row.event)
    pixel_num = int(row.PixelID)
    Z_pixel = row.Mean * elec_vel

    # Get pixel geometry
    try:
        pixel_data = rtd_df_indexed.loc[(event_num, pixel_num)]
    except KeyError:
        continue  # skip if pixel not found

    pix_x_low = pixel_data.pixel_x * 0.4 - 0.4
    pix_x_high = pixel_data.pixel_x * 0.4
    pix_y_low = pixel_data.pixel_y * 0.4 - 0.4
    pix_y_high = pixel_data.pixel_y * 0.4

    pixel_range_ext = pixel_range_ext_dict.get(event_num, 0)

    # Get all G4 tracks for this event
    hit_over_pixel = g4_by_event.get(event_num)
    if hit_over_pixel is None:
        continue

    valid_tracks = []

    for track in hit_over_pixel.itertuples(index=False):
        xi, xf = track.xi, track.xf
        yi, yf = track.yi, track.yf

        if (min(xi, xf) <= pix_x_high + pixel_range_ext and max(xi, xf) >= pix_x_low - pixel_range_ext and
            min(yi, yf) <= pix_y_high + pixel_range_ext and max(yi, yf) >= pix_y_low - pixel_range_ext):
            valid_tracks.append(track)

    if valid_tracks:
        Z_interpolated = []
        Eweight = []

        for track in valid_tracks:
            xi, xf = track.xi, track.xf
            yi, yf = track.yi, track.yf
            zi, zf = track.zi, track.zf
            En = track.E

            track_length = np.sqrt((xf - xi)**2 + (yf - yi)**2)
            pixel_overlap_length = 0

            for x in [pix_x_low, pix_x_high]:
                if xi != xf and min(xi, xf) <= x <= max(xi, xf):
                    y_int = yi + (x - xi) * (yf - yi) / (xf - xi)
                    if pix_y_low - pixel_range_ext <= y_int <= pix_y_high + pixel_range_ext:
                        z_int = zi + (x - xi) * (zf - zi) / (xf - xi)
                        pixel_overlap_length += np.abs(x - xi)
                        Z_interpolated.append(z_int)
                        Eweight.append(En)

            for y in [pix_y_low, pix_y_high]:
                if yi != yf and min(yi, yf) <= y <= max(yi, yf):
                    x_int = xi + (y - yi) * (xf - xi) / (yf - yi)
                    if pix_x_low - pixel_range_ext <= x_int <= pix_x_high + pixel_range_ext:
                        z_int = zi + (y - yi) * (zf - zi) / (yf - yi)
                        pixel_overlap_length += np.abs(y - yi)
                        Z_interpolated.append(z_int)
                        Eweight.append(En)
                        
            if (pix_x_low <= xi <= pix_x_high and pix_y_low <= yi <= pix_y_high and
                pix_x_low <= xf <= pix_x_high and pix_y_low <= yf <= pix_y_high):    
                segment_length = np.hypot(xf - xi, yf - yi)
                pixel_overlap_length = 0
                z_mid = 0.5 * (zi + zf)
                Z_interpolated.append(z_mid)
                Eweight.append(En)
                
            if pixel_overlap_length > 0:
                Eweight[-1] *= pixel_overlap_length / track_length

        if Z_interpolated:
            Z_interpolated = np.array(Z_interpolated)
            Eweight = np.array(Eweight)
            num_hits = len(Eweight)
            Z_true = np.average(Z_interpolated, weights=Eweight)
            delta_Z = Z_true - Z_pixel

            event_arr.append(event_num)
            pixelid_arr.append(pixel_num)
            num_hits_arr.append(num_hits)
            delta_Z_arr.append(delta_Z)

event_arr = np.array(event_arr)
pixelid_arr = np.array(pixelid_arr)
num_hits_arr = np.array(num_hits_arr)
delta_Z_arr = np.array(delta_Z_arr)


data = {
    'event': event_arr,
    'PixelID': pixelid_arr,
    'Hits': num_hits_arr,
    'DeltaZ': delta_Z_arr,
}

deltaZ_df = pd.DataFrame(data)
deltaZ_df.to_pickle(deltaZ_dir + '/deltaZ_df.pkl')
print("List of deltaZs written to " + deltaZ_dir + 'deltaZ_df.pkl')