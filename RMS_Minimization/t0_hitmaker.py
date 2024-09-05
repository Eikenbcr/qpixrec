#!/usr/bin/env python

# -----------------------------------------------------------------------------
# t0_hitmaker.py
#
# Determines a t0 and reconstructs the Z positions from the functional form for all events
# * Author: Carter Eikenbary
# * Creation date: 12 January 2024
#
# Usage: python /path/to/t0_hitmaker.py /path/to/t0_hitmaker/output/ binWidth verbosity
# Notes: HPRC users must load foss/2020b and source qpix-setup before running this script
# -----------------------------------------------------------------------------

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

# numpy to calculate RMS with np.std()
import numpy as np

# pandas used for handling the large amounts of data in dataframes rather than python lists
import pandas as pd

# scipy for curve fitting
from scipy.optimize import curve_fit

# optimization of RMS_Expected uses minimize function
from scipy.optimize import minimize

# sys for command line arguments
import sys

#Longitudinal Constant, Electron Drift Velocity, and the expected RMS constant
diff_L = 6.8223 #cm**2/s
elec_vel = 164800 #cm**2/s
expected_const = np.sqrt(2*diff_L/elec_vel**2)

dfoutput_dir = sys.argv[1]
t0_hitmaker_dir = sys.argv[2]
binWidth = float(sys.argv[4])

#Read in the nReset cut qpixrtd data
main_df = pd.read_pickle(dfoutput_dir + "main_subdf.pkl").reset_index(drop = True)
main_df = main_df.drop(columns=['reset_time', 'TSLR'])

total_events = int(max(main_df.event) + 1)

#Read in the qpixg4 data (used for comparison only)
g4_df = pd.read_pickle(dfoutput_dir + "g4_df.pkl")

#Read in the full qpixrtd data
full_df = pd.read_pickle(dfoutput_dir + "main_df.pkl")

def gaussFunc(x_f, A_f, x0_f, sigma_f):
    return A_f * np.exp(-(x_f - x0_f) ** 2 / (2 * sigma_f ** 2))


def gaussFitHist(data_f, bins_f):
    counts, bins = np.histogram(data_f, bins = bins_f)
    binCenters = []
    
    for i in range(len(bins) - 1):
        binCenters.append((bins[i]+bins[i+1])/2)

    gaussDataPoints = pd.DataFrame(data={'xvalues': binCenters, 'yvalues': counts
                                        })
                                                   
    gauss_params, covariance = curve_fit(gaussFunc, gaussDataPoints["xvalues"], gaussDataPoints["yvalues"],
                                p0 = [gaussDataPoints["yvalues"].max(), data_f.mean(), 2*data_f.std()])
    
    fit_errors = np.sqrt(np.diag(covariance))

    return gauss_params, fit_errors, gaussDataPoints

t0_shifts = []
search_range = 1e-9  # Define the search range (smaller than defined_range)

# Loop over all events
for event_id in range(total_events):  
    RMS_Expected_noshift = expected_const * np.sqrt(main_df[(main_df.event == event_id)].mean_TOA)
    difference_noshift = (main_df[(main_df.event == event_id)].RMS - RMS_Expected_noshift)         
    min_val = min(difference_noshift) 
    max_val = max(1.2e-7, -1*min_val)
    defined_range = min((max_val - min_val)/5, 6e-8)
    extended_range = defined_range/2  
    
   # Define the objective function for the current event
    def objective(t0_shift):
        RMS_Expected = expected_const * np.sqrt(main_df[(main_df.event == event_id)].mean_TOA - t0_shift)
        difference = (main_df[(main_df.event == event_id)].RMS - RMS_Expected)

        # Set NaN values to zero
        difference = np.nan_to_num(difference, nan=0.0)
   
        max_count = 0
        optimal_range_start = min_val
        
        # Search within a smaller interval
        for start in np.arange(min_val, max_val, search_range):
            end = start + defined_range
            count = np.sum((difference >= start) & (difference < end))

            if count > max_count:
                max_count = count
                optimal_range_start = start

        # Calculate the extended range within data bounds
        optimal_range_end = min(max_val, optimal_range_start + defined_range + extended_range)
        optimal_range_start = max(min_val, optimal_range_start - extended_range)

        # Use the adjusted range to calculate weighted average and actual average
        selected_data = difference[((difference >= optimal_range_start) & (difference <= optimal_range_end))]

        weighted_avg = np.average(selected_data)
        return (weighted_avg ** 2)
      
          # Initial guess for t0_shift (for simulation t0=0)
    initial_t0_shift = 0.0000

    # Define bounds for t0_shift
    bounds = [(-0.003, min(main_df[(main_df.event == event_id)].mean_TOA))]

    # Define the optimization tolerance for higher precision
    tolerance = 1e-10

    # Perform the nonlinear optimization using the "Nelder-Mead" algorithm
    result = minimize(objective, initial_t0_shift, method='Nelder-Mead', bounds=bounds, tol=tolerance)

    # Get the optimal t0_shift value for the current event
    optimal_t0_shift = result.x[0]

    # Append the optimal t0_shift to the array
    t0_shifts.append(optimal_t0_shift)


t0_shifts = np.array(t0_shifts)
t0_cut = t0_shifts[(t0_shifts >= (t0_shifts.mean() - 2*t0_shifts.std())) & (t0_shifts <= (t0_shifts.mean() + 2*t0_shifts.std()))]

# Plot the t0 distribution
plt.hist(t0_shifts)
xmin, xmax = plt.xlim()
plt.close()

plt.figure(figsize = (9,9))
plt.xlabel('t0 (sec)', fontsize = 16)
plt.title('t0 Distribution - All Events [DS = 1], Z=AllZ', fontsize = 20)
y,x,_ = plt.hist(t0_shifts, bins = np.arange(xmin, xmax, binWidth))

# Get a gaussian fit to the distribution
t0_gauss_params, t0_fit_errors, hist_data = gaussFitHist(t0_cut, 
                                          np.arange(xmin, xmax, binWidth)) 

# Plot the Gaussian Fit
xs = np.linspace(xmin, xmax, 1000)
ys = gaussFunc(xs, *t0_gauss_params)
# Calculate the range for the plot based on 2 sigma
sigma = t0_gauss_params[2]
mean = t0_gauss_params[1]
lower_bound = mean - 2 * sigma
upper_bound = mean + 2 * sigma

# Filter xs and ys within the 2 sigma range
filtered_xs = xs[(xs >= lower_bound) & (xs <= upper_bound)]
filtered_ys = ys[(xs >= lower_bound) & (xs <= upper_bound)]

# Plot the filtered data
plt.plot(filtered_xs, filtered_ys, '--', color='crimson', label=('Gaussian Fit: \n' + r'$\mu = {:.4e} \pm {:.4e}$' '\n' + r'$\sigma = {:.4e} \pm {:.4e}$').format(t0_gauss_params[1], t0_fit_errors[1], t0_gauss_params[2], t0_fit_errors[2]))

# Plot vertical line at t = 0
x = [0]*2
y = np.linspace(0, hist_data['yvalues'].max(), 2)
plt.plot(x, y, '--', color = 'lime')
plt.legend(loc="upper left")

# Show and close plot
plt.savefig(t0_hitmaker_dir + '/t0_distribution.png')
plt.close()

# Print distribution gaussian parameters
print('Gaussian Parameters: A: %3.5E, mu: %3.5E, sigma: %3.5E' %(t0_gauss_params[0], t0_gauss_params[1], 
                                                                 t0_gauss_params[2]))

output_str = 'Gaussian Parameters: A: %3.5E, mu: %3.5E, sigma: %3.5E' % (
    t0_gauss_params[0], t0_gauss_params[1], t0_gauss_params[2]
)

# Specify the file path where you want to save the output
output_file_path = t0_hitmaker_dir + 't0_parameters.txt'

# Write the output to the file
with open(output_file_path, 'w') as file:
    file.write(output_str)

print("t0 output written to " + output_file_path)

t0_df = pd.DataFrame(data = {'event' : range(total_events), 't0': t0_shifts})

# Merge t0 values into full_df
full_df = full_df.merge(t0_df, how = 'right', on = 'event').copy().reset_index(drop = True)
full_df['TOF'] = (full_df.mean_TOA - full_df.t0)
full_df['x_pos'] = (full_df['pixel_x'] * 4 - 2)/10 # Subtract 2mm to get in center of the pixel (pixels are 4mm wide)
full_df['y_pos'] = (full_df['pixel_y'] * 4 - 2)/10 # Subtract 2mm to get in center of the pixel (pixels are 4mm wide)
full_df['z_pos'] = elec_vel * full_df.TOF

column_order = ['event', 'PixelID', 'nResets', 'mean_TOA', 'RMS', 't0', 'TOF', 'x_pos', 'y_pos', 'z_pos', 'reset_time'] 
full_df = full_df[column_order]

full_df.to_pickle(t0_hitmaker_dir + '/reconstruction_df.pkl')
print("Dataframe written to " + t0_hitmaker_dir + 'reconstruction_df.pkl')
