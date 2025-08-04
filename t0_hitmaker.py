#!/usr/bin/env python

# -----------------------------------------------------------------------------
# t0_hitmaker.py
#
# Determines a t0 and reconstructs the Z positions from RMS and Mean of CDF fits
# * Author: Carter Eikenbary
# * Creation date: 2 December 2024
#
# Usage: python /path/to/t0_hitmaker.py /path/to/t0_hitmaker/output/ -threshold # -rmin # -rmax #
# Notes: HPRC users must load foss/2022b and source qpix-setup before running this script
# -----------------------------------------------------------------------------

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from scipy import stats, optimize

from scipy.optimize import curve_fit
from scipy.optimize import minimize

import sys
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

dfoutput_dir = sys.argv[1]
t0_hitmaker_dir = sys.argv[2]
reset_threshold = int(sys.argv[3])
reset_min = int(sys.argv[4])
reset_max = int(sys.argv[5])
clockspeed = float(sys.argv[6])

# Constants
diff_L = 6.8223 #cm**2/s
elec_vel = 164800 #cm**2/s
expected_const = np.sqrt(2*diff_L/elec_vel**2)
energy_per_reset = (23.6*reset_threshold)*(1e-6)
##########################

def std_exp(mean):
    return expected_const * np.sqrt(mean)

def std_difference(mean, std):
    return (std - std_exp(mean))

def t0_solve(mean, std):
    return (mean - (std/expected_const)**2)

def single_cdf(x, a, b, c):
    return a * stats.norm.cdf(x, loc=b, scale=c)

def inverse_singlecdf_solver(y_targets, a, b, c, x_min=-1, x_max=1):
    def root_solver(y):
        def func(x):
            return single_cdf(x, a, b, c) - y
        try:
            return optimize.brentq(func, x_min, x_max)
        except ValueError:
            return np.nan

    y_targets = np.atleast_1d(y_targets)
    results = np.array([root_solver(y) for y in y_targets])
    return results if len(results) > 1 else results[0]

def double_cdf(x, a, b, c, d, e, f):
    return (
        a * stats.norm.cdf(x, loc=b, scale=c) + 
        d * stats.norm.cdf(x, loc=e, scale=f)
    )

def inverse_doublecdf_solver(y_targets, a, b, c, d, e, f, x_min=-1, x_max=1):
    def root_solver(y):
        def func(x):
            return double_cdf(x, a, b, c, d, e, f) - y
        try:
            return optimize.brentq(func, x_min, x_max)
        except ValueError:
            return np.nan  # If no root is found in the range

    return np.array([root_solver(y) for y in y_targets])

def triple_cdf(x, a, b, c, d, e, f, g, h, i):
    return (
        a * stats.norm.cdf(x, loc=b, scale=c) +
        d * stats.norm.cdf(x, loc=e, scale=f) +
        g * stats.norm.cdf(x, loc=h, scale=i)
    )

def inverse_triplecdf_solver(y_targets, a, b, c, d, e, f, g, h, i, x_min=-1, x_max=1):
    def root_solver(y):
        def func(x):
            return triple_cdf(x, a, b, c, d, e, f, g, h, i) - y
        try:
            return optimize.brentq(func, x_min, x_max)
        except ValueError:
            return np.nan

    y_targets = np.atleast_1d(y_targets)
    results = np.array([root_solver(y) for y in y_targets])
    return results if len(results) > 1 else results[0]

def single_gaussian(x, amplitude, mean, sigma):
    return amplitude * np.exp(-0.5 * ((x - mean) / sigma)**2)

def single_cdf_nostd(x, a, b):
    return a * stats.norm.cdf(x, loc=b, scale=std_exp(b))

def double_cdf_nostd(x, a, b, c, d):
    return a * stats.norm.cdf(x, loc=b, scale=std_exp(b)) + c * stats.norm.cdf(x, loc=d, scale=std_exp(d))

def triple_cdf_nostd(x, a, b, c, d, e, f):
    return a * stats.norm.cdf(x, loc=b, scale=std_exp(b)) + c * stats.norm.cdf(x, loc=d, scale=std_exp(d)) + e * stats.norm.cdf(x, loc=f, scale=std_exp(f))

def process_singlecdf(df, plot=False):
    singlecdf_event = []
    singlecdf_pixid = []
    singlecdf_amp = []
    singlecdf_mean = []
    singlecdf_std = []
    singlecdf_diff = []
    singlecdf_t0 = []
    singlecdf_rmse = []
    
    iterator = tqdm(range(len(df)), desc="Fitting CDFs")

    for i in iterator:
        row = df.iloc[i]
        reset_times = np.asarray(row.reset_time)
        num_resets = len(reset_times)
        event = row.event
        pixelid = row.PixelID
        reset_count = np.arange(1, num_resets + 1)
        initial_params = [num_resets + 0.1, np.median(reset_times), np.std(reset_times)]
        bounds = [(num_resets, 0, 0), (np.inf, np.inf, np.inf)]
        try:
            cdf_params, _ = curve_fit(single_cdf, reset_times, reset_count, p0=initial_params, bounds=bounds)

            amp, mean, std = cdf_params
            diff = std_difference(mean, std)
            t0_val = t0_solve(mean, std)
            
            expected_reset_times = inverse_singlecdf_solver(reset_count, amp, mean, std)
            
            rmse = np.sqrt(np.mean((reset_times - expected_reset_times) ** 2))
            if rmse < 0.5*clockspeed:
                singlecdf_event.append(row.event)
                singlecdf_pixid.append(row.PixelID)
                singlecdf_amp.append(amp)
                singlecdf_mean.append(mean)
                singlecdf_std.append(std)
                singlecdf_diff.append(diff)
                singlecdf_t0.append(t0_val)
                singlecdf_rmse.append(rmse)
            
            if plot:
                # Plotting the reset data and the CDF fit
                plt.figure(figsize=(10, 8))
    
                # Plot the reset data
                plt.scatter(reset_times, reset_count, label='Reset Data', color='blue', alpha=0.5)

                # Create a smooth curve for the CDF fit
                x_fit = np.linspace(min(reset_times), max(reset_times), 100)
                y_fit = single_cdf(x_fit, *cdf_params)

                # Plot the CDF fit
                plt.plot(x_fit, y_fit, label=('CDF Fit:' '\n' + r'$\mu = {:.4e} sec$' '\n' + r'$\sigma = {:.4e} sec$').format(mean, std), color='red')
                plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
                plt.xticks(np.linspace(min(x_fit), max(x_fit), 5))
                plt.xlabel('Reset Time [sec]', fontsize=16)
                plt.ylabel('Cumulative Resets', fontsize=16)
                plt.title(f'Single-CDF Fit for Pixel {pixelid}, Event {event}', fontsize=16)
                plt.legend(fontsize=14)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.grid()
                plt.show()               
            
        except RuntimeError:
            continue

    data = {
        'event': singlecdf_event,
        'PixelID': singlecdf_pixid,
        'Amp': singlecdf_amp,
        'Mean': singlecdf_mean,
        'StD': singlecdf_std,
        'Diff': singlecdf_diff,
        't0': singlecdf_t0,
        'rmse': singlecdf_rmse,
    }

    return pd.DataFrame(data)

def process_singlehit(df, t0, plot=False):
    singlecdf_event = []
    singlecdf_pixid = []
    singlecdf_amp = []
    singlecdf_mean = []
    singlecdf_std = []
    singlecdf_rmse = []
    
    iterator = tqdm(range(len(df)), desc="Fitting CDFs")

    for i in iterator:
        row = df.iloc[i]
        reset_times = np.asarray(row.reset_time) - t0
        if np.median(reset_times) < 0:
            continue
        
        num_resets = len(reset_times)
        event = row.event
        pixelid = row.PixelID
        reset_count = np.arange(1, num_resets + 1)
        initial_params = [num_resets + 0.1, np.median(reset_times)]
        
        bounds = [(num_resets, 0), (np.inf, np.inf)]
        try:
            cdf_params, _ = curve_fit(single_cdf_nostd, reset_times, reset_count, p0=initial_params, bounds=bounds)

            amp, mean = cdf_params
            std = std_exp(mean)
            
            expected_reset_times = inverse_singlecdf_solver(reset_count, amp, mean, std)
            
            rmse = np.sqrt(np.mean((reset_times - expected_reset_times) ** 2))

            if rmse < clockspeed:
                singlecdf_event.append(row.event)
                singlecdf_pixid.append(row.PixelID)
                singlecdf_amp.append(amp)
                singlecdf_mean.append(mean)
                singlecdf_std.append(std)
                singlecdf_rmse.append(rmse)
            
            if plot:
                # Plotting the reset data and the CDF fit
                plt.figure(figsize=(10, 8))
    
                # Plot the reset data
                plt.scatter(reset_times, reset_count, label='Reset Data', color='blue', alpha=0.5)

                # Create a smooth curve for the CDF fit
                x_fit = np.linspace(min(reset_times), max(reset_times), 100)
                y_fit = single_cdf(x_fit, *cdf_params)

                # Plot the CDF fit
                plt.plot(x_fit, y_fit, label=('CDF Fit:' '\n' + r'$\mu = {:.4e} sec$' '\n' + r'$\sigma = {:.4e} sec$').format(mean, std), color='red')
                plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
                plt.xticks(np.linspace(min(x_fit), max(x_fit), 5))
                plt.xlabel('Reset Time [sec]', fontsize=16)
                plt.ylabel('Cumulative Resets', fontsize=16)
                plt.title(f'Single-CDF Fit for Pixel {pixelid}, Event {event}', fontsize=16)
                plt.legend(fontsize=14)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.grid()
                plt.show()               
            
        except RuntimeError:
            continue

    data = {
        'event': singlecdf_event,
        'PixelID': singlecdf_pixid,
        'Amp': singlecdf_amp,
        'Mean': singlecdf_mean,
        'StD': singlecdf_std,
        'rmse': singlecdf_rmse,
    }

    return pd.DataFrame(data)

def process_doublehit(df, t0, plot=False):
    doublecdf_event = []
    doublecdf_pixid = []
    doublecdf_amp1 = []
    doublecdf_mean1 = []
    doublecdf_std1 = []
    doublecdf_amp2 = []
    doublecdf_mean2 = []
    doublecdf_std2 = []
    doublecdf_rmse = []
    
    iterator = tqdm(range(len(df)), desc="Fitting CDFs")

    for i in iterator:
        row = df.iloc[i]
        reset_times = np.asarray(row.reset_time) - t0
        if np.median(reset_times) < 0:
            continue
        
        num_resets = len(reset_times)
        event = row.event
        pixelid = row.PixelID
        reset_count = np.arange(1, num_resets + 1)
        initial_params = [num_resets/2, np.median(reset_times) - std_exp(np.median(reset_times)), num_resets/2, np.median(reset_times) + std_exp(np.median(reset_times))]
        bounds = [(0.1, 0, 0.1, 0), (np.inf, np.inf, np.inf, np.inf)]
        try:
            cdf_params, _ = curve_fit(double_cdf_nostd, reset_times, reset_count, p0=initial_params, bounds=bounds)
            amp1, mean1, amp2, mean2 = cdf_params
            std1 = std_exp(mean1)
            std2 = std_exp(mean2) 
            
            expected_reset_times = inverse_doublecdf_solver(reset_count, amp1, mean1, std1, amp2, mean2, std2)
            
            rmse = np.sqrt(np.mean((reset_times - expected_reset_times) ** 2))
 
            if rmse < clockspeed:
                doublecdf_event.append(row.event)
                doublecdf_pixid.append(row.PixelID)
                doublecdf_amp1.append(amp1)
                doublecdf_mean1.append(mean1)
                doublecdf_std1.append(std1)
                doublecdf_amp2.append(amp2)
                doublecdf_mean2.append(mean2)
                doublecdf_std2.append(std2)
                doublecdf_rmse.append(rmse)
            
            if plot:
                # Plotting the reset data and the CDF fit
                plt.figure(figsize=(10, 8))
    
                # Plot the reset data
                plt.scatter(reset_times, reset_count, label='Reset Data', color='blue', alpha=0.5)

                # Create a smooth curve for the CDF fit
                x_fit = np.linspace(min(reset_times), max(reset_times), 100)
                y_fit = double_cdf(x_fit, amp1, mean1, std1, amp2, mean2, std2)
                y1_fit = single_cdf(x_fit, amp1, mean1, std1)
                y2_fit = single_cdf(x_fit, amp2, mean2, std2)
                
                # Plot the CDF fit
                plt.plot(x_fit, y_fit, label=('Double-CDF Fit:'), color='red')
                plt.plot(x_fit, y1_fit, color='orange', alpha=0.3, label=( r'$\mu_1 = {:.4e} sec$' '\n' + r'$\sigma_1 = {:.4e} sec$').format(mean1, std1))
                plt.plot(x_fit, y2_fit, color='darkorange', alpha=0.6, label=( r'$\mu_2 = {:.4e} sec$' '\n' + r'$\sigma_2 = {:.4e} sec$').format(mean2, std2))                
                plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
                plt.xticks(np.linspace(min(x_fit), max(x_fit), 5))
                plt.xlabel('Reset Time [sec]', fontsize=16)
                plt.ylabel('Cumulative Resets', fontsize=16)
                plt.title(f'Double-CDF Fit for Pixel {pixelid}, Event {event}', fontsize=16)
                plt.legend(fontsize=14)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.grid()
                plt.show()               
            
        except RuntimeError:
            continue

    data = {
        'event': doublecdf_event,
        'PixelID': doublecdf_pixid,
        'Amp1': doublecdf_amp1,
        'Mean1': doublecdf_mean1,
        'StD1': doublecdf_std1,
        'Amp2': doublecdf_amp2,
        'Mean2': doublecdf_mean2,
        'StD2': doublecdf_std2,
        'rmse': doublecdf_rmse,
    }

    return pd.DataFrame(data)

def process_triplehit(df, t0, plot=False):
    triplecdf_event = []
    triplecdf_pixid = []
    triplecdf_amp1 = []
    triplecdf_mean1 = []
    triplecdf_std1 = []
    triplecdf_amp2 = []
    triplecdf_mean2 = []
    triplecdf_std2 = []
    triplecdf_amp3 = []
    triplecdf_mean3 = []
    triplecdf_std3 = []
    triplecdf_rmse = []
    
    iterator = tqdm(range(len(df)), desc="Fitting CDFs")

    for i in iterator:
        row = df.iloc[i]
        reset_times = np.asarray(row.reset_time) - t0
        if np.median(reset_times) < 0:
            continue
        
        num_resets = len(reset_times)
        event = row.event
        pixelid = row.PixelID
        reset_count = np.arange(1, num_resets + 1)
        initial_params = [num_resets/3, np.median(reset_times) - 2*std_exp(np.median(reset_times)), num_resets/3, np.median(reset_times) + 2*std_exp(np.median(reset_times)), num_resets/3, np.median(reset_times)]
        bounds = [(0.1, 0, 0.1, 0, 0.1, 0), (np.inf, np.inf, np.inf, np.inf, np.inf, np.inf)]
        try:
            cdf_params, _ = curve_fit(triple_cdf_nostd, reset_times, reset_count, p0=initial_params, bounds=bounds)
            amp1, mean1, amp2, mean2, amp3, mean3 = cdf_params
            std1 = std_exp(mean1)
            std2 = std_exp(mean2)
            std3 = std_exp(mean3)
            
            expected_reset_times = inverse_triplecdf_solver(reset_count, amp1, mean1, std1, amp2, mean2, std2, amp3, mean3, std3)
            
            rmse = np.sqrt(np.mean((reset_times - expected_reset_times) ** 2))

            if rmse < clockspeed:
                triplecdf_event.append(row.event)
                triplecdf_pixid.append(row.PixelID)
                triplecdf_amp1.append(amp1)
                triplecdf_mean1.append(mean1)
                triplecdf_std1.append(std1)
                triplecdf_amp2.append(amp2)
                triplecdf_mean2.append(mean2)
                triplecdf_std2.append(std2)
                triplecdf_amp3.append(amp3)
                triplecdf_mean3.append(mean3)
                triplecdf_std3.append(std3)
                triplecdf_rmse.append(rmse)
            
            if plot:
                # Plotting the reset data and the CDF fit
                plt.figure(figsize=(10, 8))
    
                # Plot the reset data
                plt.scatter(reset_times, reset_count, label='Reset Data', color='blue', alpha=0.5)

                # Create a smooth curve for the CDF fit
                x_fit = np.linspace(min(reset_times), max(reset_times), 100)
                y_fit = triple_cdf(x_fit, amp1, mean1, std1, amp2, mean2, std2, amp3, mean3, std3)
                y1_fit = single_cdf(x_fit, amp1, mean1, std1)
                y2_fit = single_cdf(x_fit, amp2, mean2, std2)
                y3_fit = single_cdf(x_fit, amp3, mean3, std3)
                
                # Plot the CDF fit
                plt.plot(x_fit, y_fit, label=('Triple-CDF Fit:'), color='red')
                plt.plot(x_fit, y1_fit, color='orange', alpha=0.3, label=( r'$\mu_1 = {:.4e} sec$' '\n' + r'$\sigma_1 = {:.4e} sec$').format(mean1, std1))
                plt.plot(x_fit, y2_fit, color='darkorange', alpha=0.6, label=( r'$\mu_2 = {:.4e} sec$' '\n' + r'$\sigma_2 = {:.4e} sec$').format(mean2, std2))                
                plt.plot(x_fit, y3_fit, color='brown', alpha=0.6, label=( r'$\mu_3 = {:.4e} sec$' '\n' + r'$\sigma_3 = {:.4e} sec$').format(mean3, std3))                

                plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
                plt.xticks(np.linspace(min(x_fit), max(x_fit), 5))
                plt.xlabel('Reset Time [sec]', fontsize=16)
                plt.ylabel('Cumulative Resets', fontsize=16)
                plt.title(f'triple-CDF Fit for Pixel {pixelid}, Event {event}', fontsize=16)
                plt.legend(fontsize=14)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.grid()
                plt.show()               
            
        except RuntimeError:
            continue

    data = {
        'event': triplecdf_event,
        'PixelID': triplecdf_pixid,
        'Amp1': triplecdf_amp1,
        'Mean1': triplecdf_mean1,
        'StD1': triplecdf_std1,
        'Amp2': triplecdf_amp2,
        'Mean2': triplecdf_mean2,
        'StD2': triplecdf_std2,
        'Amp3': triplecdf_amp3,
        'Mean3': triplecdf_mean3,
        'StD3': triplecdf_std3,        
        'rmse': triplecdf_rmse,
    }

    return pd.DataFrame(data)

##########################
rtd_df = pd.read_pickle(dfoutput_dir + "rtd_df.pkl").reset_index(drop = True)
total_events = int(max(rtd_df.event) + 1)

rtd_t0candidate_df = rtd_df[(rtd_df.nResets >= reset_min) & (rtd_df.nResets <= reset_max)] 
rtd_allpix_df =  rtd_df[(rtd_df.nResets >= 2)]

t0_df = pd.DataFrame()

singlehit_df = pd.DataFrame()
doublehit_df = pd.DataFrame()
triplehit_df = pd.DataFrame()
unfitpix_df = pd.DataFrame()


for n in range(total_events):
    print("//////////////////////////")
    print("Event =", n)

    rtd_t0candidate_eventdf = rtd_t0candidate_df[(rtd_t0candidate_df.event == n)]
    
    if rtd_t0candidate_eventdf.empty:
        print(f"Skipping event {n} because no pixels have {reset_min}-{reset_max} resets.")
        continue
    
    #try a single CDF fit on all pixels
    singlecdf_noshift_results = process_singlecdf(rtd_t0candidate_eventdf)
    singlecdf_diff = singlecdf_noshift_results['Diff']
    print("single cdf fit pixels =", len(singlecdf_diff))
    
    #prune the data to remove obvious multi-hit pixels
    singlecdf_diff_cut = singlecdf_diff[singlecdf_diff < (np.median(singlecdf_diff) + (np.median(singlecdf_diff) - np.min(singlecdf_diff)))]
    singlecdf_diff_cut = singlecdf_diff_cut[singlecdf_diff_cut < (np.median(singlecdf_diff_cut) + (np.median(singlecdf_diff_cut) - np.min(singlecdf_diff_cut)))]

    print("well-measured pixels =", len(singlecdf_diff_cut))
    if len(singlecdf_diff_cut) == 0:
        print("Skipping Event, no well-measured pixels")
        continue
        
    hist, bin_edges = np.histogram(singlecdf_diff_cut, bins=12)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    p0 = [max(hist), np.median(bin_centers), np.std(bin_centers)]
    
    if len(singlecdf_diff_cut) > 100:
        try:
            popt, pcov = curve_fit(single_gaussian, bin_centers, hist, p0=p0)
            diff_amp = popt[0]
            diff_mean = popt[1]
            diff_std = popt[2]
            diff_low = diff_mean - 1*diff_std
            diff_high = diff_mean + 1*diff_std
        
        except RuntimeError:
            print("Gaussian Fitting of delta(sigma) distribution failed \n Using NumPy estimations")
            diff_mean = np.median(singlecdf_diff_cut)
            diff_std = np.std(singlecdf_diff_cut)
            diff_low = diff_mean - 1*diff_std
            diff_high = diff_mean + 1*diff_std
    else:
        print("Not enough well-measured pixels for Gaussian Fitting of delta(sigma) distribution \n Using NumPy estimations")
        diff_mean = np.median(singlecdf_diff_cut)
        diff_std = np.std(singlecdf_diff_cut)
        diff_low = diff_mean - 3*diff_std
        diff_high = diff_mean + 3*diff_std
    
    t0_event = singlecdf_noshift_results[(singlecdf_diff > diff_low) & (singlecdf_diff < diff_high)]
    
    if len(t0_event) < 2:
        print("Not enough well-measured pixels for t0 evaluation")
        continue

    t0_df = t0_df.append(t0_event, ignore_index=True)
    
    meanvaln = t0_event['Mean']
    stdvaln = t0_event['StD']
    
    def objective(t0_shift):
        RMS_Expected = expected_const * np.sqrt(meanvaln - t0_shift)
        difference = (stdvaln - RMS_Expected)
            
        weighted_avg = np.mean(difference)
        return (weighted_avg ** 2)
      
    # Initial guess for t0_shift (for simulation t0=0)
    initial_t0_shift = 0
    
    # Define the optimization tolerance for higher precision
    tolerance = 1e-15    
    # Define bounds for t0_shift
    bounds = [(-0.1, min(meanvaln))]

    # Perform the nonlinear optimization using the "Nelder-Mead" algorithm
    result = minimize(objective, initial_t0_shift, method='Nelder-Mead', bounds=bounds, tol=tolerance)

    # Get the optimal t0_shift value for the current event
    optimal_t0_shift = result.x[0]
    print("t0 = ", optimal_t0_shift) 
    

    #Shift the reset_times by t0 and look for hits
    rtd_allpix_eventdf = rtd_allpix_df[rtd_allpix_df.event == n]
    rtd_allpix_eventdf['t0'] = optimal_t0_shift
    print("pixels in event = ", len(rtd_df[rtd_df.event ==n]))

    #Create a singlehit dataframe for the pixels that fit well to a single hit  
    singlehit_event = process_singlehit(rtd_allpix_eventdf, optimal_t0_shift)
    print("single hit pixels in event = ", len(singlehit_event)) 
      
    if not singlehit_event.empty:
    
        singlehit_event['t0'] = optimal_t0_shift
        singlehit_df = singlehit_df.append(singlehit_event, ignore_index=True)
    
        notsinglehit_event = rtd_allpix_eventdf.merge(
            singlehit_event[['event', 'PixelID']],
            on=['event', 'PixelID'],
            how='left',
            indicator=True
        )

        notsinglehit_event = notsinglehit_event[notsinglehit_event['_merge'] == 'left_only'].drop(columns=['_merge'])
    else:
        notsinglehit_event = rtd_allpix_eventdf
        
    #Move on to pixels that don't fit well to a single hit
    
    #Create a doublehit dataframe for the pixels that fit well to a double hit
    doublehit_event = process_doublehit(notsinglehit_event[notsinglehit_event.nResets >= 4], optimal_t0_shift)
    print("double hit pixels in event = ", len(doublehit_event))
    
    if not doublehit_event.empty:

        doublehit_event['t0'] = optimal_t0_shift
        doublehit_df = doublehit_df.append(doublehit_event, ignore_index=True)
        
        notdoublehit_event = notsinglehit_event.merge(
            doublehit_event[['event', 'PixelID']],
            on=['event', 'PixelID'],
            how='left',
            indicator=True
        )

        notdoublehit_event = notdoublehit_event[notdoublehit_event['_merge'] == 'left_only'].drop(columns=['_merge'])
    else:
        notdoublehit_event = notsinglehit_event   
        
     #Move on to pixels that don't fit well to a double hit
        
    #Create a triplehit dataframe for the pixels that fit well to a triple hit
    triplehit_event = process_triplehit(notdoublehit_event[notdoublehit_event.nResets >= 6], optimal_t0_shift)
    print("triple hit pixels in event = ", len(triplehit_event))  
     
    if not triplehit_event.empty:
    
        triplehit_event['t0'] = optimal_t0_shift
        triplehit_df = triplehit_df.append(triplehit_event, ignore_index=True)
  
        #Create an unfitpix dataframe for the pixels that did not fit to single, double, or triple hits
        unfitpix_event = notdoublehit_event.merge(
            triplehit_event[['event', 'PixelID']],
            on=['event', 'PixelID'],
            how='left',
            indicator=True
        )
    
        unfitpix_event = unfitpix_event[unfitpix_event['_merge'] == 'left_only'].drop(columns=['_merge'])
    else:
        unfitpix_event = notdoublehit_event    

    print("unfit pixels = ", len(unfitpix_event))
    print("remaining pixels that could fit quadruple cdf = ", len(unfitpix_event[unfitpix_event.nResets >= 8])) 
    
    unfitpix_event = unfitpix_event.append(rtd_df[(rtd_df.event == n) & (rtd_df.nResets < 2)])   
    unfitpix_event['t0'] = optimal_t0_shift

    unfitpix_df = unfitpix_df.append(unfitpix_event)       

singlehit_df.to_pickle(t0_hitmaker_dir + '/singlehit_df.pkl')
print("List of single-hit pixels written to " + t0_hitmaker_dir + 'singlehit_df.pkl')

doublehit_df.to_pickle(t0_hitmaker_dir + '/doublehit_df.pkl')
print("List of double-hit pixels written to " + t0_hitmaker_dir + 'doublehit_df.pkl')

triplehit_df.to_pickle(t0_hitmaker_dir + '/triplehit_df.pkl')
print("List of triple-hit pixels written to " + t0_hitmaker_dir + 'triplehit_df.pkl')

unfitpix_df.to_pickle(t0_hitmaker_dir + '/unfitpix_df.pkl')
print("List of unfit pixels written to " + t0_hitmaker_dir + 'unfitpix_df.pkl')

#Reformat singlehit_df
singlehit_transformed = singlehit_df[['event', 'PixelID', 'Amp', 'Mean', 't0']]

# Reshape doublehit_df to include only relevant columns, including 't0'
doublehit_reshaped = doublehit_df.melt(
    id_vars=['event', 'PixelID', 't0'],  # Include 't0' in id_vars
    value_vars=['Amp1', 'Mean1', 'Amp2', 'Mean2'],
    var_name='Feature',
    value_name='Value'
)
doublehit_reshaped['Hit'] = doublehit_reshaped['Feature'].str[-1]  # Extract hit number
doublehit_reshaped['Feature'] = doublehit_reshaped['Feature'].str[:-1]  # Remove hit number from feature
doublehit_pivoted = doublehit_reshaped.pivot(
    index=['event', 'PixelID', 'Hit', 't0'], columns='Feature', values='Value'
).reset_index()
doublehit_pivoted = doublehit_pivoted[['event', 'PixelID', 'Amp', 'Mean', 't0']]  # Keep only relevant columns

# Reshape triplehit_df to include relevant columns, including 't0'
triplehit_reshaped = triplehit_df.melt(
    id_vars=['event', 'PixelID', 't0'],  # Include 't0' in id_vars
    value_vars=['Amp1', 'Mean1', 'Amp2', 'Mean2', 'Amp3', 'Mean3'],
    var_name='Feature',
    value_name='Value'
)
triplehit_reshaped['Hit'] = triplehit_reshaped['Feature'].str[-1]  # Extract hit number
triplehit_reshaped['Feature'] = triplehit_reshaped['Feature'].str[:-1]  # Remove hit number from feature
triplehit_pivoted = triplehit_reshaped.pivot(
    index=['event', 'PixelID', 'Hit', 't0'], columns='Feature', values='Value'
).reset_index()
triplehit_pivoted = triplehit_pivoted[['event', 'PixelID', 'Amp', 'Mean', 't0']]  # Keep only relevant columns

# Concatenate all DataFrames
hits_df = pd.concat(
    [singlehit_transformed, doublehit_pivoted, triplehit_pivoted],
    ignore_index=True
)

# Calculate 'Energy' and 'Z'
hits_df['Energy'] = hits_df['Amp'] * energy_per_reset
hits_df['Z'] = hits_df['Mean'] * elec_vel

# Drop 'Amp' and 'Mean' columns if no longer needed
hits_df = hits_df.drop(columns=['Amp'])
hits_df = hits_df.drop(columns=['Mean'])

# Merge with rtd_df to add 'X' and 'Y'
hits_df = pd.merge(hits_df, rtd_df[['event', 'PixelID', 'pixel_x', 'pixel_y']], on=['event', 'PixelID'], how='left')

# Calculate 'X' and 'Y'
hits_df['X'] = (hits_df['pixel_x'] * 4 - 2)/10 
hits_df['Y'] = (hits_df['pixel_y'] * 4 - 2)/10 

# Drop 'pixel_x' and 'pixel_y' columns
hits_df = hits_df.drop(columns=['pixel_x'])
hits_df = hits_df.drop(columns=['pixel_y'])

hits_df.to_pickle(t0_hitmaker_dir + '/hits_df.pkl')
print("List of hits written to " + t0_hitmaker_dir + 'hits_df.pkl')
