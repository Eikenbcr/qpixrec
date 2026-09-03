#!/usr/bin/env python

# -----------------------------------------------------------------------------
# run_analysis.py
#
# Runs the qpixrec lightless reconstruction scripts in sequential order
# * Author: Carter Eikenbary
# * Creation date: 2 December 2024
#
# Usage: python /path/to/run_analysis.py /path/to/root/file.root -threshold # -rmin # -rmax #
# Notes: HPRC users must load foss/2022b and source qpix-setup before running this script
# -----------------------------------------------------------------------------

import sys
import os
import argparse
import subprocess

# GLOBAL DEFINITIONS
run_path = os.path.dirname(os.path.abspath(__file__))

def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Run qpixrec lightless reconstruction scripts.")
    
    # Add positional argument for the root file
    parser.add_argument("root_file", type=str, help="Path to the root file.")
    
    # Add optional arguments for rmin and rmax
    parser.add_argument("-threshold", type=int, default=6250, help="Electron threshold for QPix reset (default: 6250).")
    parser.add_argument("-rmin", type=int, default=3, help="Minimum resets for t0 evaluation (default: 3).")
    parser.add_argument("-rmax", type=int, default=5, help="Maximum resets for t0 evaluation (default: 5).")
    parser.add_argument("-clockspeed", type=str, default="1e-8", help="Clock speed in sec (default: 1e-8).")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Extract arguments
    root_file = args.root_file
    reset_threshold = args.threshold
    rmin = args.rmin
    rmax = args.rmax
    clockspeed = args.clockspeed
    
    # Define output directories
    file_path = os.path.dirname(root_file) + "/"
    dfoutput_dir = file_path + "rtd_dataframe/"
    t0_hitmaker_dir = file_path + "t0_hitmaker/"
    
    # Create directories if they don't exist
    output_dirs = [dfoutput_dir, t0_hitmaker_dir]
    for output_dir in output_dirs:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            print(f"The directory '{output_dir}' already exists.")
    
    # List of scripts to run
    scripts_to_run = [
        (run_path + "/root_to_pandas.py", root_file, dfoutput_dir),
        (run_path + "/t0_hitmaker.py", dfoutput_dir, t0_hitmaker_dir, str(reset_threshold), str(rmin), str(rmax), str(clockspeed)),
    ]

    # Execute each script
    for script_info in scripts_to_run:
        script_name, *args = script_info
        print(f"Running script '{script_name}'")
        execute_script(script_name, *args)

def execute_script(script_name, *args):
    cmd = ["python", script_name, *args]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"Script '{script_name}' executed successfully!")
        print("Output:")
        print(result.stdout)
    else:
        print(f"An error occurred while running '{script_name}'.")
        print("Error message:")
        print(result.stderr)

if __name__ == "__main__":
    main()
