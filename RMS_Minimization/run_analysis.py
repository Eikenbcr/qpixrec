#!/usr/bin/env python

# -----------------------------------------------------------------------------
# run_analysis.py
#
# Runs the qpixrec lightless reconstruction scripts in sequential order (root_to_pandas.py, t0_hitmaker.py)
# * Author: Carter Eikenbary
# * Creation date: 12 January 2024
#
# Usage: python /path/to/run_analysis.py /path/to/root/ file.root -v
# Notes: HPRC users must load foss/2020b and source qpix-setup before running this script
# -----------------------------------------------------------------------------

####GLOBAL DEFINITIONS####
import sys
import os

run_path = os.path.dirname(os.path.abspath(__file__))

file_path = sys.argv[1]
root_file = file_path + sys.argv[2]
if len(sys.argv) > 3:
    verbosity = sys.argv[3]
else:
    verbosity = "false"
    
dfoutput_dir = file_path + "rtd_dataframes/"
t0_hitmaker_dir = file_path + "t0_hitmaker/"

# Define bin width to be used for distribution plots and Gaussian fits
binWidth = "8e-05"

##########################

import subprocess

output_dirs = [
    dfoutput_dir,
    t0_hitmaker_dir
]

for output_dir in output_dirs:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        print(f"The directory '{output_dir}' already exists.")

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
    # List of analysis scripts along with there system arguments
    scripts_to_run = [
        (run_path +"/root_to_pandas.py", root_file, dfoutput_dir),
        (run_path +"/t0_hitmaker.py", dfoutput_dir, t0_hitmaker_dir, binWidth, verbosity)
    ]

    for script_info in scripts_to_run:
        script_name, *args = script_info
        print(f"Running script '{script_name}'")
        execute_script(script_name, *args)
