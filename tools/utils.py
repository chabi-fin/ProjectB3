# utils.py

import os
import sys
import datetime
import matplotlib.pyplot as plt

def validate_path(path):
    """Validate if a given path exists."""
    if not os.path.exists(path):
        print(f"Error: The path '{path}' does not exist.")
        sys.exit(1)

def create_path(path):
    """Create a directory or file if it doesn't exist."""
    if not os.path.exists(path):
        try:
            # Creates intermediate directories if needed
            os.makedirs(path)  
            print(f"Path '{path}' created successfully.")
        except Exception as e:
            print(f"Error: Failed to create path '{path}': {e}")
            sys.exit(1)

def save_figure(figure, fig_path, log_name="figure_log.txt"):
    """Save a Matplotlib figure and log the information in a log file.

    Parameters
    ----------
    figure : 
        Matplotlib figure object
    fig_path : str
        Name of the figure (used in the log)
    log_name : str 
        Name of the log file (default is "figure_log.txt")

    Returns
    -------
    None.

    """
    # Get the current date and time
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Get the script name (assuming this function is called from a 
    # script)
    script_name = os.path.basename(sys.argv[0])

    # Save the figure
    figure.savefig(fig_path, dpi=300)
    fig_name = os.path.basename(fig_path)

    # Log the information
    log_file = f"{ os.path.dirname(fig_path) }/{ log_name }"
    log_entry = (f"{ current_time } - Script: { script_name },"
                 f" Figure: { fig_name }")

    # Check if the log file exists
    if os.path.exists(log_file):

        # If the log file exists, remove the previous log entry and
        # append the new entry
        with open(log_file, "r+") as f:
            existing_entries = f.readlines()
            filtered_lines = [e for e in existing_entries if fig_name not in e]
            f.seek(0)
            f.writelines(filtered_lines)
            f.write(log_entry + "\n")

    # If the log file doesn't exist, create it and write the log entry
    else:
        with open(log_file, "w") as f:
            f.write(log_entry + "\n")

    return None
