import numpy as np
import pandas as pd
import argparse
import os
import sys

def main(argv):

    try:
        parser = argparse.ArgumentParser()

        parser.add_argument("-i", "--iteration",
                            action = "store",
                            dest = "iteration",
                            default = 1,
                            help = ("Set an integer value to obtain "
                                "the next iteration of initial conforms"
                                "for the umbrella windows. Counts from "
                                "1."))
        parser.add_argument("-r", "--recalc",
                            action = "store_true",
                            dest = "recalc",
                            default = False,
                            help = ("Overwrite the existing csv for the "
                                "selected iteration if it already "
                                "exists."))
        args = parser.parse_args()

    except argparse.ArgumentError:
        print("Command line arguments are ill-defined, please check the "
            "arguments")
        raise

    # Assign command line arg
    iteration = int(args.iteration)
    recalc = args.recalc

    # Get relevant restraint points
    df_pts = pd.read_csv(f"restraint_pts_iteration{ iteration }.csv")

    # Check if the current iteration already exists
    new_pts_file = f"restraint_pts_iteration{ iteration + 1 }.csv"
    if os.path.exists(new_pts_file) and not recalc:
        print("Are you sure you want to overwrite the previous "
            "iteration? Add `-r` recalc flag to overrule, exiting now.")
        sys.exit(1)

    # Get sampled collective variables in one table
    df_cat = pd.DataFrame(columns=["time", "dot-open", "dot-closed"])
    for w in range(1,151):
        for r in range(1,5):
            file = f"window{ w }/run{ r }/COLVAR_{ w }.dat"
            if os.path.exists(file):
                df_new = add_colvar_data(w, r, file)

                df_cat = pd.concat([df_cat, df_new])

            if os.path.exists(os.path.dirname(file)):
                df_pts.loc[df_pts["Window"] == w, "InitialConform"] = True

    # Find all the window restraint points which have no COLVAR data.
    # These are the windows which still need an initial conform
    needs_conform = df_pts[df_pts["InitialConform"] == False]

    # Print the DataFrame of sampled reactions coords data
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    print(df_cat)

    # Set up some variables
    restraint_grid = np.array((needs_conform.OpenPoints, 
                            needs_conform.ClosedPoints,
                            needs_conform.Window))

    df_pts = df_pts.assign(NearestWindow=False, NearestRun=False,
                       NearestFrame=False)

    # Iterate over all the restraints to find the closest conform
    # from the sampled windows
    for g in restraint_grid.T:
        d, d_ind = cv_min_dist(g[:2], (df_cat["dot-open"], 
                                    df_cat["dot-closed"]))
        
        # If a threshold is met, record the trajectory and time frame to
        # access an initial conformation for each missing window
        if d < 0.5:
            w, r, t = df_cat.iloc[d_ind][["window", "run", "time"]]
            df_pts.loc[df_pts["Window"] == g[2], "NearestWindow"] = w
            df_pts.loc[df_pts["Window"] == g[2], "NearestRun"] = r
            df_pts.loc[df_pts["Window"] == g[2], "NearestFrame"] = t

    # New DataFrame to use for extracting data
    df_pts.to_csv(new_pts_file)

    # # Initialize window dirs with new conformations
    # for i, row in df_pts.iterrows():

    #     w = row["Window"]

    #     destination = f"window{ w }"
    #     os.makedirs(destination, exist_ok=True)
    #     plumed_file = f"{ destination }/plumed_{ w }.dat"
    #     initial_out = f"{ destination }/initial_conform.pdb"

    return None

def cv_min_dist(grid_pt, data):
    """Finds the nearest sampled data to the desired restraint point.

    Parameters
    ----------
    grid_pt : (float, float)
        The restraint point which needs an initial conformation. Give 
        point as (open-beta-vec, closed-beta-vec).
    data : np.ndarray
        A 2D array of the sampled open-beta-vec and closed-beta-vec. Note 
        that the data is transposed before combining with the restraint 
        point.

    Returns
    -------
    min_d : float
        The distance of the restraint point to the nearest sampled 
        conformation.
    min_ind : int
        The index of the point with minimum distance, which can be used 
        to find the nearest sampled conformation.
    """
    d = np.sqrt(np.sum(np.square(np.transpose(data) 
                                 - np.array(grid_pt)), axis=1))
    min_d = np.min(d)
    min_ind = np.argmin(d)
    return min_d, min_ind 

def add_colvar_data(window, run, file):
    """Adds COLVAR data to a DataFrame from the sampled window.

    Parameters
    ----------
    window : int
        The window of the sampled data.
    run : int
        The run (1-4) of the sampled data. 
    file : str
        The COLVAR file which can be read into a DataFrame.

    Returns
    df : pd.DataFrame
        The DataFrame object with data for the given window + run.

    """
    df = pd.read_csv(file, comment="#", delim_whitespace=True, 
                       names=["time", "dot-open", "dot-closed",
                              "res-bias", "res-force"])
    df["window"] = window
    df["run"] = run

    return df[10:10000]

if __name__ == "__main__":
    main(sys.argv)