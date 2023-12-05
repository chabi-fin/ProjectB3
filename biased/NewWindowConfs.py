import numpy as np
import pandas as pd
import os

def cv_min_dist(grid_pt, data):
    """How far is the grid point to the nearest data point?
    """
    d = np.sqrt(np.sum(np.square(np.transpose(data) 
                                 - np.array(grid_pt)), axis=1))
    min_d = np.min(d)
    min_ind = np.argmin(d)
    return min_d, min_ind 

def add_colvar_data(window, run, file):
    """Adds COLVAR data from the sampled window.
    """
    df = pd.read_csv(file, comment="#", delim_whitespace=True, 
                       names=["time", "dot-open", "dot-closed",
                              "res-bias", "res-force"])
    df["window"] = window
    df["run"] = run

    return df

# Get relevant restraint points
df_pts = pd.read_csv("restraint_pts.csv")
needs_conform = df_pts[df_pts["InitialConform"] == False]

# Get sampled collective variables in one table
df_cat = pd.DataFrame(columns=["time", "dot-open", "dot-closed"])
for w in range(1,151):
    for r in range(1,5):
        file = f"window{ w }/run{ r }/COLVAR_{ w }.dat"
        if os.path.exists(file):
            df_new = add_colvar_data(w, r, file)

            df_cat = pd.concat([df_cat, df_new])

print(df_cat)

# Set up some variables
restraint_grid = np.array((needs_conform.OpenPoints, 
                           needs_conform.ClosedPoints,
                           needs_conform.Window))

print("\n", restraint_grid)
df_pts[["NearestWindow", "NearestRun", "NearestFrame"]] = False

# Iterate over all the restraints to find the closest conform
# from the sampled windows
for g in restraint_grid.T:
    d, d_ind = cv_min_dist(g[:2], (df_cat["dot-open"], 
                                   df_cat["dot-closed"]))
    
    # If a threshold is met, record the trajectory and time frame to
    # access an initial conformation for each missing window
    if d < 0.5:
        w = df[["Window"] == g[2]]
        df_pts.loc[w, "NearestWindow"] = df_cat.loc[d_ind, "window"]
        df_pts.loc[w, "NearestRun"] = df_cat.loc[d_ind, "run"]
        df_pts.loc[w, "NearestFrame"] = df_cat.loc[d_ind, "time"]

# New DataFrame to use for extracting data
df_pts.to_csv("restraint_pts_iteration2.csv")

# # Initialize window dirs with new conformations
# for i, row in df_pts.iterrows():

#     w = row["Window"]

#     destination = f"window{ w }"
#     os.makedirs(destination, exist_ok=True)
#     plumed_file = f"{ destination }/plumed_{ w }.dat"
#     initial_out = f"{ destination }/initial_conform.pdb"
