import sys
import os
import pandas as pd
import subprocess
import time
import argparse
import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, "/home/lf1071fu/project_b3")
sys.path.insert(0, "/home/lf1071fu/project_b3/ProjectB3")
import config.settings as config
from tools import utils, traj_funcs

def main(argv):

    try:
        parser = argparse.ArgumentParser()

        parser.add_argument("-p", "--path",
                            action = "store",
                            dest = "paths",
                            nargs='+',
                            default = ("unbiased_sims/holo_open/nobackup"
                                " unbiased_sims/holo_closed/nobackup"),
                            help = "Set path to the data directory.")
        parser.add_argument("-r", "--recalc",
                            action = "store_true",
                            dest = "recalc",
                            default = False,
                            help = ("Chose whether the trajectory arrays "
                                "should  be recomputed."))
        parser.add_argument("-f", "--figpath",
                            action = "store",
                            dest = "fig_path",
                            default = "",
                            help = """Set a path destination for the 
                                figure.""")
        parser.add_argument("-t", "--topol",
                            action = "store",
                            dest = "topol",
                            default = "npt.pdb",
                            help = """File name for topology, inside the 
                                path directory.""")   
        parser.add_argument("-x", "--xtc",
                            action = "store",
                            dest = "xtc",
                            default = "fitted_traj_100.xtc",
                            help = """File name for trajectory, inside 
                                the path directory.""")      
        parser.add_argument("-d", "--dataframe",
                            action = "store",
                            dest = "df",
                            default = "restraint_pts.csv",
                            help = ("File name for table of restraint "
                                "points."))                             
        args = parser.parse_args()

    except argparse.ArgumentError:
        print("""Command line arguments are ill-defined, please check the
              arguments""")
        raise

    # Assign group selection from argparse 
    wrkdir = os.getcwd()
    data_paths = [f"{ config.data_head }/{ p }" for p in args.paths.split(" ")]
    fig_path = f"{ config.figure_head }/{ args.fig_path }"
    recalc = args.recalc
    topol = args.topol
    xtc = args.xtc
    df_path = f"{ wrkdir }/{ args.df }"

    # Check for valid paths
    for p in (data_paths + [fig_path] + [df_path]):
        utils.validate_path(p)

    # Make a list of trajectory paths
    traj_paths = {}
    top_paths = {}
    for p in data_paths:
        print("\n", p, "\n")
        n = p.split("/")[-2]
        traj_paths[n] = f"{ p }/{ xtc }"
        top_paths[n] = f"{ p }/{ topol }"

    # Get a table of the restraint points
    df_pts = pd.read_csv(df_path)

    # Alignment and reference structures
    align = False
    if align:
        make_align_ref(wrkdir)
        u_open = mda.Universe(f"{ config.struct_head }/open_ref_state.pdb")
        u_closed = mda.Universe(f"{ config.struct_head }/closed_ref_state.pdb")
        traj_funcs.do_alignment(u_open)
        traj_funcs.do_alignment(u_closed)
        u_open.select_atoms("all").write(f"{ wrkdir }/open_ref_aligned.pdb")
        u_closed.select_atoms("all").write(f"{ wrkdir }/closed_ref_aligned.pdb")

    # Find nearest sample for each window
    df_cat_path = f"{ config.data_head }/cat_trajs/dataframe_beta_vec_holo.csv"
    df_cat = pd.read_csv(df_cat_path)

    # Determine the nearest sampled conformation for each restraint point
    restraint_grid = np.array((df_pts.OpenPoints, df_pts.ClosedPoints))
    ds = []
    traj_names = [None] * len(df_pts.OpenPoints)
    traj_frames = [None] * len(df_pts.OpenPoints)
    for i, g in enumerate(restraint_grid.T):
        d, d_ind = cv_min_dist(g, (df_cat["dot-open"], df_cat["dot-closed"]))
        ds.append(d)
        if d < 0.5:
            traj_names[i] = df_cat.loc[d_ind, "traj"]
            traj_frames[i] = df_cat.loc[d_ind, "ts"]
    df_pts["NearestDist"] = ds
    df_pts["NearestConformName"] = traj_names
    df_pts["NearestConformFrame"] = traj_frames

    print(df_pts)
    
    # Plot the restraint points + label points with an available initial conform
    fig, ax = plt.subplots()
    ax.scatter(df_pts["OpenPoints"], df_pts["ClosedPoints"], s=100, label="All restraint points")
    masked_df = df_pts.mask(df_pts["NearestConformName"].notna())
    ax.scatter(masked_df["OpenPoints"], masked_df["ClosedPoints"], s=50, label="Initial Conform from \nUnbiased Sampling")
    plt.legend(fontsize=20)
    ax.set_xlabel(r"$\vec{\upsilon} \cdot \vec{\upsilon}_{open}$ (nm$^2$)", labelpad=5, fontsize=24)
    ax.set_ylabel(r"$\vec{\upsilon} \cdot \vec{\upsilon}_{closed}$ (nm$^2$)", labelpad=5, fontsize=24)
    plt.savefig(f"{ wrkdir }/initial_window_points.png", dpi=300)
    plt.close()

    with open("plumed.dat", "r") as f:

        plumed_lines = f.readlines() 

    # OpenPoint,ClosedPoint 
    for i, row in df_pts.iterrows():

        w = i + 1

        plumed_restraints = []
        for line in plumed_lines:
            if "RESTRAINT_OPEN" in line:
                line = line.replace("RESTRAINT_OPEN", str(row["OpenPoints"]))
            if "RESTRAINT_CLOSED" in line:
                line = line.replace("RESTRAINT_CLOSED", str(row["ClosedPoints"]))
            if "COLVAR_WINDOW" in line:
                line = line.replace("COLVAR_WINDOW", "COLVAR_" + str(w))
            plumed_restraints.append(line)

        destination = f"{ config.data_head }/umbrella/holo_state/window{ w }"
        os.makedirs(destination, exist_ok=True)
        plumed_file = f"{ destination }/plumed_{ w }.dat"
        initial_out = f"{ destination }/initial_conform.pdb"
        index = f"{ config.data_head }/unbiased_sims/holo_closed/nobackup/index.ndx"

        if row["NearestConformName"] is not None:
            traj = traj_paths[row["NearestConformName"]]
            top = top_paths[row["NearestConformName"]]
            time_frame = row["NearestConformFrame"]

            gmx = ["echo", "24", "|", "gmx22", "trjconv", "-f", traj, "-s", top, "-o", initial_out, "-b", str(time_frame - 1000), "-dump", str(time_frame), "-n", index, "-nobackup"]
            
            output = subprocess.Popen(" ".join(gmx), stdout=subprocess.PIPE, shell=True)

            time.sleep(5)

        with open(plumed_file, "w") as f:
            f.writelines(plumed_restraints)

def cv_min_dist(grid_pt, data):
    "How far is the grid point to the nearest data point?"
    d = np.sqrt(np.sum(np.square(np.transpose(data) 
                                 - np.array(grid_pt)), axis=1))
    min_d = np.min(d)
    min_ind = np.argmin(d)
    return min_d, min_ind 

def make_align_ref(wrkdir):
    """

    Parameters
    ----------
    wrkdir : str
        Path to the working directory.

    Returns
    -------
    None.

    """
    core_res, core = traj_funcs.get_core_res() 
    ref_file = f"{ wrkdir }/ref.pdb"

    vec_select = (" or (resid 206 and name CA) or "
                  "(resid 215 and name CA))")

    print(core[:-1] + vec_select)

    u = mda.Universe(f"{ config.struct_head }/ref_all_atoms.pdb")
    core_and_vec = u.select_atoms(core[:-1] + vec_select)
    core_and_vec.write(ref_file)
    print(f"Wrote reference alignment structure to { ref_file }.")

    return None

if __name__ == '__main__':
    main(sys.argv)