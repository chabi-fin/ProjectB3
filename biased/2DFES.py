import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import plumed
import wham
import MDAnalysis as mda
from MDAnalysis.analysis import align
import subprocess    
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.tri import Triangulation
from scipy.signal import find_peaks
import argparse
import pandas as pd
sys.path.insert(0, "/home/lf1071fu/project_b3")
sys.path.insert(0, "/home/lf1071fu/project_b3/ProjectB3")
import config.settings as config
from tools import utils, traj_funcs

def main(argv):

    try:
        parser = argparse.ArgumentParser()

        parser.add_argument("-p", "--path",
                            action = "store",
                            dest = "path",
                            default = ("umbrella/holo_state/nobackup"),
                            help = ("Set relative path to the data "
                                "directory."))        
        parser.add_argument("-a", "--angle",
                            action = "store_true",
                            dest = "angle_coord",
                            default = False,
                            help = ("Chose whether the reaction "
                                "coordinate should be converted into an "
                                "angle."))
        parser.add_argument("-f", "--figpath",
                            action = "store",
                            dest = "fig_path",
                            default = ("umbrella/holo_state"),
                            help = ("Set a relative path destination for"
                                " the figure."))  
        parser.add_argument("-b", "--blocks",
                            action = "store",
                            dest = "nblocks",
                            default = 100,
                            help = ("Number of blocks to use in "
                                "bootstrap analysis; 1 ns each by "
                                "default."))            
        parser.add_argument("-r", "--recalc",
                            action = "store_true",
                            dest = "recalc",
                            default = False,
                            help = ("Chose whether the reweighting "
                                "should be recomputed."))
        parser.add_argument("-n", "--nobootstrap",
                            action = "store_true",
                            dest = "no_bootstrap",
                            default = True,
                            help = ("Compute the free energy surface "
                                "without error analysis."))
        parser.add_argument("-w", "--num_windows",
                            action = "store",
                            dest = "num_windows",
                            default = 170,
                            help = ("Number of windows used in the "))
        args = parser.parse_args()

    except argparse.ArgumentError:
        print("Command line arguments are ill-defined, please check the "
            "arguments.")
        raise

    global nb, cv, nw, kBT, angle_coord, recalc, home, colvar_columns

    # Set key path variables
    home = f"{ config.data_head  }/{ args.path }"
    fig_path = f"{ config.figure_head }/{ args.fig_path }"
    struct_path = config.struct_head

    # Set other variables
    angle_coord = args.angle_coord
    recalc = args.recalc
    no_bs = args.no_bootstrap # don't do error analysis
    nb = int(args.nblocks) # number of blocks for bootstrapping
    nw = int(args.num_windows) # number of windows 
    bs = 10 # number of bootstraps
    kBT = 310 * 8.314462618 * 0.001 # use kJ/mol here

    # Helpful objects for structure alignment
    core_res, core = traj_funcs.get_core_res()
    ref_state = mda.Universe(f"{ struct_path }/ref_all_atoms.pdb", 
                             length_unit="nm")

    vec_open, vec_closed = get_ref_vecs(struct_path, core, ref_state)
    colvar_columns = ["time", "opendot", "closeddot", "theta1", "theta2", 
                      "d4", "d5", "d6", "bias", "force"]

    np_ave_files = [f"{ home }/hist_ave.npy", 
                    f"{ home }/hist_sq_ave.npy", 
                    f"{ home }/bin_counter.npy"]
    bs_path = f"{ home }/bootstrap_files"
    if not os.path.exists(bs_path):
        os.makedirs(bs_path)

    # Make the FES without error estimates 
    if True:

        fes = get_no_bs_fes(home, bs_path)

        # Make the 2D FES plot
        plot_2dfes(fes, vec_open, vec_closed, fig_path)
        
        # Also make the 1D FES plots
        for rxn_coord in ["open", "closed", "sb"]:
            plot_1dfes(bs_path, rxn_coord, fig_path)

    elif not all(list(map(lambda x : os.path.exists(x), np_ave_files))) or recalc: 

        # # Get the biasing data from COLVAR files
        bias, frames = get_bias_data(home, recalc=False)

        print(bias)
        print(f"SIZE OF BIAS { np.shape(bias) }")

        # Get collective variable data from any COLVAR file
        df = pd.read_csv(f"{ home }/COLVAR_1.dat", delim_whitespace=True,
                         comment='#', names=colvar_columns) 

        # Reshape the bias into blocks that can be used for bootstrapping
        bb = bias.reshape((nb, frames // nb, nw))

        # Initialize arrays for the histogram data
        ave, ave_sq, count = np.zeros(170**2), np.zeros(170**2), np.zeros(170**2)

        # # Construct bootstraps and determine reweighted histograms
        for i in range(bs):
            print(i)

            # Choose random blocks for bootstap
            c=np.random.choice(nb, nb)
            c = np.arange(0, nb)

            pfes_file = f"{ bs_path }/fes_catr_{ i }.dat"
            hist_file = f"{ bs_path }/hist_catr_{ i }.dat"

            if not os.path.exists(pfes_file) or recalc:

                # Get the log weights for reweighting with WHAM 
                w = wham.wham(bb[c,:,:].reshape((-1, nw)), T=kBT)

                # Write the logweights to a pandas table for reweighting
                colvar_weights = pd.DataFrame()
                for col in df.columns:
                    colvar_weights[col] = bs_cols(df, col, frames, nb , c) 
                colvar_weights["logweights"] = w["logW"]

                # Use plumed to make reweighted histograms
                make_ith_histogram(colvar_weights, bs_path, i)

            # Load in free energy estimates determined via plumed
            pfes = plumed.read_as_pandas(pfes_file).replace([np.inf, -np.inf], np.nan)

            # Use the ith histogram to estimate average probability densities
            # mask = ~np.isnan(pfes.ffr) # some bins will be NAN
            # ave = np.nansum([ave, pfes.ffr], axis=0) 
            # ave_sq = np.nansum([ave_sq, pfes.ffr*pfes.ffr], axis=0)
            # count[mask] += 1 # only bins with a bootstrap estimate should have the count increased

            ffr = pfes.ffr.astype(float)
            ave += ffr
            ave_sq += ffr*ffr

        for file, arr in zip(np_ave_files, [ave, ave_sq, count]):
            utils.save_array(file, arr)

    # Load calculated values from pandas tables and numpy arrays
    else:

        # Get bin positions for the CVs
        pfes = plumed.read_as_pandas(f"{ bs_path }/"
                        "fes_catr_0.dat").replace([np.inf, -np.inf], np.nan) #.dropna()

        # Load the sum of the free energy esimates from each bin
        ave = np.load(np_ave_files[0], allow_pickle=True)
        ave_sq = np.load(np_ave_files[1], allow_pickle=True)
        count = np.load(np_ave_files[2], allow_pickle=True)

    # Calculate free energy surface from histogram averages and 
    # free energy errors from the histograms, see expressions in ex-5 
    # https://www.plumed.org/doc-v2.8/user-doc/html/masterclass-21-2.html
    # ave = ave / bs
    # # fes = convert_fes(hist.odot, hist.cdot, ave)

    # var = (1 / (count - 1)) * ( ave_sq / count - ave * ave ) 
    # fes = - kBT * np.log(ave)
    # error = np.sqrt( var )
    # ferr = error / ave

    # Make the 2D FES plot
    # plot_2dfes(pfes.odot, pfes.cdot, fes, ferr)

    # Make the error estimate plot for the 2D FES

    return None

def plot_2dfes(fes, vec_open, vec_closed, fig_path):
    """Plots the 2D FES as a colorbar + contour map.

    fes : pd.DataFrame
        The table containing discretized free energy surface data. 
    vec_open : float
        The reference beta-vector for the open conformation.
    vec_closed : float
        The reference beta-vector for the closed conformation.
    fig_path : str
        Path for storing the figure. 
    """
    fig, ax = plt.subplots(constrained_layout=True, figsize=(12,8))

    # Get the relevant discretized arrays from table columns
    open_bins, closed_bins = fes.odot, fes.cdot 
    fes, ferr = fes.ffr, None

    mask = ((-1e2 < fes) & (fes < 1e2))
    # mask = (fes < 1e6)

    x, y = open_bins[mask], closed_bins[mask]
    z = fes[mask]
    #x, y = open_bins, closed_bins
    #z = fes
    # z, err = fes[mask], ferr[mask]

    d = ax.scatter(x, y, c=z, cmap=plt.cm.viridis)

    # if angle_coord:
    #     fes_r=plumed.read_as_pandas("fes_theta_catr.dat").replace([np.inf, -np.inf], np.nan).dropna()
    #     x = [ x for x, z in zip(fes_r.theta1, fes_r.ffthetar) if -80 < z < 80 ]
    #     y = [ y for y, z in zip(fes_r.theta2, fes_r.ffthetar) if -80 < z < 80 ]
    #     zmin_g = np.min(np.array([f for f in fes_r.ffthetar if f > -80]))
    #     zmin_l = np.min(np.array([f for x, f in zip(fes_r.theta1, fes_r.ffthetar) if ((f > -80) and (x < 0.75))]))
    #     z = [ z - zmin_g for z in fes_r.ffthetar if -80 < z < 80 ]

    # else:
    # fes_r=plumed.read_as_pandas("fes_catr.dat").replace([np.inf, -np.inf], np.nan).dropna()
    # x = [ x for x, z in zip(fes_r.opendot, fes_r.ffr) if -80 < z < 80 ]
    # y = [ y for y, z in zip(fes_r.closeddot, fes_r.ffr) if -80 < z < 80 ]
    # zmin_g = np.min(np.array([f for f in fes_r.ffr if f > -80]))
    # zmin_l = np.min(np.array([f for x, f in zip(fes_r.opendot, fes_r.ffr) if ((f > -80) and (x > 4))]))
    # z = [ z - zmin_g for z in fes_r.ffr if -80 < z < 80 ]
    # print(f"Global min: { zmin_g } kJ/mol\tLocal min: { zmin_l } kJ/mol")

    tri = Triangulation(x, y)

    # Create contour lines on the XY plane using tricontour
    contours = ax.tricontour(tri, z, cmap="inferno")

    # Colormap settings
    cbar = plt.colorbar(d)
    # if angle_coord:
    #     cbar.set_label(r'$F(\theta_{open}, \theta_{closed})$ (kJ / mol)', fontsize=24, labelpad=10)
    # else: 
    cbar.set_label(r'$F(\vec{\upsilon} \cdot \vec{\upsilon}_{open}, \vec{\upsilon} \cdot \vec{\upsilon}_{closed})$ (kJ / mol)', fontsize=28, labelpad=10)
    cbar.ax.tick_params(labelsize=18, direction='out', width=2, length=5)
    cbar.outline.set_linewidth(2)
    ax.clabel(contours, inline=1, fontsize=20)

    # Find minima
    # minima_indices, _ = find_peaks(-np.array(z), prominence=70)
    # mins = np.array(minima_indices)
    # xmins = [x[i] for i in mins]
    # ymins = [y[i] for i in mins]

    # Add both reference positions
    if angle_coord:
        ax.scatter(calc_theta(vec_open, vec_open), calc_theta(vec_open, vec_closed), 
                label="Open ref.", marker="X", alpha=1, edgecolors="#404040", 
                s=550, lw=3, color="#EAAFCC")
        ax.scatter(calc_theta(vec_open, vec_closed), 0, label="Closed ref.", 
                marker="X", alpha=1, edgecolors="#404040", s=550, lw=3, color="#A1DEA1")
    else:
        ax.scatter(np.dot(vec_open, vec_open), np.dot(vec_open, vec_closed), 
                label="Open ref.", marker="X", alpha=1, edgecolors="#404040", 
                s=550, lw=3, color="#EAAFCC")
        ax.scatter(np.dot(vec_open, vec_closed), np.dot(vec_closed, vec_closed), 
                label="Closed ref.", marker="X", alpha=1, edgecolors="#404040", 
                s=550, lw=3, color="#A1DEA1")

    ax.tick_params(axis='y', labelsize=18, direction='in', width=2, \
                    length=5, pad=10)
    ax.tick_params(axis='x', labelsize=18, direction='in', width=2, \
                    length=5, pad=10)
    if angle_coord:
        ax.set_xlabel(r"$\theta_{open}$ (rad)", labelpad=5, fontsize=24)
        ax.set_ylabel(r"$\theta_{closed}$ (rad)", labelpad=5, fontsize=24)
    else: 
        ax.set_xlabel(r"$\vec{\upsilon} \cdot \vec{\upsilon}_{open}$ (nm$^2$)", labelpad=5, fontsize=24)
        ax.set_ylabel(r"$\vec{\upsilon} \cdot \vec{\upsilon}_{closed}$ (nm$^2$)", labelpad=5, fontsize=24)
    _, xmax = ax.get_xlim()
    _, ymax = ax.get_ylim()
    ax.set_xlim(0,xmax)
    ax.set_ylim(0,ymax)
    plt.legend(fontsize=24)

    if angle_coord:
        utils.save_figure(fig, f"{ fig_path }/2dfes_angles.png")
    else:
        utils.save_figure(fig, f"{ fig_path }/2dfes.png")

def get_ref_vecs(struct_path, core, ref_state):
    """Calculates the beta-vectors for the reference structures.

    Parameters
    ----------
    struct_path : str
        Path to the structure directory, i.e. project_b3/structures
    core : str 
        The core residues used for alignment throughout this project.
    ref_state: MDAnalysis.Universe
        The reference structure object.

    Returns
    -------
    vec_open : np.ndarray
        The beta vector in Angstrom for the aligned open reference state.
    vec_closed : np.ndarray
        The beta vector in Angstrom for the aligned closed reference state.

    """
    # Load in relevant reference structures
    open_ref = mda.Universe(f"{ struct_path }/open_ref_state.pdb", length_unit="nm")
    closed_ref = mda.Universe(f"{ struct_path }/closed_ref_state.pdb", length_unit="nm")

    # Align the traj and ref states to one structure
    align.AlignTraj(open_ref, ref_state, select=core, in_memory=True).run()
    align.AlignTraj(closed_ref, ref_state, select=core, in_memory=True).run()

    # Determine open + closed reference beta flap vectors in units of Angstrom
    r1, r2 = 206, 215
    r1_open = open_ref.select_atoms(f"name CA and resnum { r1 }").positions[0]
    r2_open = open_ref.select_atoms(f"name CA and resnum { r2 }").positions[0]
    vec_open = r2_open/10 - r1_open/10
    r1_closed = closed_ref.select_atoms(f"name CA and resnum { r1 }").positions[0]
    r2_closed = closed_ref.select_atoms(f"name CA and resnum { r2 }").positions[0]
    vec_closed = r2_closed/10 - r1_closed/10

    return vec_open, vec_closed

def calc_theta(vec_ref, vec_sim):
    """Determine the angle between two 3D vectors.
    
    Solves the expression theta = cos^(-1)((A · B) / (|A| * |B|))

    Parameters
    ----------
    vec_ref : nd.array
        The referenece beta vector as a 3D array.

    vec_sim : nd.array
        An instance of the simulated beta as a 3D array. 

    Returns
    -------
    theta : float
        The angle formed by the two vectors, in radians.

    """
    # Calculate the dot product of A and B
    dot_product = np.dot(vec_ref, vec_sim)

    # Calculate the magnitudes of A and B
    magnitude_ref = np.linalg.norm(vec_ref)
    magnitude_sim = np.linalg.norm(vec_sim)

    # Calculate the angle (theta) between A and B using the formula
    theta = np.arccos(dot_product / (magnitude_ref * magnitude_sim))

    return theta

def plot_1dfes(bs_path, rxn_coord, fig_path):
    """Makes a plot against the 1D reaction coordinate.

    Parameters
    ----------
    rxn_coord : str
        Name of the reaction coordinate should match the column name.
    fig_path : str
        Path to figure directory for particular fes. 

    Returns
    -------
    None.

    """
    # Initialize the plot
    fig, ax = plt.subplots(constrained_layout=True, figsize=(12,8))

    # Load table containing discretized free energy surface data.
    fes = plumed.read_as_pandas(f"{ bs_path }/fes1d_{ rxn_coord }_fulldata.dat")
    fes = fes.replace([np.inf, -np.inf], np.nan).dropna()

    # Labels for axes for the reaction coordinates
    rxn_coord_labs = {
        "open" : r"$\vec{\upsilon} \cdot \vec{\upsilon}_{open}$",
        "closed" : r"$\vec{\upsilon} \cdot \vec{\upsilon}_{closed}$",
        "sb" : "K57--E200"
        }
    # Select reaction coordinate and lable axes accordingly
    if rxn_coord == "open":
        ax.plot(fes["odot"], fes["ffr1d_open"], label="reweighted FES")
        ax.set_xlabel(f"{rxn_coord_labs[rxn_coord]} (nm$^2$)", labelpad=5, 
                      fontsize=24)
    elif rxn_coord == "closed":
        ax.plot(fes["cdot"], fes["ffr1d_closed"], label="reweighted FES")
        ax.set_xlabel(f"{rxn_coord_labs[rxn_coord]} (nm$^2$)", labelpad=5, 
                      fontsize=24)
    elif rxn_coord == "sb":
        ax.plot(fes["sb"], fes["ffr1d_sb"], label="reweighted FES")
        ax.set_xlabel(f"{rxn_coord_labs[rxn_coord]} (nm)", labelpad=5, 
                    fontsize=24)

    ax.set_ylabel(f"F({rxn_coord_labs[rxn_coord]}) (kJ / mol)", labelpad=5, 
                    fontsize=24)
    
    # Plot settings
    plt.legend(fontsize=18)
    _, xmax = ax.get_xlim()
    _, ymax = ax.get_ylim()
    ax.set_xlim(0,xmax)
    ax.set_ylim(0,ymax)

    # Save figure and close figure object
    utils.save_figure(fig, f"{ fig_path }/fes_{ rxn_coord }.png")
    plt.close()

    return None

def get_bias_data(home, recalc=False):
    """Get the 2D array of bias data from all windows.

    Parameters
    ----------
    recalc : bool
        Redetermine the collective variable from plumed files rather 
        than loading arrays from as numpy files.

    Returns
    -------
    bias : np.ndarray
        The bias applied to the full, concatenated trajectory for each 
        window, dim = (n-frames-concat, n-windows).
    frames : int
        The number of frames in the full concatenated trajectory.
    """
    bias_file = f"{ home }/bias_concat.npy"

    if not os.path.exists(bias_file) or recalc: 

        # Add all the COLVAR data to one large DataFrame 
        # NB requires a lot of memory!
        for i in range(nw):
            df = pd.read_csv(f"{ home }/COLVAR_" + str(i+1)+".dat",
                       delim_whitespace=True, comment='#')
            if "bias" not in locals():
                frames = len(df.iloc[:,-2])
                print("Number of frames", frames)
                bias = np.zeros((frames, nw))
            # Reshape the bias array
            bias[:,i] = df.iloc[:,-2]

        print(f"BIAS ARRAY {bias.shape}\n")

        utils.save_array(bias_file, bias)

    else:

        bias = np.load(bias_file, allow_pickle=True)
        frames = len(bias[:,0])

    return bias, frames

def make_ith_histogram(df, bs_path, i):
    """Use plumed HISTOGRAM function to obtain reweighted histograms for 
    the bootstrap.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the collective variables and logweights.
    bs_path : str
        Path to directory for bootstraping data.
    i : int
        Integer for the bootstrap.

    Returns
    -------
    None.

    """
    print(df, df.head())
    plumed.write_pandas(df, f"{ bs_path }/colvar_weights_{ i }.dat")

    with open(f"{ bs_path }/colvar_histograms_{ i }.dat","w") as f:
        print(f"""
    # vim:ft=plumed
    odot: READ FILE={ bs_path }/colvar_weights_{ i }.dat VALUES=opendot IGNORE_TIME
    cdot: READ FILE={ bs_path }/colvar_weights_{ i }.dat VALUES=closeddot IGNORE_TIME
    theta1: READ FILE={ bs_path }/colvar_weights_{ i }.dat VALUES=theta1 IGNORE_TIME
    theta2: READ FILE={ bs_path }/colvar_weights_{ i }.dat VALUES=theta2 IGNORE_TIME
    sb: READ FILE={ bs_path }/colvar_weights_{ i }.dat VALUES=saltbridge IGNORE_TIME

    lw: READ FILE={ bs_path }/colvar_weights_{ i }.dat VALUES=logweights IGNORE_TIME

    hhr: HISTOGRAM ARG=odot,cdot GRID_MIN=0,0 GRID_MAX=6,6 GRID_BIN=150,150 BANDWIDTH=0.05,0.05 LOGWEIGHTS=lw
    DUMPGRID GRID=hhr FILE={ bs_path }/hist_catr_{ i }.dat
    ffr: CONVERT_TO_FES GRID=hhr 
    DUMPGRID GRID=ffr FILE={ bs_path }/fes_catr_{ i }.dat

    hhr1d_open: HISTOGRAM ARG=odot GRID_MIN=0 GRID_MAX=6 GRID_BIN=100 BANDWIDTH=0.05 LOGWEIGHTS=lw
    ffr1d_open: CONVERT_TO_FES GRID=hhr1d_open
    DUMPGRID GRID=ffr1d_open FILE={ bs_path }/fes1d_open_{ i }.dat

    hhr1d_closed: HISTOGRAM ARG=cdot GRID_MIN=0 GRID_MAX=6 GRID_BIN=100 BANDWIDTH=0.05 LOGWEIGHTS=lw
    ffr1d_closed: CONVERT_TO_FES GRID=hhr1d_closed
    DUMPGRID GRID=ffr1d_closed FILE={ bs_path }/fes1d_closed_{ i }.dat

    hhr1d_sb: HISTOGRAM ARG=sb GRID_MIN=0 GRID_MAX=2.5 GRID_BIN=100 BANDWIDTH=0.05 LOGWEIGHTS=lw
    ffr1d_sb: CONVERT_TO_FES GRID=hhr1d_sb
    DUMPGRID GRID=ffr1d_sb FILE={ bs_path }/fes1d_sb_{ i }.dat
    """, file=f)

    subprocess.run((f"plumed driver --noatoms --plumed { bs_path }/colvar" 
        f"_histograms_{ i }.dat --kt { kBT }"), shell=True)

    return None

def get_no_bs_fes(home, bs_path):
    """Estimate the free energy surface, without bootstrapping.

    Parameters
    ----------
    home : str
        Path to the primary data path, containing plumed-driver output.
    bs_path : str
        Path to bootstrapping calculations directory.

    Returns
    -------
    fes : pd.DataFrame
        Pandas table with collective variable data and free energy 
        estimates in kJ/mol.

    """
    fes_dat = f"{ bs_path }/fes_catr_fulldata.dat"

    if not os.path.exists(fes_dat) or recalc:

        # Get the biasing data from COLVAR files
        bias, frames = get_bias_data(home, recalc=False)

        # Get collective variable data from any COLVAR file
        df = pd.read_csv(f"{ home }/COLVAR_1.dat", delim_whitespace=True,
                         comment='#', names=colvar_columns) 
        df["saltbridge"] = np.minimum(df["d5"], df["d6"])

        # Reshape data for bootstrapping; this actually does nothing here
        # since the original column order is retained in "c", but the 
        # bootstrapping step is still performed for consistency
        bb = bias.reshape((nb, frames // nb, nw))
        c = np.arange(0, nb)
        colvar_weights = pd.DataFrame()
        for col in df.columns:
            colvar_weights[col] = bs_cols(df, col, frames, nb , c) 

        # Get the log weights for reweighting with WHAM 
        w = wham.wham(bb[c,:,:].reshape((-1, nw)), T=kBT)
        print("wham shape : ", np.shape(w["logW"]), type(w["logW"]))
        print(w["logW"])
        
        # Write the logweights to the pandas table for reweighting
        colvar_weights["logweights"] = w["logW"]

        # Use plumed to make reweighted histograms
        make_ith_histogram(colvar_weights, bs_path, "fulldata")

    # Load in FES table and remove NaN
    fes = plumed.read_as_pandas(fes_dat)
    fes = fes.replace([np.inf, -np.inf], np.nan).dropna()

    return fes

def bs_cols(df, col, frames, nb, c):
    """Reshape column arrays for bootstrapping.

    Parameters
    ----------
    df : pd.DataFrame
        Table of data from a COLVAR file.
    col : str
        Column name.
    frames : int
        Number of frames in the concatenated traj
    nb : int
        Number of bootstraps
    c : np.array
        Array for extracting columns for the bootstrap 

    Returns
    .......
    bs_flat : np.1darray
        Data is re-aranged into blocks according to "c" for 
        bootstrapping.

    """
    # Get the columns data and reshape into blocks
    arr = np.array(df[col].iloc[1:])
    arr = arr.reshape((nb, frames // nb))

    # Select blocks and flatten back into a 1d array with the 
    # original array length
    bs_flat = arr[c,:].flatten()

    return bs_flat

if __name__ == '__main__':
    main(sys.argv)