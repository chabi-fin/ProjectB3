import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import plumed
import wham
import MDAnalysis as mda
from MDAnalysis.analysis import align
from MDAnalysis.analysis.distances import distance_array
import subprocess    
import argparse
import pandas as pd
sys.path.insert(0, "/home/lf1071fu/project_b3/ProjectB3")
import config.settings as config
from tools import utils, traj_funcs
from VectorCoordCombo import get_ref_vecs, get_vec_dataframe

def main(argv):

    try:
        parser = argparse.ArgumentParser()

        parser.add_argument("-p", "--path",
                            action = "store",
                            dest = "path",
                            default = ("umbrella/salt_bridge/E200G/"
                                "nobackup/plumed_driver"),
                            help = ("Set relative path to the data "
                                "directory.")) 
        parser.add_argument("-f", "--figpath",
                            action = "store",
                            dest = "fig_path",
                            default = ("umbrella/salt_bridge"),
                            help = ("Set a relative path destination for"
                                " the figure."))
        parser.add_argument("-w", "--num_windows",
                            action = "store",
                            dest = "num_windows",
                            default = 30,
                            help = ("Number of windows used in the "))
        parser.add_argument("-r", "--recalc",
                            action = "store_true",
                            dest = "recalc",
                            default = False,
                            help = ("Chose whether the reweighting "
                                "should be recomputed."))
        parser.add_argument("-m", "--mutant",
                            action = "store",
                            dest = "mutant",
                            default = "E200G",
                            help = ("Select a mutation experiment, e.g."
                                " E200G."))
        args = parser.parse_args()

    except argparse.ArgumentError:
        print("Command line arguments are ill-defined, please check the "
            "arguments.")
        raise

    # Set key path variables
    home = f"{ config.data_head  }/{ args.path }"
    fig_path = f"{ config.figure_head }/{ args.fig_path }"
    struct_path = config.struct_head

    # Set other variables
    recalc = args.recalc
    kBT = 310 * 8.314462618 * 0.001 # use kJ/mol here
    nw = int(args.num_windows) # number of windows 
    mutant = args.mutant

    # Helpful objects for structure alignment and reference structures
    core_res, core = traj_funcs.get_core_res()
    ref_state = mda.Universe(f"{ struct_path }/ref_all_atoms.pdb", 
                             length_unit="nm")
    open_ref = mda.Universe(f"{ config.struct_head }/open_ref_state.pdb")
    closed_ref = mda.Universe(f"{ config.struct_head }/closed_ref_state.pdb")

    # Get the reference salt bridge distances of the alpha carbons for 
    # open and closed conformations
    res_200_sel = ("resid 200 and name CA")
    res_57_sel = ("resid 57 and name CA")
    open_dist = distance_array(open_ref.select_atoms(res_57_sel).positions, 
                    open_ref.select_atoms(res_200_sel).positions)[0][0]
    closed_dist = distance_array(closed_ref.select_atoms(res_57_sel).positions, 
                    closed_ref.select_atoms(res_200_sel).positions)[0][0]

    fes_dat = f"{ home }/fes_catr.dat"

    if not os.path.exists(fes_dat) or recalc:

        # Get bias data from COLVAR outputs
        bias, frames = get_bias_data(home, nw, recalc=recalc)

        # Get collective variable data from any COLVAR file
        columns = ["time", "sb", "bias", "force"]
        df = pd.read_csv(f"{ home }/COLVAR_1.dat", delim_whitespace=True,
                            comment='#', names=columns)
        df_reshape = pd.DataFrame()
        for col in df.columns:
            df_reshape[col] = df[col].values[1:]

        # Reweight from biases using WHAM
        w = wham.wham(bias.reshape((-1, nw)), T=kBT)
        df_reshape["logweights"] = w["logW"]

        # Make histograms and convert to FES using plumed
        make_histogram(df_reshape, kBT, home)

    # Map mutant experiments to labels
    rxn_coord_labs = {"native" :  r"K57 C$_\alpha-$ E200 C$_\alpha$",
        "E200G" : r"K57 C$_\alpha-$ G200 C$_\alpha$",
        "K57G" :  r"G57 C$_\alpha-$ E200 C$_\alpha$",
        "double_mut" :  r"G57 C$_\alpha-$ G200 C$_\alpha$"}

    # Check window overlap -- gets a dataframe of the beta vecs and then
    # plots data against the salt bridge distance for each window
    df = extra_cvs(home, struct_path, rxn_coord_labs, recalc=recalc)
    plot_window_overlap(home, df, mutant, rxn_coord_labs[mutant], 
        fig_path, "open")
    plot_window_overlap(home, df, mutant, rxn_coord_labs[mutant], 
        fig_path, "closed")

    # Plot FES
    plot_1dfes(home, mutant, rxn_coord_labs[mutant], fig_path)

    plot_multi_fes(rxn_coord_labs, fig_path)

def extra_cvs(home, struct_path, rxn_coord_labs, recalc=False):
    """Makes a DataFrame for the beta-vec values.

    Parameters 
    ----------
    home : str
        Path to the working directory with trajectory data.
    struct_path : str
        Path for structures, same as the configuration file.

    Returns
    -------
    df : DataFrame
        Table of the beta-vec values during the simulation.
    """
    # Get the reference beta vectors
    r1, r2 = 206, 215
    ref_state, vec_open, vec_closed = get_ref_vecs(struct_path, r1, r2)

    # Get dataframe of the betavecs
    trajs, tops = {}, {}
    for key in rxn_coord_labs.keys():
        tops[key] = f"{ home }/topol_protein.top"
        trajs[key] = f"{ home }/fitted_10.xtc"
    utils.process_topol(home, "topol_protein.top")
    df = get_vec_dataframe(trajs, tops, f"{ home }/df_beta_vecs", 
            r1, r2, ref_state, vec_open, vec_closed, recalc=recalc)

    return df

def get_bias_data(home, nw, recalc=False):
    """Get the array of bias data from all windows.

    Bias data is stored in the COLVAR files produced in post-processing 
    on the full, concatenated trajectory for all the plumed input files.

    Parameters
    ----------
    home : str
        Path to the working directory.
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
        columns = ["time", "sb", "bias", "force"]
        for i in range(nw):
            df = pd.read_csv(f"{ home }/COLVAR_" + str(i+1)+".dat",
                       delim_whitespace=True, comment='#', names=columns)
            if "bias_arr" not in locals():
                frames = len(df["bias"]) - 1
                print("Number of frames", frames)
                bias = np.zeros((frames, nw))
            # Reshape the bias array
            bias[:,i] = df["bias"].iloc[1:]

        utils.save_array(bias_file, bias)

    else:

        bias = np.load(bias_file, allow_pickle=True)
        frames = len(bias[:,0])

    return bias, frames

def make_histogram(df, kBT, home):
    """Use plumed HISTOGRAM function to obtain reweighted histograms.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the collective variables and logweights.
    kBT : float
        KbT is needed for Boltzmann reweighting.
    home : str
        Path to directory for bootstraping data.

    Returns
    -------
    None.

    """
    print(df.head())
    plumed.write_pandas(df, f"{ home }/colvar_weights.dat")

    with open(f"{ home }/colvar_histograms.dat","w") as f:
        print(f"""
    # vim:ft=plumed
    sb: READ FILE={ home }/colvar_weights.dat VALUES=sb IGNORE_TIME
    lw: READ FILE={ home }/colvar_weights.dat VALUES=logweights IGNORE_TIME

    hhr1d_sb: HISTOGRAM ARG=sb GRID_MIN=1.1 GRID_MAX=2.2 GRID_BIN=100 BANDWIDTH=0.05 LOGWEIGHTS=lw
    ffr1d_sb: CONVERT_TO_FES GRID=hhr1d_sb
    DUMPGRID GRID=ffr1d_sb FILE={ home }/fes_catr.dat
    """, file=f)

    subprocess.run((f"plumed driver --noatoms --plumed { home }/colvar" 
        f"_histograms.dat --kt { kBT }"), shell=True)

    return None

def plot_1dfes(home, mutant, rxn_coord, fig_path):
    """Makes a plot against the 1D reaction coordinate.

    Parameters
    ----------
    home : str
        Path to the working directory.
    mutant : str
        Name of the mutation experiment, e.g. E200G.
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
    fes = plumed.read_as_pandas(f"{ home }/fes_catr.dat")
    fes = fes.replace([np.inf, -np.inf], np.nan).dropna()

    # Select reaction coordinate and lable axes accordingly
    ax.plot(fes["sb"], fes["ffr1d_sb"], label=f"{ mutant } FES")
    ax.set_xlabel(f"{ rxn_coord } (nm)", labelpad=5, fontsize=24)

    ax.set_ylabel(f"F({ rxn_coord }) (kJ / mol)", labelpad=5, 
                  fontsize=24)
    
    # Plot settings
    plt.legend(fontsize=18)
    # _, xmax = ax.get_xlim()
    # _, ymax = ax.get_ylim()
    # ax.set_xlim(0,xmax)
    ax.set_ylim(17,20)

    # Save figure and close figure object
    utils.save_figure(fig, f"{ fig_path }/fes_{ mutant }.png")
    plt.close()

    return None

def plot_window_overlap(home, df, mutant, rxn_coord, fig_path, vec_state):
    """Makes a plot to verify window overlap.

    Parameters
    ----------
    home : str
        Path to the working directory.
    df : DataFrame
        Table of the beta-vec values during the simulation.
    mutant : str
        Name of the mutation experiment, e.g. E200G.
    rxn_coord : str
        Name of the reaction coordinate should match the column name.
    fig_path : str
        Path to figure directory for particular fes. 
    vec_state : str
        Choose a secondary coordinate to check overlap, e.g. "open" or
        "closed".
    
    Returns
    -------
    None.

    """
    columns = ["time", "sb", "bias", "force"]
    colvar = pd.read_csv(f"{ home }/COLVAR_1.dat", delim_whitespace=True,
                     comment='#', names=columns)
    df_reshape = pd.DataFrame()
    for col in colvar.columns:
        df_reshape[col] = colvar[col].values[::10]

    print(df_reshape.head())
    print(df.head())

    merge = pd.concat([df_reshape, df], axis=1)

    fig, ax = plt.subplots(constrained_layout=True, figsize=(10,8))    

    colors = [
    "#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", 
    "#b15928", "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", 
    "#cab2d6", "#fdae61", "#207dff", "#4bff33", "#ff0000", 
    "#ffa500", "#9400d3", "#800000", "#00ffff", "#00ff00", 
    "#f0e68c", "#ff1493", "#9932cc", "#8b0000", "#ff6347", 
    "#8a2be2", "#32cd32", "#dda0dd", "#00ced1", "#ff4500"
    ]

    # Add vertical lines at the restraint points
    # ax.vlines(x=x, ymin=[y_min]*len(x), ymax=ave_y, color='#949494', 
    #            ls=':', lw=3, clip_on=False)

    # Set labels
    ax.set_xlabel(f"Salt bridge { mutant } : { rxn_coord }")
    if vec_state == "open":
        for i in range(1,31):
            ax.scatter(merge["sb"][1001*i:(1001*(i+1))], 
                merge["dot-open"][1001*i:(1001*(i+1))], marker="o", 
                label=f"window {i}", s=50, color=colors[i-1])
        ax.set_ylabel(r"$\vec{\upsilon} \cdot \vec{\upsilon}_{open}$ (nm$^2$)")
    elif vec_state == "closed":
        for i in range(1,31):
            ax.scatter(merge["sb"][1001*i:(1001*(i+1))], 
                merge["dot-closed"][1001*i:(1001*(i+1))], marker="o", 
                label=f"window {i}", s=50, color=colors[i-1])
        ax.set_ylabel(r"$\vec{\upsilon} \cdot \vec{\upsilon}_{closed}$ (nm$^2$)")
    plt.legend(ncol=6, fontsize=10)

    plt.savefig(f"{ fig_path }/Windows_{ mutant }_{ vec_state }.png", dpi=300)
    plt.close()

def plot_multi_fes(rxn_coord_labs, fig_path):
    """Makes a plot against the 1D reaction coordinate.

    Parameters
    ----------
    rxn_coord_labs : dict
        The reaction coordinates for each experiment.
    fig_path : str
        Path to figure directory for particular fes. 

    Returns
    -------
    None.

    """
    # Initialize the plot
    fig, ax = plt.subplots(constrained_layout=True, figsize=(12,8))

    # Define paths for the data and load in fes
    path_head = f"{ config.data_head }/umbrella/salt_bridge"
    fes = {}
    for m in rxn_coord_labs.keys():
        p = f"{ path_head }/{ m }/nobackup/plumed_driver"

        # Load table containing discretized free energy surface data.   
        f = plumed.read_as_pandas(f"{ p }/fes_catr.dat")
        fes[m] = f.replace([np.inf, -np.inf], np.nan).dropna()

    # Select reaction coordinate and label axes accordingly
    labels = {"E200G" : "E200G", "K57G" : "K57G", "native" : "native",
        "double_mut" : "double mutant"}
    for k, fes in fes.items():
        ax.plot(fes["sb"], fes["ffr1d_sb"], label=f"{ labels[k] } FES")
     
    # Plot settings
    ax.set_xlabel(r"57 C$_\alpha-$ 200 C$_\alpha$ (nm)", labelpad=5, fontsize=24)
    ax.set_ylabel(r"F(57 C$_\alpha-$ 200 C$_\alpha$) (kJ / mol)", labelpad=5, 
                  fontsize=24)
    plt.legend(fontsize=18)
    # _, xmax = ax.get_xlim()
    # _, ymax = ax.get_ylim()
    # ax.set_xlim(0,xmax)
    ax.set_ylim(16,18.5)

    # Save figure and close figure object
    utils.save_figure(fig, f"{ fig_path }/fes_combo.png")
    plt.show()
    plt.close()

    return None

if __name__ == '__main__':
    main(sys.argv)