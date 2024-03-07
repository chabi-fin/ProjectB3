import sys
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import pandas as pd
import argparse
import MDAnalysis as mda
from tools import utils, traj_funcs
sys.path.insert(0, "/home/lf1071fu/project_b3/ProjectB3")
import config.settings as cf
from tools import utils, traj_funcs

def main(argv):

    try:
        parser = argparse.ArgumentParser()

        parser.add_argument("-s", "--state",
                            action = "store",
                            dest = "state",
                            default = "apo",
                            help = ("Chose the type of simulation i.e. 'unbiased'",
                                "or 'mutation'."))
        
        args = parser.parse_args()

    except argparse.ArgumentError:
        print("Command line arguments are ill-defined, please check the arguments")
        raise

    # Assign group selection from argparse
    state = args.state
    
    # Set up some path variables for state 'unbiased' or 'mutation'
    if state == "unbiased":
        fig_path = f"{ cf.figure_head }/unbiased_sims/combo_{ state }"
        state_paths = {
            "apo-open" : f"{ cf.data_head }/unbiased_sims/apo_open/analysis", 
            "apo-closed" : f"{ cf.data_head }/unbiased_sims/apo_closed/analysis",
            "holo-open" : f"{ cf.data_head }/unbiased_sims/holo_open/analysis",
            "holo-closed" : f"{ cf.data_head }/unbiased_sims/holo_closed/analysis"}
    elif state == "mutation":
        fig_path = f"{ cf.figure_head }/mutation/combo_{ state }"
        state_paths = {
            "K57G" : f"{ cf.data_head }/unbiased_sims/mutation/K57G/analysis", 
            "E200G" : f"{ cf.data_head }/unbiased_sims/mutation/E200G/analysis",
            "double-mut" : f"{ cf.data_head }/unbiased_sims/mutation/double_mut/analysis"}
    utils.create_path(fig_path)
    print(fig_path)

    states = [State(name, path) for name, path in state_paths.items()]

    # A list of tuples for selection strings of the contacts of interest
    from config.settings import selections

    # Add data into the State objects
    for state in states:

        # Load in np.array data produced by BasicMD.py 
        load_arrs(state)

        # Get arrays for the contacts of interest
        get_dist_arrs(state, selections)

    # Make plots for simulation analysis
    # stride = 50
    # plot_rmsf(states, fig_path)
    # plot_rmsd_time(states, fig_path, stride=stride)
    # plot_rgyr(states, fig_path, stride=stride)

    # plot_salt_bridges(states, fig_path, stride=stride)
    # plot_hbonds(states, fig_path, stride=stride)

    styles = {"apo-open" : ("#2E7D32", "solid", "X"), 
          "apo-closed" : ("#1976D2", "dashed", "X"),
          "holo-open" : ("#FF6F00", "solid", "o"), 
          "holo-closed" : ("#8E24AA", "dashed", "o"),
          "K57G" : ("#00897B", "solid", "o"),
          "E200G" : ("#FF6F00", "solid", "o"),
          "double-mut" : ("#5E35B1", "solid", "o")}
     
    # Make histograms of the contact distances
    for contact in selections.keys():
        plot_hist(states, contact, styles, fig_path)

    return None

class State:
    def __init__(self, name, path):
        self.name = name
        self.path = path
        self.arrays = {}
        self.data_path = f"{ os.path.dirname(path) }/nobackup"
        if "holo" in name:
            self.ligand = "holo"
        else:
            self.ligand = "apo"
        if "open" in name:
            self.conform = "open"
        else:
            self.conform = "closed"

    def add_array(self, name, array):
        self.arrays[name] = array

    def get_array(self, name):
        return self.arrays[name]

    def remove_array(self, name):
        del self.arrays[name]

def get_dist_arrs(state, selections):
    """Get distance arrays for contacts of interest.

    Parameters
    ----------
    s : State
        Each simulation state gets a state object in which the np
        arrays are stored.
    
    Returns
    -------
    None.

    """
    # Load in some reference structures
    open_state = mda.Universe(f"{ cf.struct_head }/open_ref_state.pdb")
    closed_state = mda.Universe(f"{ cf.struct_head }/closed_ref_state.pdb")
    ref_state = mda.Universe(f"{ cf.struct_head }/alignment_struct.pdb")

    # Indicies of the inflexible residues
    core_res, core = traj_funcs.get_core_res()

    get_contacts(state, selections)

    return None

def load_arrs(state):
    """Add pre-computed arrays to State object.

    Parameters
    ----------
    s : State
        Each simulation state gets a state object in which the np
        arrays are stored. 

    Returns
    -------
    None. 

    """
    analysis_path = state.path

    # Read calculated outputs from numpy arrays
    if "open" in state.path:
        np_files = { "RMSD" : f"{ analysis_path }/rmsd_open.npy"}
    else: 
        np_files = { "RMSD" : f"{ analysis_path }/rmsd_closed.npy"}
    np_files.update({"RMSF" : f"{ analysis_path }/rmsf.npy", 
                        "calphas" : f"{ analysis_path }/calphas.npy",
                        "rad_gyr" : f"{ analysis_path }/rad_gyration.npy", 
                        "salt" : f"{ analysis_path }/salt_dist.npy", 
                        "timeser" : f"{ analysis_path }/timeseries.npy",
                        "hbonds" : f"{ analysis_path }/hpairs.npy"
                    })

    # Load numpy arrays into the State object
    if all(list(map(lambda x : os.path.exists(x), np_files.values()))):

        for key, file in np_files.items(): 
            state.add_array(key, np.load(file, allow_pickle=True))

    else: 

        print("Missing Numpy files! Run BasicMD.py first.")
        exit(1)

def get_contacts(state, selections):
    """Add timeseries of critical contacts to the State object.

    Parameters
    ----------
    s : State
        Each simulation state gets a state object in which the np
        arrays are stored. 
    selections : ((str) tuple) dict
        Selection strings for critical contacts are given as tuples.    
    """
    data_path = state.data_path

    if "holo" in state.name:
        topol = f"{ data_path }/topol_Pro_Lig.top"
    else:
        topol = f"{ data_path }/topol_protein.top"
    xtc = f"{ data_path }/fitted_traj_100.xtc"

    u = mda.Universe(topol, xtc, topology_format='ITP')
    u.transfer_to_memory()
    u = traj_funcs.do_alignment(u)

    from MDAnalysis.analysis.distances import distance_array

    for key, select in selections.items():

        # Skip over inapplicable selections
        if ("IP6" in key) & ("holo" not in state.name):
            continue
        if ("K57" in key) & ("K57G" in state.name):
            continue
        if ("K57" in key) & ("double-mut" in state.name):
            continue
        if ("E200" in key) & ("E200G" in state.name):
            continue   
        if ("E200" in key) & ("double-mut" in state.name):
            continue

        # Define the distance using the tuple of selection strings
        sel_a = u.select_atoms(select[0])
        sel_b = u.select_atoms(select[1])
    
        # Iterate over trajectory framse to get the time series
        distances = np.zeros(u.trajectory.n_frames)
        for ts in u.trajectory:
            d = distance_array(sel_a.positions, sel_b.positions)
            # Use the smallest pair distance
            distances[ts.frame] = np.min(d)
        
        # Update data in the State object
        state.add_array(key, distances)
    
    # Also store array of the time series
    time_ser = np.zeros(u.trajectory.n_frames)
    for ts in u.trajectory:
        time_ser[ts.frame] = ts.time
    state.add_array("timeseries", time_ser)

    return None

def plot_rmsf(states, path):
    """Makes an RMSF plot.

    Parameters
    ----------
    states : (State object) list
        The list of State objects contains the relevant numpy arrays for each state, 
        accessed via the "get_array()" module.
    path : str
        Path to the figure storage directory.

    Returns
    -------
    None.

    """
    fig, ax = plt.subplots(constrained_layout=True, figsize=(12,4))

    # resids = list(map(lambda x : x + 544, calphas))
    colors = {"open" : "#EAAFCC", "closed" : "#A1DEA1"}
    linestyles = {"holo" : "-", "apo" : "--"}

    for i, c in enumerate(states):

        calphas = c.get_array("calphas")
        rmsf = c.get_array("RMSF")
        ax.plot(calphas, rmsf, lw=3, color=colors[c.conform], label=f"{ c.ligand }-{ c.conform }",
                alpha=0.8, dash_capstyle='round', ls=linestyles[c.ligand], path_effects=[pe.Stroke(linewidth=5, facecolor="k",
                foreground='#595959'), pe.Normal()])

    # Plot settings
    ax.tick_params(axis='y', labelsize=18, direction='in', width=2, \
                    length=5, pad=10)
    ax.tick_params(axis='x', labelsize=18, direction='in', width=2, \
                    length=5, pad=10)
    for i in ["top","bottom","left","right"]:
        ax.spines[i].set_linewidth(2)
    ax.grid(True)
    ax.set_xlabel(r"Residue number", labelpad=5, fontsize=24)
    ax.set_ylabel(r"RMSF ($\AA$)", labelpad=5, fontsize=24)
    bottom, top = ax.get_ylim()
    ax.vlines([195,217.5], bottom, top, linestyles="dashed", lw=3,
              colors="#FF1990")
    ax.vlines([219.5,231], bottom, top, linestyles="dashed", lw=3,
              colors="#dba61f")
    ax.set_ylim(-1,6)
    plt.legend(fontsize=20, loc=2)

    utils.save_figure(fig, f"{ path }/rmsf.png")
    plt.close()

    return None

def plot_rmsd_time(states, path, stride=1):
    """Makes a timeseries RMSD plot, against the respective reference structure.

    Parameters
    ----------
    states : (State object) list
        The list of State objects contains the relevant numpy arrays for each state, 
        accessed via the "get_array()" module.
    path : str
        Path to the figure storage directory.
    stride : int
        Set the stride for the data, increasing if it is too much data in the plot

    Returns
    -------
    None. 

    """
    fig, ax = plt.subplots(constrained_layout=True, figsize=(12,4))

    colors = {"open" : "#EAAFCC", "closed" : "#A1DEA1"}
    linestyles = {"holo" : "-", "apo" : "--"}

    for i, c in enumerate(states):

        rmsd = c.get_array("RMSD")
        time = c.get_array("timeser")

        plt.plot(time[::stride], rmsd[::stride,3], lw=3, color=colors[c.conform], 
                label=f"{ c.ligand }-{ c.conform }", alpha=0.8, ls=linestyles[c.ligand],
                path_effects=[pe.Stroke(linewidth=5, foreground='#595959'), pe.Normal()])

    # Plot settings
    ax.tick_params(axis='y', labelsize=18, direction='in', width=2, \
                    length=5, pad=10)
    ax.tick_params(axis='x', labelsize=18, direction='in', width=2, \
                    length=5, pad=10)

    x_labels = list(map(lambda x : str(x/1e6).split(".")[0], ax.get_xticks()))
    ax.set_xticklabels(x_labels)
    for i in ["top","bottom","left","right"]:
        ax.spines[i].set_linewidth(2)
    ax.grid(True)
    ax.set_xlabel(r"Time ($\mu s$)", labelpad=5, fontsize=24)
    ax.set_ylabel(r"RMSD ($\AA$)", labelpad=5, fontsize=24)
    _, ymax = ax.get_ylim()
    ax.set_ylim(0,ymax)
    plt.legend(fontsize=20)

    utils.save_figure(fig, f"{ path }/rmsd_time.png")
    plt.close()

    return None

def plot_salt_bridges(states, path, stride=1):
    """Makes a timeseries plot for the key salt bridge K57--E200.

    Parameters
    ----------
    states : (State object) list
        The list of State objects contains the relevant numpy arrays for each state, 
        accessed via the "get_array()" module.
    path : str
        Path to the figure storage directory.

    Returns
    -------
    None.

    """

    fig, ax = plt.subplots(constrained_layout=True, figsize=(12,6))

    colors = {"open" : "#EAAFCC", "closed" : "#A1DEA1"}
    linestyles = {"holo" : "-", "apo" : "--"}
    filled_marker_style = dict(marker='o', markersize=10, linestyle="-", lw=3,
                               markeredgecolor='#595959')

    for i, c in enumerate(states):
        
        sc = c.get_array("salt")
        time = c.get_array("timeser")

        plt.plot(time[::stride], sc[::stride,3], lw=3, color=colors[c.conform], 
                label=f"{ c.ligand }-{ c.conform }", alpha=0.8, ls=linestyles[c.ligand],
                path_effects=[pe.Stroke(linewidth=5, foreground='#595959'), pe.Normal()])

    # Plot settings
    ax.tick_params(axis='y', labelsize=18, direction='in', width=2, \
                    length=5, pad=10)
    ax.tick_params(axis='x', labelsize=18, direction='in', width=2, \
                    length=5, pad=10)
    x_labels = list(map(lambda x : str(x/1e6).split(".")[0], ax.get_xticks()))
    ax.set_xticklabels(x_labels)
    ax.set_xlabel(r"Time ($\mu s$)", labelpad=5, fontsize=24)
    for i in ["top","bottom","left","right"]:
        ax.spines[i].set_linewidth(2)
    ax.grid(True)
    ax.set_ylabel(r"Distance K57--E200 $(\AA)$", labelpad=5, fontsize=24)
    plt.legend(fontsize=20)

    utils.save_figure(fig, f"{ path }/salt_bridge_K57-E200.png")
    plt.close()

    return None

def plot_hbonds(states, path, stride=1):
    """Makes a timeseries plot for the key hbond N53--E200.

    Parameters
    ----------
    states : (State object) list
        The list of State objects contains the relevant numpy arrays for each state, 
        accessed via the "get_array()" module.
    path : str
        Path to the figure storage directory.

    Returns
    -------
    None.

    """

    fig, ax = plt.subplots(constrained_layout=True, figsize=(12,6))

    colors = {"open" : "#EAAFCC", "closed" : "#A1DEA1"}
    linestyles = {"holo" : "-", "apo" : "--"}
    filled_marker_style = dict(marker='o', markersize=10, linestyle="-", lw=3,
                               markeredgecolor='#595959')

    for i, c in enumerate(states):
        
        hbonds = c.get_array("hbonds")
        time = c.get_array("timeser")

        plt.plot(time[::stride], hbonds[::stride,3], lw=3, color=colors[c.conform], 
                label=f"{ c.ligand }-{ c.conform }", alpha=0.8, ls=linestyles[c.ligand],
                path_effects=[pe.Stroke(linewidth=5, foreground='#595959'), pe.Normal()])

    # Plot settings
    ax.tick_params(axis='y', labelsize=18, direction='in', width=2, \
                    length=5, pad=10)
    ax.tick_params(axis='x', labelsize=18, direction='in', width=2, \
                    length=5, pad=10)
    x_labels = list(map(lambda x : str(x/1e6).split(".")[0], ax.get_xticks()))
    ax.set_xticklabels(x_labels)
    ax.set_xlabel(r"Time ($\mu s$)", labelpad=5, fontsize=24)
    for i in ["top","bottom","left","right"]:
        ax.spines[i].set_linewidth(2)
    ax.grid(True)
    ax.set_ylabel(r"Distance N53--E200 $(\AA)$", labelpad=5, fontsize=24)
    plt.legend(fontsize=20)

    utils.save_figure(fig, f"{ path }/hbond_N53-E200.png")
    plt.close()

    return None

def plot_rgyr(states, path, stride=1):
    """Makes a Radius of Gyration plot.

    Parameters
    ----------
    states : (State) list
        The list of State objects contains the relevant numpy arrays for each state, 
        accessed via the "get_array()" module.
    path : str
        Path to the figure storage directory.

    Returns
    -------
    None. 

    """

    fig, ax = plt.subplots(constrained_layout=True, figsize=(12,4))

    colors = {"open" : "#EAAFCC", "closed" : "#A1DEA1"}
    linestyles = {"holo" : "-", "apo" : "--"}

    for i, c in enumerate(states):

        r_gyr = c.get_array("rad_gyr")
        time = c.get_array("timeser")
        plt.plot(time[::stride], r_gyr[::stride], lw=3, color=colors[c.conform], 
                label=f"{ c.ligand }-{ c.conform }", alpha=0.8, ls=linestyles[c.ligand],
                path_effects=[pe.Stroke(linewidth=5, foreground='#595959'), pe.Normal()])

    # Plot settings
    ax.tick_params(axis='y', labelsize=18, direction='in', width=2, \
                    length=5, pad=10)
    ax.tick_params(axis='x', labelsize=18, direction='in', width=2, \
                    length=5, pad=10)
    x_labels = list(map(lambda x : str(x/1e6).split(".")[0], ax.get_xticks()))
    ax.set_xticklabels(x_labels)

    for i in ["top","bottom","left","right"]:
        ax.spines[i].set_linewidth(2)
    ax.grid(True)
    ax.set_xlabel(r"Time ($\mu s$)", labelpad=5, fontsize=24)
    ax.set_ylabel(r"$R_G$ ($\AA$)", labelpad=20, fontsize=24)
    plt.legend(fontsize=20)
    y_min, ymax = ax.get_ylim()
    ax.set_ylim(y_min-0.2,ymax+0.5)

    utils.save_figure(fig, f"{ path }/rad_gyration.png")
    plt.close()

    return None

def plot_hist(states, contact, styles, fig_path, bins=15, 
    fs=24, **kwargs):
    """Makes a histogram plot of the contact all the State objects.

    Parameters
    ----------
    states : (State) list
        The list of State objects contains the relevant numpy arrays
        for each state, accessed via the "get_array()" module.
    contact : str
        Name of the contact which is visualized by the histogram. The
        string is used to access the array from the relevant
        dictionaries. 

    Returns
    -------
    None.

    """
    fig, ax = plt.subplots(figsize=(8,4))
    
    for state in states:

        key = state.name

        # Skip over inapplicable selections
        if ("IP6" in contact) & ("holo" not in key):
            continue
        if ("K57" in contact) & ("K57G" in key):
            continue
        if ("K57" in contact) & ("double" in key):
            continue
        if ("E200" in contact) & ("E200G" in key):
            continue   
        if ("E200" in contact) & ("double" in key):
            continue

        print(state.name)

        dist = state.get_array(contact)
        ax.hist(dist, bins=bins, density=True, color=styles[key][0], 
                ls=styles[key][1], lw=6, histtype='step', label=key)
    
    ax.set_xlabel(f"Distance { contact } " + r"($\AA$)", fontsize=fs)
    ax.set_ylabel("Relative frequency", fontsize=fs)
    if "xlim" in kwargs:
        plt.xlim(kwargs["xlim"])
    plt.legend(fontsize=fs)

    utils.save_figure(fig, f"{ fig_path }/{ contact }.png")

    plt.close()

    return None

if __name__ == '__main__':
    main(sys.argv)
