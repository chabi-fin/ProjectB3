import sys
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import pandas as pd
import argparse
from tools import utils, traj_funcs

def main(argv):

    try:
        parser = argparse.ArgumentParser()

        parser.add_argument("-p", "--path",
                            action = "store",
                            nargs='+',
                            dest = "path",
                            default = None,
                            help = """Set path to the data directory.""")
        parser.add_argument("-s", "--state",
                            action = "store",
                            dest = "state",
                            default = "apo",
                            help = """Chose the type of simulation i.e. "holo", "apo", "all" or in the case of mutational, "K57G" etc.""")
        args = parser.parse_args()

    except argparse.ArgumentError:
        print("Command line arguments are ill-defined, please check the arguments")
        raise

    # Assign group selection from argparse
    state_paths = args.path
    state = args.state

    path_head = "/home/lf1071fu/project_b3"
    fig_path = f"/home/lf1071fu/project_b3/figures/unbiased_sims/combo_{ state }"

    if not state_paths:
        state_paths = [f"{ path_head }/simulate/unbiased_sims/apo_open/analysis", 
                     f"{ path_head }/simulate/unbiased_sims/apo_closed/analysis",
                     f"{ path_head }/simulate/unbiased_sims/holo_open/analysis",
                     f"{ path_head }/simulate/unbiased_sims/holo_closed/analysis"]

    states = [State(s) for s in state_paths]

    # Load in np.array data into Conform objects
    for s in states:

        data_path = s.path

        # Read calculated outputs from numpy arrays
        if "open" in s.path:
            np_files = { "RMSD" : f"{ data_path }/rmsd_open.npy"}
        else: 
            np_files = { "RMSD" : f"{ data_path }/rmsd_closed.npy"}
        np_files.update({"RMSF" : f"{ data_path }/rmsf.npy", 
                         "calphas" : f"{ data_path }/calphas.npy",
                         "rad_gyr" : f"{ data_path }/rad_gyration.npy", 
                         "salt" : f"{ data_path }/salt_dist.npy", 
                         "timeser" : f"{ data_path }/timeseries.npy",
                         "hbonds" : f"{ data_path }/hpairs.npy"
                        })

        # Load numpy arrays into the Conform objects
        if all(list(map(lambda x : os.path.exists(x), np_files.values()))):

            for key, file in np_files.items(): 
                s.add_array(key, np.load(file, allow_pickle=True))

        else: 

            print("Missing Numpy files! Run BasicMD.py first.")
            exit(1)

    # Make plots for simulation analysis
    print(fig_path)
    stride = 50
    plot_rmsf(states, fig_path)
    plot_rmsd_time(states, fig_path, stride=stride)
    plot_rgyr(states, fig_path, stride=stride)

    plot_salt_bridges(states, fig_path, stride=stride)
    plot_hbonds(states, fig_path, stride=stride)

    return None

class State:
    def __init__(self, path):
        self.path = path
        self.arrays = {}
        if "holo" in path:
            self.ligand = "holo"
        else:
            self.ligand = "apo"
        if "open" in path:
            self.conform = "open"
        else:
            self.conform = "closed"

    def add_array(self, name, array):
        self.arrays[name] = array

    def get_array(self, name):
        return self.arrays[name]

    def remove_array(self, name):
        del self.arrays[name]

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

if __name__ == '__main__':
    main(sys.argv)
