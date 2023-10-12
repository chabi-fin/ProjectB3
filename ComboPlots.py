import sys
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import pandas as pd
import argparse

def main(argv):

    try:
        parser = argparse.ArgumentParser()

        parser.add_argument("-s", "--state",
                            action = "store",
                            dest = "state",
                            default = "apo",
                            help = """Chose the type of simulation i.e. "holo" or "apo" or in the case of mutational, "K57G" etc.""")
        args = parser.parse_args()

    except argparse.ArgumentError:
        print("Command line arguments are ill-defined, please check the arguments")
        raise

    # Assign group selection from argparse
    state = args.state

    path_head = "/home/lf1071fu/project_b3"
    fig_path = f"/home/lf1071fu/project_b3/figures/combo-{ state }"

    opened = Conform(f"{ path_head }/simulate/{ state }_state/open/analysis")
    closed = Conform(f"{ path_head }/simulate/{ state }_state/closed/analysis")

    # Load in np.array data into Conform objects
    for c in [opened, closed]:

        data_path = c.path

        # Read calculated outputs from numpy arrays
        if c == opened:
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
                c.add_array(key, np.load(file, allow_pickle=True))

        else: 

            print("Missing Numpy files! Run BasicMD.py first.")
            exit(1)

    # Make plots for simulation analysis
    plot_rmsf(opened, closed, fig_path)
    plot_rmsd_time(opened, closed, fig_path)
    plot_rgyr(opened, closed, fig_path)

    plot_salt_bridges(opened, closed, fig_path)
    plot_hbonds(opened, closed, fig_path)

    return None

class Conform:
    def __init__(self, path):
        self.path = path
        self.arrays = {}

    def add_array(self, name, array):
        self.arrays[name] = array

    def get_array(self, name):
        return self.arrays[name]

    def remove_array(self, name):
        del self.arrays[name]

def plot_rmsf(opened, closed, path):
    """Makes an RMSF plot.

    Parameters
    ----------
    opened : Conform object
        The conform object for the open state contains the relevant numpy arrays, 
        accessed via the "get_array()" module.
    closed : Conform object
        The conform object for the closed state contains the relevant numpy arrays, 
        accessed via the "get_array()" module.
    path : str
        Path to the figure storage directory.

    Returns
    -------
    None.

    """
    fig, ax = plt.subplots(constrained_layout=True, figsize=(12,4))

    # resids = list(map(lambda x : x + 544, calphas))
    colors = ["#EAAFCC", "#A1DEA1"]
    linestyles = ["--", ":"]
    labels = ["Open conform", "Closed conform"]

    for i, c in enumerate([opened, closed]):

        calphas = c.get_array("calphas")
        rmsf = c.get_array("RMSF")
        plt.plot(calphas, rmsf, lw=3, color=colors[i], label=labels[i], alpha=0.8,
                path_effects=[pe.Stroke(linewidth=5, foreground='#595959'), pe.Normal()])

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
    ax.set_ylim(-1,10)
    plt.legend(fontsize=24)

    plt.savefig(f"{ path }/rmsf.png", dpi=300)
    plt.close()

    return None

def plot_rmsd_time(opened, closed, path, stride=1):
    """Makes a timeseries RMSD plot, against the respective reference structure.

    Parameters
    ----------
    opened : Conform object
        The conform object for the open state contains the relevant numpy arrays, 
        accessed via the "get_array()" module.
    closed : Conform object
        The conform object for the closed state contains the relevant numpy arrays, 
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

    colors = ["#EAAFCC", "#A1DEA1"]
    linestyles = ["--", ":"]
    labels = [r"Open conform", r"Closed conform"]

    for i, c in enumerate([opened, closed]):

        rmsd = c.get_array("RMSD")
        time = c.get_array("timeser")

        plt.plot(time[::stride], rmsd[::stride,3], lw=3, color=colors[i], alpha=0.8, label=labels[i],
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
    plt.legend(fontsize=24)

    plt.savefig(f"{ path }/rmsd_time.png", dpi=300)
    plt.close()

    return None

def plot_salt_bridges(opened, closed, path, stride=1):
    """Makes a timeseries plot for the key salt bridge K57--E200.

    Parameters
    ----------
    opened : Conform object
        The conform object for the open state contains the relevant numpy arrays, 
        accessed via the "get_array()" module.
    closed : Conform object
        The conform object for the closed state contains the relevant numpy arrays, 
        accessed via the "get_array()" module.
    path : str
        Path to the figure storage directory.

    Returns
    -------
    None.

    """

    fig, ax = plt.subplots(constrained_layout=True, figsize=(12,6))

    colors = ["#EAAFCC", "#A1DEA1"]
    linestyles = ["--", ":"]
    labels = [r"Open conform", r"Closed conform"]
    filled_marker_style = dict(marker='o', markersize=10, linestyle="-", lw=3,
                               markeredgecolor='#595959')

    for i, c in enumerate([opened, closed]):
        
        sc = c.get_array("salt")
        time = c.get_array("timeser")

        plt.plot(time[::stride], sc[::stride,3], lw=3, color=colors[i], label=labels[i], alpha=0.8,
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
    plt.legend(fontsize=24)

    plt.savefig(f"{ path }/salt_bridge_K57-E200.png", dpi=300)
    plt.close()

    return None

def plot_hbonds(opened, closed, path, stride=1):
    """Makes a timeseries plot for the key hbond N53--E200.

    Parameters
    ----------
    opened : Conform object
        The conform object for the open state contains the relevant numpy arrays, 
        accessed via the "get_array()" module.
    closed : Conform object
        The conform object for the closed state contains the relevant numpy arrays, 
        accessed via the "get_array()" module.
    path : str
        Path to the figure storage directory.

    Returns
    -------
    None.

    """

    fig, ax = plt.subplots(constrained_layout=True, figsize=(12,6))

    colors = ["#EAAFCC", "#A1DEA1"]
    linestyles = ["--", ":"]
    labels = [r"Open conform", r"Closed conform"]
    filled_marker_style = dict(marker='o', markersize=10, linestyle="-", lw=3,
                               markeredgecolor='#595959')

    for i, c in enumerate([opened, closed]):
        
        hbonds = c.get_array("hbonds")
        time = c.get_array("timeser")

        plt.plot(time[::stride], hbonds[::stride,3], lw=3, color=colors[i], label=labels[i], alpha=0.8,
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
    plt.legend(fontsize=24)

    plt.savefig(f"{ path }/hbond_N53-E200.png", dpi=300)
    plt.close()

    return None

def plot_rgyr(opened, closed, path, stride=1):
    """Makes a Radius of Gyration plot.

    Parameters
    ----------
    opened : Conform object
        The conform object for the open state contains the relevant numpy arrays, 
        accessed via the "get_array()" module.
    closed : Conform object
        The conform object for the closed state contains the relevant numpy arrays, 
        accessed via the "get_array()" module.
    path : str
        Path to the figure storage directory.

    Returns
    -------
    None. 

    """

    fig, ax = plt.subplots(constrained_layout=True, figsize=(12,4))

    colors = ["#EAAFCC", "#A1DEA1"]
    linestyles = ["--", ":"]
    labels = [r"Open conform", r"Closed conform"]

    for i, c in enumerate([opened, closed]):

        r_gyr = c.get_array("rad_gyr")
        time = c.get_array("timeser")
        plt.plot(time[::stride], r_gyr[::stride], lw=3, color=colors[i], label=labels[i], alpha=0.8,
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
    plt.legend(fontsize=24)
    y_min, ymax = ax.get_ylim()
    ax.set_ylim(y_min-0.2,ymax+0.5)

    plt.savefig(f"{ path }/rad_gyration.png", dpi=300)
    plt.close()

    return None

if __name__ == '__main__':
    main(sys.argv)
