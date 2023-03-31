import sys
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import pandas as pd

def main(argv):

    path_head = "/home/lf1071fu/project_b3"
    fig_path = f"/home/lf1071fu/project_b3/figures/combo"

    holo = Conform(f"{ path_head }/simulate/holo_conf/data")
    apo = Conform(f"{ path_head }/simulate/apo_conf/initial_10us")

    for c in [holo, apo]:

        data_path = c.path

        # Store calculated outputs as numpy arrays
        np_files = [f"{ data_path }/rmsd_holo.npy", f"{ data_path }/rmsd.npy",
                    f"{ data_path }/rmsf.npy", f"{ data_path }/calphas.npy",
                    f"{ data_path }/rad_gyration.npy", f"{ data_path }/salt.npy", ]
                    # f"{ data_path }/hbonds.npy",
                    #f"{ data_path }/hbond_count.npy"]

        if all(list(map(lambda x : os.path.exists(x), np_files))):

            c.add_array("R_holo", np.load(np_files[0], allow_pickle=True))
            c.add_array("R_apo", np.load(np_files[1], allow_pickle=True))
            c.add_array("RMSF", np.load(np_files[2], allow_pickle=True))
            c.add_array("calphas", np.load(np_files[3], allow_pickle=True))
            c.add_array("rad_gyr", np.load(np_files[4], allow_pickle=True))
            c.add_array("salt", np.load(np_files[5], allow_pickle=True))
            # hbonds = np.load(np_files[7], allow_pickle=True)
            # hbond_count = np.load(np_files[8], allow_pickle=True)
            # times = np.load(np_files[9], allow_pickle=True)

        else: 

            print("Missing Numpy files! Run BasicMD.py first.")
            exit(1)

    # Make plots for simulation analysis
    plot_rmsf(holo, apo, fig_path)
    plot_rmsd_time(holo, apo, fig_path)
    plot_rgyr(holo, apo, fig_path)

    plot_salt_bridges(holo, apo, fig_path)
    # plot_hbonds_count(hbond_count, times, fig_path)

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

def plot_rmsf(holo, apo, path):
    """Makes an RMSF plot.

    Parameters
    ----------
    holo : Conform object
        The conform object for the holo/open state contains the relevant numpy arrays, 
        accessed via the "get_array()" module.
    apo : Conform object
        The conform object for the apo/closed state contains the relevant numpy arrays, 
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

    for i, c in enumerate([holo, apo]):

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

    plt.savefig(f"{ path }/rmsf.pdf", dpi=300)
    plt.close()

    return None

def plot_rmsd_time(holo, apo, path):
    """Makes an RMSF plot.

    Parameters
    ----------
    holo : Conform object
        The conform object for the holo/open state contains the relevant numpy arrays, 
        accessed via the "get_array()" module.
    apo : Conform object
        The conform object for the apo/closed state contains the relevant numpy arrays, 
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
    stride = 100

    for i, c in enumerate([holo, apo]):

        if i == 0:
            rmsd = c.get_array("R_holo")
            time_lab = rmsd[::stride,0]
        else:
            rmsd = c.get_array("R_apo")
        time = rmsd[::stride,0]

        plt.plot(time, rmsd[::stride,3], lw=3, color=colors[i], alpha=0.8, label=labels[i],
                path_effects=[pe.Stroke(linewidth=5, foreground='#595959'), pe.Normal()])

    # Plot settings
    ax.tick_params(axis='y', labelsize=18, direction='in', width=2, \
                    length=5, pad=10)
    ax.tick_params(axis='x', labelsize=18, direction='in', width=2, \
                    length=5, pad=10)
    xticks = np.arange(0,len(time_lab),1000)
    ax.set_xticks(xticks)
    ax.set_xticklabels(list(map(lambda x : str(x/1000).split(".")[0], xticks)))
    for i in ["top","bottom","left","right"]:
        ax.spines[i].set_linewidth(2)
    ax.grid(True)
    ax.set_xlabel(r"Time ($\mu s$)", labelpad=5, fontsize=24)
    ax.set_ylabel(r"RMSD ($\AA$)", labelpad=5, fontsize=24)
    _, ymax = ax.get_ylim()
    ax.set_ylim(0,ymax)
    plt.legend(fontsize=24)

    plt.savefig(f"{ path }/rmsd_time.pdf", dpi=300)
    plt.close()

    return None

def plot_salt_bridges(holo, apo, path):
    """Make a plot for the fraction of salt contacts compared to the reference.

    A soft cut-off from 4 A was used to determine the fraction of salt bridges
    retained from the reference structure. This includes bridges between the
    charged residues, i.e. N in ARG and LYS with O in ASP and GLU.

    Parameters
    ----------
    holo : Conform object
        The conform object for the holo/open state contains the relevant numpy arrays, 
        accessed via the "get_array()" module.
    apo : Conform object
        The conform object for the apo/closed state contains the relevant numpy arrays, 
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
    stride = 100
    filled_marker_style = dict(marker='o', markersize=10, linestyle="-", lw=3,
                               markeredgecolor='#595959')

    for i, c in enumerate([holo, apo]):
        
        sc = c.get_array("salt")
        if i == 0:
            time_lab = sc[:,0]
        time = sc[::stride,0]

        # plt.plot(time, sc[::stride,1], color=colors[i], label=labels[i], 
        #          **filled_marker_style)
        plt.plot(time, sc[::stride,1], lw=3, color=colors[i], label=labels[i], alpha=0.8,
                path_effects=[pe.Stroke(linewidth=5, foreground='#595959'), pe.Normal()])

        ave_contacts = np.mean(sc[:, 1])
        print(f"average contacts = { ave_contacts }")

    # Plot settings
    ax.tick_params(axis='y', labelsize=18, direction='in', width=2, \
                    length=5, pad=10)
    ax.tick_params(axis='x', labelsize=18, direction='in', width=2, \
                    length=5, pad=10)
    xticks = np.arange(0,len(time_lab),1000)
    ax.set_xticks(xticks)
    ax.set_xticklabels(list(map(lambda x : str(x/1000).split(".")[0], xticks)))
    ax.set_xlabel(r"Time ($\mu s$)", labelpad=5, fontsize=24)
    for i in ["top","bottom","left","right"]:
        ax.spines[i].set_linewidth(2)
    ax.grid(True)
    ax.set_ylabel("Fraction of \nNative Contacts", labelpad=5, fontsize=24)
    plt.ylim([-0.1, 1])
    _, xmax = ax.get_xlim()
    ax.set_xlim(0,xmax)
    plt.legend(fontsize=24)

    plt.savefig(f"{ path }/salt_contacts.pdf")
    plt.close()

    return None

def plot_hbonds_count(hbond_count, times, path):
    """Make a plot for the number of hbonds at each time step.

    Parameters
    ----------
    holo : Conform object
        The conform object for the holo/open state contains the relevant numpy arrays, 
        accessed via the "get_array()" module.
    apo : Conform object
        The conform object for the apo/closed state contains the relevant numpy arrays, 
        accessed via the "get_array()" module.
    path : str
        Path to the figure storage directory.

    Returns
    -------
    None.

    """
    fig, ax = plt.subplots(constrained_layout=True, figsize=(12,8))

    plt.scatter(times[::100]/(1e6), hbond_count[::100], s=150,
                color="#00C27B")

    # Plot settings
    ax.tick_params(axis='y', labelsize=18, direction='in', width=2, \
                    length=5, pad=10)
    ax.tick_params(axis='x', labelsize=18, direction='in', width=2, \
                    length=5, pad=10)
    for i in ["top","bottom","left","right"]:
        ax.spines[i].set_linewidth(2)
    ax.grid(True)
    ax.set_xlabel(r"Time ($\mu$s)", labelpad=5, fontsize=24)
    ax.set_ylabel(r"$N_{HB}$", labelpad=5, fontsize=24)
    plt.ylim([0,100])

    plt.savefig(f"{ path }/hbonds_count.pdf")
    plt.close()

    return None

def plot_rgyr(holo, apo, path):
    """Makes a Radius of Gyration plot.

    Parameters
    ----------
    holo : Conform object
        The conform object for the holo/open state contains the relevant numpy arrays, 
        accessed via the "get_array()" module.
    apo : Conform object
        The conform object for the apo/closed state contains the relevant numpy arrays, 
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
    labels = [r"$R_G$ Open conform", r"$R_G$ Closed conform"]

    for i, c in enumerate([holo, apo]):

        r_gyr = c.get_array("rad_gyr")
        time = r_gyr[:,0] / 1000
        if i == 0:
            time_lab = time
        plt.plot(time[::10], r_gyr[:,1][::10], lw=3, color=colors[i], label=labels[i], alpha=0.8,
                path_effects=[pe.Stroke(linewidth=5, foreground='#595959'), pe.Normal()])

    # Plot settings
    ax.tick_params(axis='y', labelsize=18, direction='in', width=2, \
                    length=5, pad=10)
    ax.tick_params(axis='x', labelsize=18, direction='in', width=2, \
                    length=5, pad=10)
    xticks = np.arange(0,len(time_lab),1000)

    ax.set_xticks(xticks)
    x_labels = list(map(lambda x : str(x/1000).split(".")[0], xticks))
    ax.set_xticklabels(x_labels)
    for i in ["top","bottom","left","right"]:
        ax.spines[i].set_linewidth(2)
    ax.grid(True)
    ax.set_xlabel(r"Time ($\mu s$)", labelpad=5, fontsize=24)
    ax.set_ylabel(r"$R_G$ ($\AA$)", labelpad=20, fontsize=24)
    plt.legend(fontsize=24)
    y_min, ymax = ax.get_ylim()
    ax.set_ylim(y_min-0.2,ymax+0.5)

    plt.savefig(f"{ path }/rad_gyration.pdf", dpi=300)
    plt.close()

    return None

if __name__ == '__main__':
    main(sys.argv)
