import sys
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.patches as mpatches
import mdtraj as md

def main(argv):

    try:
        parser = argparse.ArgumentParser()

        parser.add_argument("-c", "--conform",
                            action = "store",
                            dest = "conform",
                            default = "holo",
                            help = """Chose a conformer for analysis. I.e. "holo" or "apo".""")
        parser.add_argument("-s", "--subset",
                            action = "store_true",
                            dest = "subset",
                            default = False,
                            help = """Make a DSSP plot for just a subset of residues.""")
        args = parser.parse_args()

    except argparse.ArgumentError:
        print("Command line arguments are ill-defined, please check the arguments")
        raise

    # Assign group selection from argparse
    conform = args.conform   
    subset = args.subset                             

    path_head = "/home/lf1071fu/project_b3"
    fig_path = f"/home/lf1071fu/project_b3/figures/{ conform }"
    if conform == "holo":
        data_path = f"{ path_head }/simulate/holo_conf/data"
    elif conform == "apo":
        data_path = f"{ path_head }/simulate/apo_conf/initial_10us"
    else: 
        print("ERROR: chose a valid conform from command line analysis.")
        sys.exit(1)    

    t = md.load(f"{ data_path }/fitted_traj.xtc", top=f"{ data_path }/protein.gro")

    if subset:
        t = t.atom_slice(t.top.select('resid 180 to 254'))

    dssp = md.compute_dssp(t, simplified=False)

    plot_dssp(dssp, subset, fig_path)

    return None

def plot_dssp(dssp, subset, path):
    """Make a plot of the secondary structure categories over a trajectory.

    Parameters
    ----------
    dssp : np.ndarray
        A 2D array of the character codes of the secondary structure of each
        residue at each time frame; shape=(n_frames, n_residues).
    path : str
        Path to the primary working directory.

    Returns
    -------
    None.

    """
    fig, ax = plt.subplots(constrained_layout=True, figsize=(12,4))

    #codes = {"H" : "#fcba03", "E" : "#f556e0", "C" : "#0055cc"}
    codes = {"H" : 0, "B" : 1, "E" : 2, "G" : 3, "I" : 4, "T" : 5, "S" : 6,
             " " : 7, "C" : 8}
    codename = {0 : "alpha-helix", 1 : "beta-bridge", 2: "beta-strand",
                3 : r"$3_{10}$-helix", 4 : "pi-helix", 5 : "turn", 6 : "bend",
                7 : "loops", 8 : "coil"}
    dssp_colors = np.vectorize(codes.get)(dssp.T)
    im = ax.imshow(dssp_colors, aspect='auto') #, cmap='rainbow')
    micro_s = int(dssp.T.shape[1] / 1000)
    values = np.unique(dssp_colors)

    left, right = ax.get_xlim()
    if subset:
        ax.hlines([195-180,218-180], left, right, linestyles="dashed", lw=3, colors="#FF1990",
            path_effects=[pe.Stroke(linewidth=5, foreground='#595959'), pe.Normal()])
        ax.hlines([219.5-180,231-180], left, right, linestyles="dashed", lw=3, colors="#dba61f",
            path_effects=[pe.Stroke(linewidth=5, foreground='#595959'), pe.Normal()])
        ax.set_yticks(np.arange(1, 75, 10))
        ax.set_yticklabels(np.arange(180, 255, 10))

    # Plot settings
    ax.tick_params(axis='y', labelsize=18, direction='in', width=2, \
                    length=5, pad=10)
    ax.tick_params(axis='x', labelsize=18, direction='in', width=2, \
                    length=5, pad=10)
    ax.set_xticks(np.arange(0, micro_s * 1000 + 1, 1000))
    ax.set_xticklabels(np.arange(0, micro_s + 1))
    for i in ["top","bottom","left","right"]:
        ax.spines[i].set_linewidth(2)
    ax.set_xlabel("Time (µs)", labelpad=5, fontsize=24)
    ax.set_ylabel("Residue Number", labelpad=5, fontsize=24)
    # get the colors of the values, according to the colormap used by imshow
    colors = [im.cmap(im.norm(value)) for value in values]
    # create a patch (proxy artist) for every color
    patches = [mpatches.Patch(color=colors[i], label=codename[i]) \
                for i in range(len(values))]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2,
               borderaxespad=0., fontsize=18)

    if subset:
        plt.savefig(f"{ path }/dssp_flaps.pdf", dpi=300)
    else:
        plt.savefig(f"{ path }/dssp.pdf", dpi=300)
    plt.close()

    return None

if __name__ == '__main__':
    main(sys.argv)
