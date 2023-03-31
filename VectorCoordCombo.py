import sys
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import MDAnalysis as mda
import TrajFunctions as tf
from MDAnalysis.analysis import align

global home_path, struct_path, holo_state, apo_state

def main(argv):

    # Add command line arg to control color bar
    try:
        parser = argparse.ArgumentParser()

        parser.add_argument("-c", "--colorbar",
    						action = "store_true",
    						dest = "colorbar",
    						default = False,
    						help = "Include a colorbar for trajectory time.")
        parser.add_argument("-g", "--group",
    						action = "store",
    						dest = "group",
    						default = "beta",
    						help = "Choose between alpha and beta groups for RMSD.")
        parser.add_argument("-r", "--recalc",
                            action = "store_true",
                            dest = "recalc",
                            default = False,
                            help = """Chose whether the trajectory arrays should  be recomputed.""")
        args = parser.parse_args()

    except argparse.ArgumentError:
    	print("Command line arguments are ill-defined, please check the arguments")
    	raise

    # Assign colorbar boolean from argparse
    colorbar = args.colorbar
    group = args.group
    recalc = args.recalc

    path = os.getcwd()
    path_head = "/home/lf1071fu/project_b3"
    fig_path = f"{ path_head }/figures/rxn_coord"
    struct_path = f"{ path_head }/structures"
    af_path = f"{ path_head }/simulate/af_replicas"
    holo_data = f"{ path_head }/simulate/holo_conf/data"
    apo_data = f"{ path_head }/simulate/apo_conf/initial_10us"

    # Load in relevant reference structures
    holo_state = mda.Universe(f"{ struct_path }/holo_state.pdb")
    apo_state = mda.Universe(f"{ struct_path }/apo_state1.pdb")
    ref_state = mda.Universe(f"{ apo_data }/initial_CPD.pdb")

    # Load in universe objects for the simulation and the reference structures
    top = f"{ holo_data }/topol.top"

    # Find low rmsf residues for alignment fitting
    top = f"{ holo_data }/topol.top"
    if not os.path.exists(f"{ holo_data }/core_res.npy"): # or recalc:
        u = mda.Universe(top, f"{ path }/full_holo_apo.xtc",
                         topology_format="ITP")
        calphas, rmsf = tf.get_rmsf(u, top, path)
        core_res = calphas[(rmsf < 1.5)]
        np.save(f"{ holo_data }/core_res.npy", core_res)
    else:
        core_res = np.load(f"{ holo_data }/core_res.npy")

    aln_str = "protein and name CA and ("
    core_holo = [f"resid {i} or " for i in core_res]
    core_apo = [f"resid {i + 544} or " for i in core_res]
    core = aln_str + "".join((core_holo + core_apo))[:-4] + ")"

    # Align the traj and ref states to one structure
    align.AlignTraj(holo_state, ref_state, select=core, in_memory=True).run()
    align.AlignTraj(apo_state, ref_state, select=core, in_memory=True).run()

    # Determine apo and holo alpha or beta flap vectors
    if group == "alpha":
        r1 = 220
        r2 = 227
    else:
        r1 = 206
        r2 = 215
    r1_holo = holo_state.select_atoms(f"name CA and resnum { r1 }").positions[0]
    r2_holo = holo_state.select_atoms(f"name CA and resnum { r2 }").positions[0]
    vec_holo = r2_holo - r1_holo
    r1_apo = apo_state.select_atoms(f"name CA and resnum { r1 }").positions[0]
    r2_apo = apo_state.select_atoms(f"name CA and resnum { r2 }").positions[0]
    vec_apo = r2_apo - r2_holo

    # Make a list of trajectory paths
    trajs = {}
    xtc = "fitted_traj.xtc"
    if not colorbar:
        for i in range(1,11):
            trajs[f"af {i}"] = f"{ af_path }/af_{i}/md_run/{ xtc }"
    trajs["open conform"] = f"{ holo_data }/{ xtc }"
    trajs["closed conform"] = f"{ apo_data }/{ xtc }"

    # Initialize a list of lists or dictionary
    dot_prods = {}

    for name, traj in trajs.items():
        
        u = mda.Universe(top, traj, topology_format="ITP", dt=10.0)
        align.AlignTraj(u, ref_state, select=core, in_memory=True).run()

        dot_holo = np.zeros(u.trajectory.n_frames)
        dot_apo = np.zeros(u.trajectory.n_frames)

        # Iterate over traj
        for ts in u.trajectory:

            # Determine the vector between two alpha carbons
            atom1 = u.select_atoms(f"name CA and resnum { r1 }").positions[0]
            atom2 = u.select_atoms(f"name CA and resnum { r2 }").positions[0]
            vec = atom2 - atom1

            # Calculate and store the dot product against each reference
            dot_holo[ts.frame] = np.dot(vec, vec_holo)
            dot_apo[ts.frame] = np.dot(vec, vec_apo)

        dot_prods[name] = np.array((dot_holo, dot_apo))

    # Plot the two products over the traj
    fig, ax = plt.subplots(constrained_layout=True, figsize=(12,8))
    colors = ["#ff5500", "#ffcc00", "#d9ff00", "#91ff00", "#2bff00", "#00ffa2",
              "#00ffe1", "#00e5ff", "#0091ff", "#0037ff", "#EAAFCC", "#A1DEA1"]

    for i, t in enumerate(trajs.keys()):

        if t in ["open conform", "closed conform"]:
            alpha = 1
            m = "o"
        else:
            m, alpha = (".", 0.4)

        print(t, m)

        # Plot with or without the colorbar
        if colorbar:
            l = len(dot_prods[t][0])

            d = ax.scatter(dot_prods[t][0], dot_prods[t][1], c=np.arange(0,l),
                           cmap="cividis", label=t, marker=m, s=150, 
                           alpha=alpha, edgecolors="#8a8a8a", linewidths=1)

            # Colormap settings
            cbar = plt.colorbar(d)
            cbar.set_label(r'Time ($\mu$s)', fontsize=28, labelpad=10)
            cbar.ax.yaxis.set_ticks(np.arange(0,l,1000))
            cticks = list(map(lambda x: str(x/1000).split(".")[0],
                              np.arange(0,l,1000)))
            cbar.ax.yaxis.set_ticklabels(cticks)
            cbar.ax.tick_params(labelsize=16, direction='out', width=2, length=5)
            cbar.outline.set_linewidth(2)

        else:
            ax.scatter(dot_prods[t][0], dot_prods[t][1], label=t, alpha=1,
                       marker=m, color=colors[i], s=150)

    # Plot settings
    ax.tick_params(axis='y', labelsize=18, direction='in', width=2, \
                    length=5, pad=10)
    ax.tick_params(axis='x', labelsize=18, direction='in', width=2, \
                    length=5, pad=10)
    for i in ["top","bottom","left","right"]:
        ax.spines[i].set_linewidth(2)
    ax.grid(True)
    ax.set_xlabel(r"dot product with open state", labelpad=5, fontsize=24)
    ax.set_ylabel(r"dot product with closed state", labelpad=5, fontsize=24)
    plt.legend(fontsize=18, ncol=3)

    if colorbar:
        plt.savefig(f"{ fig_path }/dot_prod_{ group }_c.pdf", dpi=300)
    else:
        plt.savefig(f"{ fig_path }/dot_prod_{ group }.pdf", dpi=300)
    plt.close()

    return None

if __name__ == '__main__':
    main(sys.argv)
