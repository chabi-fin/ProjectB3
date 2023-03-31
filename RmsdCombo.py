import sys
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import MDAnalysis as mda
import TrajFunctions as tf
from MDAnalysis.analysis import align, rms

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

    # Set up paths
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

    # Set up selection strings for atom groups
    #beta_flap = "backbone and (resnum 195-231 or resnum 740-776)"
    if group == "alpha":
        flap = "backbone and (resid 219-231 or resid 763-775)"
    elif group == "backbone":
        flap = "backbone and (resid 8-251 or resid 552-795)"
    else:
        flap = "backbone and (resid 195-218 or resid 739-762)"
    aln_str = "protein and name CA and ("
    core_holo = [f"resid {i} or " for i in core_res]
    core_apo = [f"resid {i + 544} or " for i in core_res]
    core = aln_str + "".join((core_holo + core_apo))[:-4] + ")"

    # Align the ref states to one structure
    align.AlignTraj(holo_state, ref_state, select=core, in_memory=True).run()
    align.AlignTraj(apo_state, ref_state, select=core, in_memory=True).run()
    holo_flap = holo_state.select_atoms(flap)
    apo_flap = apo_state.select_atoms(flap)

    # Make a list of trajectory paths
    xtc = "fitted_traj.xtc"
    trajs = {"open conform" : f"{ holo_data }/{ xtc }",
             "closed conform" : f"{ apo_data }/{ xtc }"}
    if not colorbar:
        for i in range(1,11):
           trajs[f"af {i}"] = f"{ af_path }/af_{i}/md_run/{ xtc }"

    # Store all the beta flap rmsd's of each traj in a dictionay
    rmsds = {}

    ref_holo_on_apo = rms.rmsd(holo_flap.positions, apo_flap.positions)
    ref_apo_on_holo = rms.rmsd(apo_flap.positions, holo_flap.positions)

    print(ref_holo_on_apo, ref_apo_on_holo)

    # Get data from the various trajectories
    for name, traj in trajs.items():
        print(name)

        # Load in and align the traj
        u = mda.Universe(top, traj, topology_format="ITP")
        align.AlignTraj(u, ref_state, select=core, in_memory=True).run()

        r_holo = np.zeros(u.trajectory.n_frames)
        r_apo = np.zeros(u.trajectory.n_frames)

        uf = u.select_atoms(flap)

        for ts in u.trajectory:

            A = uf.positions.copy()

            r_holo[ts.frame] = rms.rmsd(A,holo_flap.positions)
            r_apo[ts.frame] = rms.rmsd(A,apo_flap.positions)

        #rmsds[name] = (R_holo.results.rmsd[:,3], R_apo.results.rmsd[:,3])
        rmsds[name] = np.array((r_holo, r_apo))

    # Plot the rsmd against each reference over the traj
    fig, ax = plt.subplots(constrained_layout=True, figsize=(12,8))
    colors = ["#EAAFCC", "#A1DEA1", "#ff5500", "#ffcc00", "#d9ff00", "#91ff00",
              "#2bff00", "#00ffa2", "#00ffe1", "#00e5ff", "#0091ff", "#0037ff"]

    # Add data from each traj to the plot
    for i, t in enumerate(trajs.keys()):

        if t in ["open conform", "closed conform"]:
            alpha = 1
            m = "o"
        else:
            m, alpha = (".", 0.4)

        # Plot with or without the colorbar
        if colorbar:
            l = len(rmsds[t][0])
            d = ax.scatter(rmsds[t][0], rmsds[t][1], c=np.arange(0,l),
                        cmap="cividis", label=t, marker=m)

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
            ax.scatter(rmsds[t][0], rmsds[t][1], label=t, alpha=1,
                           marker=m, color=colors[i], s=200)

    # Plot settings
    ax.tick_params(axis='y', labelsize=18, direction='in', width=2, \
                    length=5, pad=10)
    ax.tick_params(axis='x', labelsize=18, direction='in', width=2, \
                    length=5, pad=10)
    for i in ["top","bottom","left","right"]:
        ax.spines[i].set_linewidth(2)
    ax.grid(True)
    ax.set_xlabel(r"RMSD to open state", labelpad=5, fontsize=24)
    ax.set_ylabel(r"RMSD to closed state", labelpad=5, fontsize=24)
    _, xmax = ax.get_xlim()
    _, ymax = ax.get_ylim()
    ax.set_xlim(0,xmax)
    ax.set_ylim(0,ymax)
    if group in ["alpha", "beta"]:
        plt.title(f"RMSD of the { group } flap", fontsize=24)
    else:
        plt.title(f"RMSD of the { group }", fontsize=24)
    plt.legend(fontsize=18, ncol=3)

    if colorbar:
        plt.savefig(f"{ fig_path }/rmsd_{ group }_c.pdf", dpi=300)
    else:
        plt.savefig(f"{ fig_path }/rmsd_{ group }.pdf", dpi=300)
    plt.close()

    return None

if __name__ == '__main__':
    main(sys.argv)
