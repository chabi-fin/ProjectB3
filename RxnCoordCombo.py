import sys
import numpy as np
import os
import matplotlib.pyplot as plt
import MDAnalysis as mda
import TrajFunctions as tf
from MDAnalysis.analysis import align

global home_path, struct_path, holo_state, apo_state

def main(argv):

    path = os.getcwd()
    struct_path = "/home/lf1071fu/project_b3/structures"

    holo_state = mda.Universe(f"{ struct_path }/holo_state.pdb")
    apo_state = mda.Universe(f"{ struct_path }/apo_state.pdb")
    ref_state = mda.Universe(f"{ struct_path}/cpd_initial_holo.gro")

    # Load in universe objects for the simulation and the reference structures
    top = f"{ path }/topol.top"

    # find some core residues with low RMSF
    if not os.path.exists(f"{ path }/core_res.npy"):
        u = mda.Universe(top, f"{ path }/full_holo_apo.xtc", topology_format="ITP")
        calphas, rmsf = tf.get_rmsf(u, top, path)
        core_res = calphas[(rmsf < 1.5)]
        np.save(f"{ path }/core_res.npy", core_res)
    else:
        core_res = np.load(f"{ path }/core_res.npy")

    # Make a selection string for the core alignment
    aln_str = "protein and name CA and ("
    core_holo = [f"resnum {i} or " for i in core_res]
    core_apo = [f"resnum {i + 543} or " for i in core_res]
    core_apo = aln_str + "".join((core_holo + core_apo))[:-4] + ")"
    core_holo = aln_str + "".join(core_holo)[:-4] + ")"

    # Align the ref states to one structure
    align.AlignTraj(holo_state, ref_state, select=core_holo,
                    in_memory=True).run()
    align.AlignTraj(apo_state, ref_state, select=core_apo, in_memory=True).run()

    # Determine apo and holo 206-215 vectors
    r1_holo = holo_state.select_atoms("name CA and resnum 206").positions[0]
    r2_holo = holo_state.select_atoms("name CA and resnum 215").positions[0]
    vec_holo = r2_holo - r1_holo
    r1_apo = apo_state.select_atoms("name CA and resnum 749").positions[0]
    r2_apo = apo_state.select_atoms("name CA and resnum 758").positions[0]
    vec_apo = r2_apo - r2_holo

    # Make a list of trajectory paths
    af_path = "/home/lf1071fu/project_b3/simulate/af_replicas"
    trajs = {"initial holo" : f"{ path }/full_traj_holo.xtc",
             "initial apo" : f"{ path }/full_traj_apo.xtc"}
    for i in range(1,11):
        trajs[f"af {i}"] = f"{ af_path }/af_{i}/md_run/fitted_traj.xtc"

    # Initialize a dictionary to store dot product data
    dot_prods = {}

    for name, traj in trajs.items():

        dot_holo = []
        dot_apo = []

        u = mda.Universe(top, traj, topology_format="ITP")
        align.AlignTraj(u, ref_state, select=core_holo, in_memory=True).run()

        # Iterate over traj
        for ts in u.trajectory:

            # Determine the vector for res 206 to res 215
            r1 = u.select_atoms("name CA and resnum 206").positions[0]
            r2 = u.select_atoms("name CA and resnum 215").positions[0]
            vec = r2 - r1

            # Calculate and store the dot product against each reference
            dot_holo.append(np.dot(vec, vec_holo))
            dot_apo.append(np.dot(vec, vec_apo))

        dot_prods[name] = (dot_holo, dot_apo)

    # Plot the two products over the traj
    fig, ax = plt.subplots(constrained_layout=True, figsize=(12,8))
    colors = ["#c556fc", "#f74f4f", "#ff5500", "#ffcc00", "#d9ff00", "#91ff00",
              "#2bff00", "#00ffa2", "#00ffe1", "#00e5ff", "#0091ff", "#0037ff"]

    for i, t in enumerate(trajs.keys()):

        if t in ["initial holo", "initial apo"]:
            m = "P"
            alpha = 1
        else:
            m = "."
            alpha = 0.4

        d = ax.scatter(dot_prods[t][0], dot_prods[t][1], label=t, alpha=1,
                       marker=m, color=colors[i])

    # Plot settings
    ax.tick_params(axis='y', labelsize=18, direction='in', width=2, \
                    length=5, pad=10)
    ax.tick_params(axis='x', labelsize=18, direction='in', width=2, \
                    length=5, pad=10)
    for i in ["top","bottom","left","right"]:
        ax.spines[i].set_linewidth(2)
    ax.grid(True)
    ax.set_xlabel(r"dot product with holo state", labelpad=5, fontsize=28)
    ax.set_ylabel(r"dot product with apo state", labelpad=5, fontsize=28)
    plt.legend(fontsize=18)

    plt.savefig(f"/home/lf1071fu/project_b3/figures/rxn_coord/apo+holo+af.png")
    plt.close()

    return None

if __name__ == '__main__':
    main(sys.argv)
