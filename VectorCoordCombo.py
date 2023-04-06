import sys
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import MDAnalysis as mda
import TrajFunctions as tf
from MDAnalysis.analysis import align
import pandas as pd

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
        parser.add_argument("-p", "--plot_coord",
    						action = "store_true",
    						dest = "plot_coord",
    						default = False,
    						help = "Make a plot of the reaction coordinates.")
        parser.add_argument("-u", "--restraint_conforms",
                            action = "store_true",
                            dest = "restrain",
                            default = False,
                            help = "Extract conformations for restraints in umbrella sampling.")
        args = parser.parse_args()

    except argparse.ArgumentError:
    	print("Command line arguments are ill-defined, please check the arguments")
    	raise

    global home_path, struct_path, holo_state, apo_state, fig_path, group, colorbar, restrain

    # Assign colorbar boolean from argparse
    colorbar = args.colorbar
    group = args.group
    recalc = args.recalc
    restrain = args.restrain

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

    df_path = f"{ path_head }/simulate/cat_trajs/dataframe_beta_vector.csv"

    if not os.path.exists(df_path) or colorbar: 

        columns = ["traj", "ts", "doth", "dota"]
        df = pd.DataFrame(columns=columns)

        for name, traj in trajs.items():
            
            u = mda.Universe(top, traj, topology_format="ITP", dt=1000)
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

                new_row = {"traj" : name, "ts" : ts.frame * 1000, 
                            "doth" : np.dot(vec, vec_holo), "dota" : np.dot(vec, vec_apo)}

                # Append the new row to the DataFrame
                df = pd.concat([df, pd.DataFrame(new_row, index=[0])])

            dot_prods[name] = np.array((dot_holo, dot_apo))

            df.to_csv(df_path, index=False)

    else: 

        df = pd.read_csv(df_path)

    num_us = 10

    if restrain:

        restraint_pts = np.zeros((num_us,2))
        restraint_pts[:,0] = np.linspace(210, 450, num_us)
        restraint_pts[:,1] = [-43/40 * i + 675/2 for i in restraint_pts[:,0]]

        with open(f"{ path_head }/simulate/cat_trajs/select_conforms.txt", "w") as f:
            f.truncate()

        with open(f"{ path_head }/simulate/cat_trajs/select_conforms.txt", "a") as f:

            for i in range(num_us):

                distances = np.sqrt(np.sum(np.square(df[['doth', 'dota']] - restraint_pts[i,:]), axis=1))

                df[f"distances_{i}"] = distances

                # Select the row with the minimum distance
                n = df.loc[df[f'distances_{i}'].idxmin()]

                # Print the nearest row
                f.write(f"""Point {i+1} : (Traj : {n["traj"]}), 
        (Time (ps) : {n["ts"]}),
        (Dot holo : {n["doth"]}), 
        (Dot apo : {n["dota"]}), 
        (Distance : {n[f"distances_{i}"]})\n""")

    print(df)

    if plot_rxn_coord:
        plot_rxn_coord(df, restraint_pts)

def plot_rxn_coord(df, restraint_pts):
    # Plot the two products over the traj
    fig, ax = plt.subplots(constrained_layout=True, figsize=(12,8))
    colors = ["#ff5500", "#ffcc00", "#d9ff00", "#91ff00", "#2bff00", "#00ffa2",
              "#00ffe1", "#00e5ff", "#0091ff", "#0037ff", "#EAAFCC", "#A1DEA1"]

    trajs = unique_values = df['traj'].unique().tolist()

    for i, t in enumerate(trajs):

        if t in ["open conform", "closed conform"]:
            alpha = 1
            m = "o"
        else:
            m, alpha = (".", 0.4)

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
            new_df = df[df["traj"] == t]
            ax.scatter(new_df.doth, new_df.dota, label=t, alpha=1,
                       marker=m, color=colors[i], s=150)

    ax.scatter(restraint_pts[:,0], restraint_pts[:,1], label="Restrain at", 
                marker="o", color="#949494", edgecolors="#404040", s=150)

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
    elif restrain:
        plt.savefig(f"{ fig_path }/dot_prod_points.png", dpi=300)
    else:
        plt.savefig(f"{ fig_path }/dot_prod_{ group }.pdf", dpi=300)
    plt.close()

    return None

if __name__ == '__main__':
    main(sys.argv)
