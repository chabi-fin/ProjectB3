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

        parser.add_argument("-r", "--recalc",
                            action = "store_true",
                            dest = "recalc",
                            default = False,
                            help = """Chose whether the trajectory arrays should  be recomputed.""")
        parser.add_argument("-l", "--plot_coord",
    						action = "store_true",
    						dest = "plot_coord",
    						default = False,
    						help = "Make a plot of the reaction coordinates.")
        parser.add_argument("-u", "--restrain",
                            action = "store_true",
                            dest = "restrain",
                            default = False,
                            help = "Extract conformations for restraints in umbrella sampling.")
        parser.add_argument("-c", "--conform",
                            action = "store",
                            dest = "conform",
                            default = "closed",
                            help = "Select a reference conformation for restraints in umbrella sampling.")  
        parser.add_argument("-s", "--state",
                            action = "store",
                            dest = "state",
                            default = "apo",
                            help = "Select a system state, i.e. 'holo', 'apo' or 'mutants'.")
        parser.add_argument("-a", "--alphafold",
                            action = "store_true",
                            dest = "alphafold",
                            default = False,
                            help = "Include alpha fold trajectories.")    
        parser.add_argument("-p", "--path",
                            action = "store",
                            dest = "path",
                            default = "/home/lf1071fu/project_b3/simulate/mutation/K57G_c /home/lf1071fu/project_b3/simulate/mutation/double_mutant",
                            help = """Set path to the data directory.""")                       
        args = parser.parse_args()

    except argparse.ArgumentError:
    	print("Command line arguments are ill-defined, please check the arguments")
    	raise

    global path_head, struct_path, open_ref, closed_ref, fig_path, restrain
    global vec_closed, vec_open, beta_vec_path, conform, state

    # Assign booleans from argparse
    recalc = args.recalc
    plot_coord = args.plot_coord
    restrain = args.restrain
    conform = args.conform
    state = args.state
    alphafold = args.alphafold
    data_paths = [i for i in args.path.split(" ")]

    path = os.getcwd()
    path_head = "/home/lf1071fu/project_b3"
    fig_path = f"{ path_head }/figures/rxn_coord"
    struct_path = f"{ path_head }/structures"
    beta_vec_path =f"{ path_head }/simulate/umbrella_sampling/beta_vec_{ conform }"
    sim_paths = {
        "apo-open" : f"{ path_head }/simulate/apo_state/open/data",
        "apo-closed" : f"{ path_head }/simulate/apo_state/closed/data",
        "holo-open" : f"{ path_head }/simulate/holo_state/open/data",
        "holo-closed" : f"{ path_head }/simulate/holo_state/closed/data"}

    # Load in relevant reference structures
    open_ref = mda.Universe(f"{ struct_path }/open_ref_state.pdb", length_unit="nm")
    closed_ref = mda.Universe(f"{ struct_path }/closed_ref_state.pdb", length_unit="nm")
    ref_state = mda.Universe(f"{ beta_vec_path }/plumed_files/ref2.pdb", length_unit="nm")

    # Indicies of the inflexible residues
    core_res, core = get_core_res()

    # Align the traj and ref states to one structure
    align.AlignTraj(open_ref, ref_state, select=core, in_memory=True).run()
    align.AlignTraj(closed_ref, ref_state, select=core, in_memory=True).run()

    # Determine open + closed reference beta flap vectors in units of Angstrom
    r1, r2 = 206, 215
    r1_open = open_ref.select_atoms(f"name CA and resnum { r1 }").positions[0]
    r2_open = open_ref.select_atoms(f"name CA and resnum { r2 }").positions[0]
    vec_open = r2_open/10 - r1_open/10
    r1_closed = closed_ref.select_atoms(f"name CA and resnum { r1 }").positions[0]
    r2_closed = closed_ref.select_atoms(f"name CA and resnum { r2 }").positions[0]
    vec_closed = r2_closed/10 - r1_closed/10

    # Make a list of trajectory paths
    trajs = {}
    tops = {}
    xtc = "fitted_traj_100.xtc"
    top = "topol_protein.top"
    if alphafold:
        af_path = f"{ path_head }/simulate/af_replicas"
        for i in range(1,10):
            trajs[f"af {i}"] = f"{ af_path }/af_{i}/md_run/{ xtc }"
            tops[f"af {i}"] = f"{ af_path }/af_{i}/md_run/{ top }"
    for p in data_paths:
        n = p.split("/")[-1]
        trajs[n] = f"{ p }/{ xtc }"
        tops[n] = f"{ p }/{ top }"

    # Initialize a list of lists or dictionary
    dot_prods = {}

    df_path = f"{ path_head }/simulate/cat_trajs/dataframe_beta_vec_{ state }.csv"

    if not os.path.exists(df_path) or recalc: 

        columns = ["traj", "ts", "dot-open", "dot-closed", "angle-open", "angle-closed"]
        df = pd.DataFrame(columns=columns)

        print("DETERMINING REACTION COORDINATES FROM TRAJ DATA...")

        for name, traj in trajs.items():
            
            u = mda.Universe(tops[name], traj, topology_format="ITP", dt=1000, length_unit="nm")
            align.AlignTraj(u, ref_state, select=core, in_memory=True).run()

            dot_open = np.zeros(u.trajectory.n_frames)
            dot_closed = np.zeros(u.trajectory.n_frames)

            # Iterate over traj
            for ts in u.trajectory:

                # Determine the vector between two alpha carbons in nm
                atom1 = u.select_atoms(f"name CA and resnum { r1 }").positions[0]
                atom2 = u.select_atoms(f"name CA and resnum { r2 }").positions[0]
                vec = atom2/10 - atom1/10

                # Calculate and store the dot product against each reference
                dot_open[ts.frame] = np.dot(vec, vec_open)
                dot_closed[ts.frame] = np.dot(vec, vec_closed)

                # Check that each timestep in traj is separated by 1 ns
                instance = {"traj" : [name], "ts" : [ts.frame * 1000], 
                            "dot-open" : [np.dot(vec, vec_open)], 
                            "dot-closed" : [np.dot(vec, vec_closed)],
                            "angle-open" : [calc_theta(vec_open, vec)], 
                            "angle-closed" : [calc_theta(vec_closed, vec)]}

                # Append the new row to the DataFrame
                df_new = pd.DataFrame.from_dict(instance, orient="columns")

                df = pd.concat([df, df_new])

            dot_prods[name] = np.array((dot_open, dot_closed))

            df.to_csv(df_path, index=False)

    else: 

        df = pd.read_csv(df_path)

    if restrain:

        p1 = (np.dot(vec_closed, vec_open), np.dot(vec_closed, vec_closed))
        p2 = (3.5, 3.5)
        p3 = (np.dot(vec_open, vec_open), np.dot(vec_open, vec_closed))

        f = three_point_function(p1, p2, p3)

        if conform == "open":

            # Original linear scheme
            # restraint_pts = np.zeros((num_us,2))
            # restraint_pts[:,0] = np.linspace(2, 5, num_us)
            # restraint_pts[:,1] = [-0.83333 * i + 6.66666 for i in restraint_pts[:,0]]

            restraint_pts = np.zeros((num_us,2))
            restraint_pts[:,0] = np.linspace(p1[0],p3[0],20)
            restraint_pts[:,1] = [f[0]*x**2 + f[1]*x + f[2] for x in restraint_pts[:,0]] 

        else:

            restraint_pts = np.zeros((num_us-1,2))
            restraint_pts[:,1] = np.linspace(p1[1], p3[1], num_us-1)
            restraint_pts[:,0] = [np.roots([f[0],f[1],f[2] - y])[0] for y in restraint_pts[:,1]]

            restraint_pts = np.vstack([np.array(p1), restraint_pts])
            print(restraint_pts)

        with open(f"{ beta_vec_path }/select_conforms.txt", "w") as f:
            f.truncate()

        with open(f"{ beta_vec_path }/select_conforms.txt", "a") as f:

            col_names = ["window", "traj", "time", "dot open", "restraint open", "dot closed", "restraint closed", "distance"]
            struct_select = pd.DataFrame(columns=col_names)

            for i in range(num_us):

                distances = np.sqrt(np.sum(np.square(df[['doth', 'dota']] - restraint_pts[i,:]), axis=1))

                df[f"distances_{i}"] = distances
                df["restrain open"] = restraint_pts[i,0]
                df["restrain closed"] = restraint_pts[i,1]

                # Select the row with the minimum distance
                n = df.loc[df[f'distances_{i}'].idxmin()]

                row_data = [i, n["traj"],n["ts"],n["doth"],n["restrain open"],n["dota"],
                            n["restrain closed"],n[f"distances_{i}"]]
                struct_select = struct_select.append(pd.Series(row_data, index=col_names))

                # Print the nearest row
                f.write(f"""Point {i+1} : (Traj : {n["traj"]}), 
        (Time (ps) : {n["ts"]}),
        (Dot holo : {n["doth"]}), 
        (Restraint open: {n["restrain open"]})
        (Dot apo : {n["dota"]}), 
        (Restraint closed: {n["restrain closed"]}) 
        (Distance : {n[f"distances_{i}"]})\n""")

            struct_select.to_csv(f"{ beta_vec_path }/select_initial_struct_{ state }.csv", index=False)

    if restrain and plot_coord:
        plot_rxn_coord(df, restraints=restraint_pts)
    elif plot_coord:
        plot_rxn_coord(df, angles_coord=False)

def plot_rxn_coord(df, restraints=False, angles_coord=False):
    # Plot the two products over the traj
    fig, ax = plt.subplots(constrained_layout=True, figsize=(12,8))

    def get_color(traj):
        if "open" in traj:
            return "#EAAFCC"
        elif "closed" in traj:
            return "#A1DEA1"
        else: 
            colors = {"af 1" : "#ff5500", "af 2" : "#ffcc00", "af 3" : "#d9ff00", 
              "af 4" : "#91ff00", "af 5" : "#2bff00", "af 6" : "#00ffa2",
              "af 7" : "#00ffe1", "af 8" : "#00e5ff", "af 9" : "#0091ff",
              "K57G_c" : "#ebba34", "double_mutant" : "#36b3cf"}
            return colors[traj]

    trajs = unique_values = df['traj'].unique().tolist()
    print(trajs)

    for i, t in enumerate(trajs):

        traj_df = df[df["traj"] == t]

        if angles_coord:

            ax.scatter(traj_df["angle-open"], traj_df["angle-closed"], label=t, 
                        alpha=1, marker="o", color=get_color(t), s=150)       

        else:

            ax.scatter(traj_df["dot-open"], traj_df["dot-closed"], label=t, alpha=1,
                       marker="o", color=get_color(t), s=150)

    # Add in reference positions
    if angles_coord: 
        ax.scatter(calc_theta(vec_open, vec_open), calc_theta(vec_open, vec_closed), label="Open ref.", 
                marker="X", color="#EAAFCC", edgecolors="#404040", s=550,lw=3)
        ax.scatter(calc_theta(vec_open, vec_closed), 0, label="Closed ref.", 
                marker="X", color="#A1DEA1", edgecolors="#404040", s=550, lw=3)

    else: 
        ax.scatter(np.dot(vec_open, vec_open), np.dot(vec_open, vec_closed), label="Open ref.", 
                    marker="X", color="#EAAFCC", edgecolors="#404040", s=550,lw=3)
        ax.scatter(np.dot(vec_open, vec_closed), np.dot(vec_closed, vec_closed), label="Closed ref.", 
                    marker="X", color="#A1DEA1", edgecolors="#404040", s=550, lw=3)

        # Add restraint points
        if restrain:
            ax.scatter(restraints[:,0], restraints[:,1], label="Restrain at", 
                    marker="o", color="#949494", edgecolors="#EAEAEA", lw=3, s=150)

    # Plot settings
    ax.tick_params(axis='y', labelsize=18, direction='in', width=2, \
                    length=5, pad=10)
    ax.tick_params(axis='x', labelsize=18, direction='in', width=2, \
                    length=5, pad=10)
    for i in ["top","bottom","left","right"]:
        ax.spines[i].set_linewidth(2)
    ax.grid(True)
    if angles_coord:
        ax.set_xlabel(r"$\theta_{open}$ (rad)", labelpad=5, fontsize=24)
        ax.set_ylabel(r"$\theta_{closed}$ (rad)", labelpad=5, fontsize=24)
    else:
        ax.set_xlabel(r"$\vec{\upsilon} \cdot \vec{\upsilon}_{open}$ (nm$^2$)", labelpad=5, fontsize=24)
        ax.set_ylabel(r"$\vec{\upsilon} \cdot \vec{\upsilon}_{closed}$ (nm$^2$)", labelpad=5, fontsize=24)
    plt.legend(fontsize=18, ncol=2)

    if restrain:
        plt.savefig(f"{ fig_path }/beta_vec_{ conform }_{ state }_pts.png", dpi=300)
    elif angles_coord:
        plt.savefig(f"{ fig_path }/beta_vec_angle_{ state }.png", dpi=300)
    else:
        plt.savefig(f"{ fig_path }/beta_vec_{ state }.png", dpi=300)
    plt.show()
    plt.close()

    return None

def three_point_function(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    A = np.array([[x1**2, x1, 1], [x2**2, x2, 1], [x3**2, x3, 1]])
    b = np.array([y1, y2, y3])
    coeffs = np.linalg.solve(A, b)
    return coeffs

def calc_theta(vec_ref, vec_sim):
    """Determine the angle between two 3D vectors.
    
    Solves the expression theta = cos^(-1)((A · B) / (|A| * |B|))

    Parameters
    ----------
    vec_ref : nd.array
        The referenece beta vector as a 3D array.

    vec_sim : nd.array
        An instance of the simulated beta as a 3D array. 

    Returns
    -------
    theta : float
        The angle formed by the two vectors, in radians.

    """
    # Calculate the dot product of A and B
    dot_product = np.dot(vec_ref, vec_sim)

    # Calculate the magnitudes of A and B
    magnitude_ref = np.linalg.norm(vec_ref)
    magnitude_sim = np.linalg.norm(vec_sim)

    # Calculate the angle (theta) between A and B using the formula
    theta = np.arccos(dot_product / (magnitude_ref * magnitude_sim))

    return theta

def get_core_res(recalc=False):
    """Finds the core residues which are immobile across the conformational states.

    Uses data from the combined simulation of the apo states open and closed simulations,
    to get the calphas of the residues with an RMSF below 1.5.

    Parameters
    ----------
    recalc : boolean
        Indicates whether the core_res array should be redetermined.

    Returns
    -------
    core_res : nd.array
        Indicies for the less mobile residues across conformational states. 
    core : str
        Selection string for the core residues.

    """
    core_res_path = f"{ path_head }/simulate/apo_state/open/data"
    if not os.path.exists(f"{ core_res_path }/core_res.npy") or recalc:
        top = f"{ core_res_path }/topol.top"
        a = mda.Universe(top, f"{ core_res_path }/simulate/holo_conf/data/full_holo_apo.xtc",
                         topology_format="ITP")
        calphas, rmsf = get_rmsf(a, top, core_res_path)
        core_res = calphas[(rmsf < 1.5)]
        np.save(f"{ core_res_path }/core_res.npy", core_res)
    else:
        core_res = np.load(f"{ core_res_path }/core_res.npy")

    aln_str = "protein and name CA and ("
    core_open = [f"resid {i} or " for i in core_res]
    core_closed = [f"resid {i + 544} or " for i in core_res]
    core = aln_str + "".join((core_open + core_closed))[:-4] + ")"

    return core_res, core

if __name__ == '__main__':
    main(sys.argv)
