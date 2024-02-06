import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.analysis import rms, align
from MDAnalysis.analysis.distances import distance_array
sys.path.insert(0, "/home/lf1071fu/project_b3/ProjectB3")
import config.settings as config
from tools import utils, traj_funcs
import matplotlib.colors as mcolors

def main(argv):

    try:
        parser = argparse.ArgumentParser()

        parser.add_argument("-p", "--path",
                            action = "store",
                            dest = "path",
                            default = ("umbrella/holo_state/nobackup"),
                            help = ("Set relative path to the data "
                                "directory."))
        parser.add_argument("-f", "--figpath",
                            action = "store",
                            dest = "fig_path",
                            default = ("umbrella/holo_state"),
                            help = ("Set a relative path destination for"
                                " the figure."))
        parser.add_argument("-s", "--state",
                            action = "store",
                            dest = "state",
                            default = "holo",
                            help = ("Select state as 'apo', 'holo' or "
                                "'mutant'."))
        parser.add_argument("-r", "--recalc",
                            action = "store_true",
                            dest = "recalc",
                            default = False,
                            help = ("Chose whether the trajectory arrays"
                                " should  be recomputed."))
        args = parser.parse_args()

    except argparse.ArgumentError:
        print("Command line arguments are ill-defined, please check the arguments")
        raise

    global path_head

    recalc = args.recalc
    home = f"{ config.data_head }/{ args.path }"
    fig_path = f"{ config.figure_head }/{ args.fig_path }"
    state = args.state

    # Get values for the restraint points
    points = pd.read_csv(f"{ home }/restraint_pts.csv")

    if recalc or not os.path.exists(f"{ home }/averages.csv"): 

        df = points[["Window", "OpenPoints", "ClosedPoints"]]

        # Repeat each row in df four times for run replicates
        df = df.iloc[df.index.repeat(4)]
        df = df.reset_index(drop=True)
        df['Run'] = [1, 2, 3, 4] * (len(df) // 4)

        # Initialize the new columns
        cols = ["ave_Rg", "ave_RMSD_open", "ave_RMSD_closed",
                "K57--E200", "N53--E200", "L212--E200", "K249--K232",
                "K249--S230a", "K249--S230b", "K249--S230c", "S230--I226",
                "K249--E233", "R209--S231", "R209--N197", "R209--E253",
                "R202--E210", "K221--E223", "R208--E222", "K57--G207",
                "K57--V201", "R28--S205"]
        if state == "holo":
            cols.extend(["K221--IP6", "R208--IP6", "R28--IP6", 
                "R32--IP6", "Y34--IP6", "K249--IP6", "R209--IP6", 
                "K104--IP6", "K57--IP6", "K232--IP6", "Y234--IP6"])
        for col in cols:
            df[col] = 0

        open_state = mda.Universe(f"{ config.struct_head }/open_ref_state.pdb")
        closed_state = mda.Universe(f"{ config.struct_head }/closed_ref_state.pdb")

        # Indicies of the inflexible residues
        core_res, core = traj_funcs.get_core_res()

        topol = f"{ home }/topol_protein.top"

        c = 0
        nw = {"apo" : 181, "holo" : 170, "mutant" : 0}

        for window in np.arange(1,nw[state]):
            for run in [1,2,3,4]:
                
                run_path = f"{ home }/window{ window }/run{ run }"

                # Load in universe objects for the simulation run
                u = mda.Universe(topol, f"{ run_path }/fitted_traj.xtc", 
                        topology_format='ITP')
                protein = u.select_atoms("protein")
                align.AlignTraj(u, protein, select=core, in_memory=True).run()

                # Write out some sample snapshots
                with mda.Writer(f"{ run_path }/snap_shots.pdb", protein.n_atoms) as W:
                    for ts in u.trajectory:
                        if ((ts.time % 5000) == 0):
                            W.write(protein)

                # Determine and store the radius of gyration
                ave_r_gyr = get_rgyr(u)
                df.loc[c, "ave_Rg"] = ave_r_gyr

                # Determine and store the RMSD to ref structures
                ave_open = get_rmsd(u, open_state, core, ["backbone and (resid 195-218)"], "open")
                ave_closed = get_rmsd(u, closed_state, core, ["backbone and (resid 195-218)"], "closed")
                df.loc[c, "ave_RMSD_open"] = ave_open
                df.loc[c, "ave_RMSD_closed"] = ave_closed

                # Determine the distances for critical contacts
                ave_ds = get_contact_dists(u)
                for col, ave_d in zip(cols[3:], ave_ds):
                    df.loc[c, col] = ave_d

                print(df.loc[c,:])

                c += 1
      
        df.to_csv(f"{ home }/averages.csv")

    else: 

        df = pd.read_csv(f"{ home }/averages.csv")

    print(df[8:])
    print(df.columns[8:])
    print("Min:", np.min(df[8:]), "\nMax:", 
        np.max(df[8:]))

    #plot_Rgyr(df, fig_path)
    #plot_rmsd(df, "open", fig_path)
    #plot_rmsd(df, "closed", fig_path)
    for col in df.columns[8:]:
        plot_ave_dists(df, col, fig_path)
    plot_windows(df, fig_path)

def get_rgyr(u):
    """Determine the average radius of gyration.

    The radius of gyration is a measure of how compact the 
    structure is, such that an increase may indicate unfolding 
    or opening.

    Parameters
    ----------
    u : MDAnalysis.core.universe.Universe
        The universe object, with the trajectory loaded in.

    Returns
    -------
    r_gyr : float
        The averageradius of gyration in units of Angstrom.
    
    """
    r_gyr = []
    time_ser = []
    protein = u.select_atoms("protein")
    for ts in u.trajectory:
       r_gyr.append(protein.radius_of_gyration())
    r_gyr = np.array(r_gyr)

    ave_r_gyr = np.mean(r_gyr)

    return ave_r_gyr

def get_rmsd(system, reference, alignment, group, ref_state):
    """Determines the rmsd over the trajectory against a reference structure.

    The MDAnalysis.analysis.rms.results array is saved as a numpy array file,
    which can be loaded if it has alreay been determined.

    Parameters
    ----------
    system : MDAnalysis.core.universe
        The universe object, with the trajectory loaded in.
    reference : MDAnalysis.core.universe
        The topology of the reference structure.
    alignment : MDAnalysis.core.groups.AtomGroup
        The group for the alignment, i.e. all the alpha carbons.
    group : MDAnalysis.core.groups.AtomGroup
        The group for the RMSD, e.g. the beta flap residues.
    ref_state : str
        The reference state as a conformational description, i.e. "open" or "closed".

    Returns
    -------
    rmsd_arr : np.ndarray
        A timeseries of the RMSD against the given reference, for the given
        atom group.
    """
    if type(group) != list:
        group = [group]
    R = rms.RMSD(system,
                 reference,  # reference universe or atomgroup
                 select=alignment,  # group to superimpose and calculate RMSD
                 groupselections=group)  # groups for RMSD
    R.run()

    rmsd_ave = np.mean(R.results.rmsd[:,3])

    return rmsd_ave

def get_contact_dists(u):
    """Calculates the distance for a key salt bridge.

    Calculates the distance between the nitrogen in the side chain of the 
    lysine residue (K57) or the alpha carbon for the mutated residue (K57G) and
    the oxygen in the side chain of glutamate (E200).

    Parameters
    ----------
    system : MDAnalysis.core.universe
        The universe object, with the trajectory loaded in.
    
    Returns
    -------
    ave_d : float
        The mean distance K57--E200 in Angstrom. 
    """
    acid_base_pairs = [
            (("resid 57 and name NZ*"),("resid 200 and name OE*")),
            (("resid 53 and name ND2"),("resid 200 and name OE*")),
            (("resid 212 and name N"),("resid 200 and name O")),
            (("resid 249 and name NZ"),("resid 232 and name O")),
            (("resid 249 and name N"),("resid 230 and name OG")),
            (("resid 249 and name N"),("resid 230 and name O")),
            (("resid 249 and name NZ"),("resid 230 and name O")),
            (("resid 230 and name N"),("resid 226 and name O")),
            (("resid 249 and name NZ"),("resid 233 and name OE*")),
            (("resid 209 and name NH*"),("resid 231 and (name OG or name O)")),
            (("resid 209 and name NH*"),("resid 197 and name OD*")),
            (("resid 209 and name NH*"),("resid 253 and name OE*")),
            (("resid 202 and name NE"),("resid 210 and name OE*")),
            (("resid 221 and name NZ"),("resid 223 and name OE*")),
            (("resid 208 and name NH*"),("resid 222 and name OE*")),
            (("resid 57 and name NZ"),("resid 207 and name O")),
            (("resid 57 and name NZ"),("resid 201 and name O")),
            (("resid 28 and name NH*"),("resid 205 and name O")),
            # (("resid 221 and name NZ"),("resname IPL and name O*")),
            # (("resid 208 and name N*"),("resname IPL and name O*")),
            # (("resid 28 and name NH*"),("resname IPL and name O*")),
            # (("resid 32 and name NH*"),("resname IPL and name O*")),
            # (("resid 34 and name OH"),("resname IPL and name O*")),
            # (("resid 249 and name NZ"),("resname IPL and name O*")),
            # (("resid 209 and name NH*"),("resname IPL and name O*")),
            # (("resid 104 and name NZ"),("resname IPL and name O*")),
            # (("resid 57 and name NZ"),("resname IPL and name O*")),
            # (("resid 232 and name NZ"),("resname IPL and name O*")),
            # (("resid 234 and name OH"),("resname IPL and name O*")),
    ]

    pairs = []

    for b, a in acid_base_pairs:
        sel_basic = u.select_atoms(b)
        sel_acidic = u.select_atoms(a)
        pairs.append((sel_basic, sel_acidic))

    print(pairs)

    #dist_pair1 = distance_array(u.coord[ca_pair1_1], u.coord[ca_pair1_2])
    distances = np.zeros((u.trajectory.n_frames, len(acid_base_pairs)))

    # Loop over all frames in the trajectory
    for ts in u.trajectory:
        # Calculate the distances between the four acid-base pairs for this frame
        for i in range(len(pairs)):
            d = distance_array(pairs[i][0].positions, pairs[i][1].positions)

            # Store the distances in the distances array
            distances[ts.frame, i] = np.min(d)

    ave_d = np.mean(distances, axis=0)
    print(f"Average distances for window: \n{ ave_d }\nShape { ave_d.shape }")

    return ave_d

def plot_Rgyr(df, path):
    fig, ax = plt.subplots(constrained_layout=True, figsize=(12,8))

    ave_Rg = df.groupby('Id')['ave_Rg'].min().values
    print(ave_Rg)

    d = ax.scatter(df[df["Run"] == 1]["OpenPoint"], 
                    df[df["Run"] == 1]["ClosedPoint"], c=ave_Rg, 
                    cmap="cividis", marker="o", edgecolors="#404040", s=200, lw=2)

    # Colormap settings
    cbar = plt.colorbar(d)
    cbar.set_label(r'Radius of Gyration ($\AA$)', fontsize=32, labelpad=10)

    ax.set_xlabel(r"$\xi_1$ (nm$^2$)")
    ax.set_ylabel(r"$\xi_2$ (nm$^2$)")

    plt.savefig(f"{ path }/window_Rgyr.png", dpi=300)
    plt.close()

def plot_rmsd(df, ref_state, path):
    fig, ax = plt.subplots(constrained_layout=True, figsize=(12,8))

    rmsd = df.groupby('Id')[f"ave_RMSD_{ ref_state }"].min().values

    d = ax.scatter(df["OpenPoint"][df["Run"] == 1], 
                    df["ClosedPoint"][df["Run"] == 1], c=rmsd, 
                    cmap="cividis", marker="o", edgecolors="#404040", s=200, lw=2)

    # Colormap settings
    cbar = plt.colorbar(d)
    cbar.set_label(f'RMSD$_{ ref_state }$ ($\AA$)', fontsize=32, labelpad=10)

    ax.set_xlabel(r"$\xi_1$ (nm$^2$)")
    ax.set_ylabel(r"$\xi_2$ (nm$^2$)")

    plt.savefig(f"{ path }/window_RMSD_{ ref_state }.png", dpi=300)
    plt.close()

def plot_ave_dists(df, col, path):
    fig, ax = plt.subplots(constrained_layout=True, figsize=(12,8))

    ave_salt = df.groupby('Window')[col].mean().values
    print(min(ave_salt), max(ave_salt))

    d = ax.scatter(df[df["Run"] == 1]["OpenPoints"], 
                    df[df["Run"] == 1]["ClosedPoints"], c=ave_salt, 
                    cmap="coolwarm", marker="o", edgecolors="#404040", 
                    s=200, lw=2, norm=mcolors.Normalize(vmin=2, vmax=27))

    # Colormap settings
    cbar = plt.colorbar(d, ticks=np.arange(2, 27.1, 5))
    cbar.set_label(f"Distance { col } " + r"($\AA$)", fontsize=32, 
        labelpad=10)

    ax.set_xlabel(r"$\xi_1$ (nm$^2$)")
    ax.set_ylabel(r"$\xi_2$ (nm$^2$)")

    plt.savefig(f"{ path }/window_ave_{ col }.png", dpi=300)
    plt.close()

def plot_windows(df, path):
    fig, ax = plt.subplots(constrained_layout=True, figsize=(12,8))

    window_labs = df.groupby('Window')["Window"].mean().values.astype(int)
    x = np.array(df[df["Run"] == 1]["OpenPoints"])
    y = np.array(df[df["Run"] == 1]["ClosedPoints"])

    d = ax.scatter(x,y, marker="o", edgecolors="#404040", s=200, lw=2)

    # Add labels to each point
    for i, label in enumerate(window_labs):
        ax.annotate(label, (x[i], y[i]), textcoords="offset points", xytext=(0, 10), ha='center')

    ax.set_xlabel(r"$\xi_1$ (nm$^2$)")
    ax.set_ylabel(r"$\xi_2$ (nm$^2$)")

    plt.savefig(f"{ path }/window_labels.png", dpi=300)
    plt.close()

if __name__ == '__main__':
    main(sys.argv)