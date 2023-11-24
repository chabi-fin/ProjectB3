import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.analysis import rms, align

def main(argv):

    try:
        parser = argparse.ArgumentParser()

        parser.add_argument("-r", "--recalc",
                            action = "store_true",
                            dest = "recalc",
                            default = False,
                            help = ("""Chose whether the trajectory """
                                    """arrays should  be recomputed."""))
        parser.add_argument("-c", "--coordinate",
                            action = "store",
                            dest = "rxn_coord",
                            default = "beta_vec_open",
                            help = ("""Select the desired reaction coordinate, """
                                    """e.g. "beta_vec_open" or "beta_vec_closed"."""))

        args = parser.parse_args()

    except argparse.ArgumentError:
        print("Command line arguments are ill-defined, please check the arguments")
        raise

    global path_head, df, rxn_coord

    recalc = args.recalc
    rxn_coord = args.rxn_coord
    path_head = "/home/lf1071fu/project_b3"
    home = f"{ path_head }/simulate/umbrella_sampling/{ rxn_coord }/nobackup"

    # Get values for the restraint points
    points = pd.read_csv(f"{ home }/select_initial_struct.csv")

    # Initialize DataFrames for trajectory data
    # How to convert these three so they are just one table?# 
    multi = pd.MultiIndex.from_product([[str(i) for i in range(1, 21)], ["CV", 
                                "Rad Gyr", "RMSD open beta", "RMSD open alpha", 
                                "RMSD closed beta", "RMSD closed alpha"]], 
                                names=["Window", "Property"])
    window_timesers = pd.DataFrame(columns=multi, index=np.arange(1,10004))

    if recalc or not all(list(map(lambda x : os.path.exists(x), 
        [f"{ home }/averages.csv", f"{ home }/rad_gyrs.csv",
         f"{ home }/rmsd_open.csv", f"{ home }/rmsd_closed.csv"]))):

        df = points[["window", "restraint open", "restraint closed"]] 

        # Repeat each row in df four times for run replicates
        df = df.iloc[df.index.repeat(4)]
        df = df.reset_index(drop=True)
        df['Run'] = [1, 2, 3, 4] * (len(df) // 4)

        open_state = mda.Universe(f"{ path_head }/structures/open_ref_state.pdb")
        closed_state = mda.Universe(f"{ path_head }/structures/closed_ref_state.pdb")

        # Indicies of the inflexible residues
        core_res, core = get_core_res()

        topol = f"{ home }/topol_protein.top"

        c = 0

        for window in np.arange(1,21):

            beta_vec = np.array([])
            for r in range(4): 
                if rxn_coord == "beta_vec_closed":
                    temp = np.loadtxt(f"{ home }/window{ window }/run{ r+1 }/COLVAR_250.dat")
                    window_timesers.loc[start:end, (str(window), "CV")] = temp[::10,2]
                elif rxn_coord == "beta_vec_open":
                    temp = np.loadtxt(f"{ home }/window{ window }/run{ r+1 }/COLVAR.dat")
                    window_timesers.loc[start:end, (str(window), "CV")] = temp[::10,1]
            #window_timesers.loc[:,(str(window), "CV")] = beta_vec
            rad_gs = np.zeros(10004)
            R_opens = np.zeros((10004, 2))
            R_closeds = np.zeros((10004, 2))

            print(window_timesers)
            sys.exit(1)

            for run in range(1,5):

                start = (run - 1) * 2501
                end = run * 2501
                run_path = f"{ home }/window{ window }/run{ run }"

                # Load in universe objects for the simulation run
                u = mda.Universe(topol, f"{ run_path }/fitted_traj.xtc", topology_format='ITP')
                protein = u.select_atoms("protein")
                align.AlignTraj(u, protein, select=core, in_memory=True).run()

                # Write out some sample snapshots
                with mda.Writer(f"{ run_path }/snap_shots.pdb", protein.n_atoms) as W:
                    for ts in u.trajectory:
                        if ((ts.time % 5000) == 0):
                            W.write(protein)

                # Determine and store the radius of gyration
                r_gyr = get_rgyr(u, c)
                rad_gs[start:end] = r_gyr

                # Determine and store the RMSD to ref structures
                R_open = get_rmsd(u, open_state, core, ["backbone and (resid 195-218)", 
                                    "backbone and (resid 219-231)"], "open", c)
                R_closed = get_rmsd(u, closed_state, core, ["backbone and (resid 195-218)", 
                                    "backbone and (resid 219-231)"], "closed", c)
                R_opens[start:end, :] = R_open
                R_closeds[start:end, :] = R_closed

                c += 1

            ## CONVERT TO MULTIINDEX -- ANOTHER KEY FOR THE WINDOW
            
            window_timesers_new = pd.DataFrame({f"CV" : beta_vec,
                                        f"Rad Gyr" : rad_gs,
                                        f"RMSD open beta" : R_opens[:,0],
                                        f"RMSD open alpha" : R_opens[:,1],
                                        f"RMSD closed beta" : R_closeds[:,0],
                                        f"RMSD closed alpha" : R_closeds[:,1]})

            window_timesers = pd.concat([window_timesers, window_timesers_new], axis=1)

        print("RMSD Open", rmsd_open)
        print("RMSD Closed", rmsd_closed)

        df.to_csv(f"{ home }/averages.csv")
        window_timesers.to_csv(f"{ home }/window_timesers.csv")

    else: 

        df = pd.read_csv(f"{ home }/averages.csv")
        rad_gyr = pd.read_csv(f"{ home }/rad_gyrs.csv")
        rmsd_open = pd.read_csv(f"{ home }/rmsd_open.csv")
        rmsd_closed = pd.read_csv(f"{ home }/rmsd_closed.csv")

    fig_path = f"{ path_head }/figures/umbrella/{ rxn_coord }"
    plot_Rgyr(rad_gyr, df, fig_path)
    plot_rmsd(rmsd_open, df, "open", "beta", fig_path)
    plot_rmsd(rmsd_closed, df, "closed", "beta", fig_path)
    plot_rmsd(rmsd_open, df, "open", "alpha", fig_path)
    plot_rmsd(rmsd_closed, df, "closed", "alpha", fig_path)

def get_core_res():
    """Finds the core residues which are immobile across the conformational states.

    Uses data from the combined simulation of the apo states open and closed simulations,
    to get the calphas of the residues with an RMSF below 1.5.

    Returns
    -------
    core_res : nd.array
        Indicies for the less mobile residues across conformational states. 
    core : str
        Selection string for the core residues.

    """
    core_res_path = f"{ path_head }/simulate/apo_state/open/data"
    core_res = np.load(f"{ core_res_path }/core_res.npy")

    aln_str = "protein and name CA and ("
    core_open = [f"resid {i} or " for i in core_res]
    core_closed = [f"resid {i + 544} or " for i in core_res]
    core = aln_str + "".join((core_open + core_closed))[:-4] + ")"

    return core_res, core

def get_rgyr(u, c):
    """Determine the radius of gyration and the average value.

    The radius of gyration is a measure of how compact the 
    structure is, such that an increase may indicate unfolding 
    or opening.

    Parameters
    ----------
    u : MDAnalysis.core.universe.Universe
        The universe object, with the trajectory loaded in.
    c : int
        Index for storage in the data table.

    Returns
    -------
    r_gyr : np.ndarray
        The radius of gyration as a timeseries in units of Angstrom.
    
    """
    r_gyr = []
    time_ser = []
    protein = u.select_atoms("protein")
    for ts in u.trajectory:
       r_gyr.append(protein.radius_of_gyration())
    r_gyr = np.array(r_gyr)

    df.loc[c, "ave_Rg"] = np.mean(r_gyr)

    return r_gyr

def get_rmsd(system, reference, alignment, group, ref_state, c):
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
    c : int
        Index for storage in the data table.

    Returns
    -------
    rmsd_beta : np.ndarray
        A timeseries of the RMSD against the given reference, for the first
        atom group.
    rmsd_alpha : np.ndarray
        A timeseries of the RMSD against the given reference, for the second
        atom group.
    """
    if type(group) != list:
        group = [group]
    R = rms.RMSD(system,
                 reference,  # reference universe or atomgroup
                 select=alignment,  # group to superimpose and calculate RMSD
                 groupselections=group)  # groups for RMSD
    R.run()

    rmsd = R.results.rmsd
                    
    df.loc[c, f"ave_R_{ ref_state }_beta"] = np.mean(rmsd[:,3])
    df.loc[c, f"ave_R_{ ref_state }_alpha"] = np.mean(rmsd[:,4])

    return rmsd[:,3:5]

def plot_Rgyr(df, ave, path):
    fig, ax = plt.subplots(constrained_layout=True, figsize=(12,8))

    colors = ['#0D2644', '#F1684E', '#1B9AAA', '#CE3175', '#3E3D3D',
            '#6ABE30', '#5C5D66', '#8D230F', '#515B6B', '#C03A2B',
            '#2E3D49', '#CB7920', '#8D3B72', '#4F6457', '#EDAD0B',
            '#76295D', '#2D566C', '#F4AA76', '#4B5754', '#F05133']

    vec_state = rxn_coord.split("_")[-1] 
    x = ave.groupby("window")[f"restraint { vec_state }"].mean().values
    ave_y = ave.groupby("window")["ave_Rg"].mean().values

    for i in range(1,21):

        ax.scatter(df[f"Window_{i}_C"], df[f"Window_{i}_rg"],
                    marker="o", label=f"window {i}", s=50, color=colors[i-1])

    y_min, _ = ax.get_ylim()

    ax.vlines(x=x, ymin=[y_min]*len(x), ymax=ave_y, color='#949494', ls=':', lw=3,
                clip_on=False)

    ax.scatter(x, ave_y, marker="X", s=300, color="#404040", label=r"mean $R_{G}$",
                edgecolors="#F5F5F5")

    ax.set_ylabel(r"Radius of Gyration ($\AA$)")
    if vec_state == "open":
        ax.set_xlabel(r"$\vec{\upsilon} \cdot \vec{\upsilon}_{open}$ (nm$^2$)")
    else:
        ax.set_xlabel(r"$\vec{\upsilon} \cdot \vec{\upsilon}_{closed}$ (nm$^2$)")

    plt.legend(ncol=3, fontsize=18)

    plt.savefig(f"{ path }/Rgyr_{ vec_state }.png", dpi=300)
    plt.close()

def plot_rmsd(rmsd, ave, ref_state, group, path):
    fig, ax = plt.subplots(constrained_layout=True, figsize=(12,8))

    colors = ['#0D2644', '#F1684E', '#1B9AAA', '#CE3175', '#3E3D3D',
            '#6ABE30', '#5C5D66', '#8D230F', '#515B6B', '#C03A2B',
            '#2E3D49', '#CB7920', '#8D3B72', '#4F6457', '#EDAD0B',
            '#76295D', '#2D566C', '#F4AA76', '#4B5754', '#F05133']

    vec_state = rxn_coord.split("_")[-1] 
    x = ave.groupby("window")[f"restraint { vec_state }"].mean().values
    ave_y = ave.groupby("window")[f"ave_R_{ ref_state }_{ group }"].mean().values

    for i in range(1,21):

        ax.scatter(rmsd[f"Window_{ i }_C"], rmsd[f"Window_{ i }_R_{ group }"],
                marker="o", label=f"window {i}", s=50, color=colors[i-1])

    y_min, _ = ax.get_ylim()

    ax.vlines(x=x, ymin=[y_min]*len(x), ymax=ave_y, color='#949494', ls=':', lw=3,
                clip_on=False)

    ax.scatter(x, ave_y, marker="X", s=300, color="#404040", label=r"mean RMSD",
                edgecolors="#F5F5F5")

    if vec_state == "open":
        ax.set_xlabel(r"$\vec{\upsilon} \cdot \vec{\upsilon}_{open}$ (nm$^2$)")
    else:
        ax.set_xlabel(r"$\vec{\upsilon} \cdot \vec{\upsilon}_{closed}$ (nm$^2$)")
    if ref_state == "open" and  group == "beta":
        ax.set_ylabel(r"$RMSD_{open}$ of $\beta-$flap ($\AA$)")
    elif ref_state == "open" and  group == "alpha":
        ax.set_ylabel(r"$RMSD_{open}$ of $\alpha-$flap ($\AA$)")
    elif ref_state == "closed" and  group == "beta":
        ax.set_ylabel(r"$RMSD_{closed}$ of $\beta-$flap ($\AA$)")
    elif ref_state == "closed" and  group == "alpha":
        ax.set_ylabel(r"$RMSD_{closed}$ of $\alpha-$flap ($\AA$)")

    plt.legend(ncol=3, fontsize=18)

    plt.savefig(f"{ path }/RMSD_{ ref_state }_{ group }.png", dpi=300)
    plt.close()

if __name__ == '__main__':
    main(sys.argv)