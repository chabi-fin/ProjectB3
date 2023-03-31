import sys
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import MDAnalysis as mda
from MDAnalysis.analysis import rms, align, dihedrals, contacts, distances, \
                                hydrogenbonds
from MDAnalysis.analysis.distances import distance_array
import pandas as pd

def main(argv):

    try:
        parser = argparse.ArgumentParser()

        parser.add_argument("-c", "--conform",
                            action = "store",
                            dest = "conform",
                            default = "holo",
                            help = """Chose a conformer for analysis. I.e. "holo" or "apo".""")
        parser.add_argument("-r", "--recalc",
                            action = "store_true",
                            dest = "recalc",
                            default = False,
                            help = """Chose whether the trajectory arrays should  be recomputed.""")
        args = parser.parse_args()

    except argparse.ArgumentError:
        print("Command line arguments are ill-defined, please check the arguments")
        raise

    # Assign group selection from argparse
    conform = args.conform
    recalc = args.recalc

    path_head = "/home/lf1071fu/project_b3"
    fig_path = f"/home/lf1071fu/project_b3/figures/{ conform }"
    str_path = f"{ path_head }/structures"
    if conform == "holo":
        data_path = f"{ path_head }/simulate/holo_conf/data"
    elif conform == "apo":
        data_path = f"{ path_head }/simulate/apo_conf/initial_10us"
    else: 
        print("ERROR: chose a valid conform from command line analysis.")
        sys.exit(1)

    #TO DO: CHECK THESE!
    traj_name = "fitted_traj"

    # Load in universe objects for the simulation and the reference structures
    u = mda.Universe(f"{ data_path }/topol.top", f"{ data_path }/{ traj_name }.xtc",
                     topology_format="ITP")

    holo_state = mda.Universe(f"{ str_path }/holo_state.pdb")
    apo_state = mda.Universe(f"{ str_path }/apo_state1.pdb")
    # Starting structure as the ref state
    ref_state = u.select_atoms("protein")

    # Find the core residues with low rmsf for both starting conformations
    if not os.path.exists(f"{ data_path }/core_res.npy"):
        top = f"{ data_path }/topol.top"
        a = mda.Universe(top, f"{ path_head }/simulate/holo_conf/data/full_holo_apo.xtc",
                         topology_format="ITP")
        calphas, rmsf = get_rmsf(a, top, data_path)
        core_res = calphas[(rmsf < 1.5)]
        np.save(f"{ data_path }/core_res.npy", core_res)
    else:
        core_res = np.load(f"{ data_path }/core_res.npy")

    # Atom group selection strings
    beta_flap_group = "backbone and (resid 195-218 or resid 739-762)"
    alpha_flap_group = "backbone and (resid 219-231 or resid 763-775)"
    combo_group = "backbone and (resid 195-231 or  resid 739-775)"
    aln_str = "protein and name CA and ("
    core_holo = [f"resid {i} or " for i in core_res]
    core_apo = [f"resid {i + 544} or " for i in core_res]
    core = aln_str + "".join((core_holo + core_apo))[:-4] + ")"

    # Store calculated outputs as numpy arrays
    np_files = [f"{ data_path }/rmsd_holo.npy", f"{ data_path }/rmsd_apo.npy",
                f"{ data_path }/rmsf.npy", f"{ data_path }/calphas.npy",
                f"{ data_path }/rad_gyration.npy",
                f"{ data_path }/rama.npy", f"{ data_path }/chi_1s.npy",
                f"{ data_path }/salt.npy", f"{ data_path}/salt_dist.npy",
                f"{ data_path }/hpairs.npy"] #, f"{ data_path }/hbonds.npy",
                #f"{ data_path }/hbond_count.npy", f"{ data_path }/times.npy"]

    if all(list(map(lambda x : os.path.exists(x), np_files))) and not recalc:

        R_holo = np.load(np_files[0], allow_pickle=True)
        R_apo = np.load(np_files[1], allow_pickle=True)
        rmsf = np.load(np_files[2], allow_pickle=True)
        calphas = np.load(np_files[3], allow_pickle=True)
        r_gyr = np.load(np_files[4], allow_pickle=True)
        rama = np.load(np_files[5], allow_pickle=True)
        # # chi_1s is a dictionay object
        chi_1s = np.load(np_files[6], allow_pickle=True)
        sc = np.load(np_files[7], allow_pickle=True)
        salt_dist = np.load(np_files[8], allow_pickle=True)
        hpairs = np.load(np_files[9], allow_pickle=True)

        # hbonds = np.load(np_files[7], allow_pickle=True)
        # hbond_count = np.load(np_files[8], allow_pickle=True)
        # times = np.load(np_files[9], allow_pickle=True)

    else:

        u.transfer_to_memory()

        align.AlignTraj(u, ref_state, select=core, in_memory=True).run()

        protein = u.select_atoms("protein")
        with mda.Writer(f"{ data_path }/snap_shots.pdb", protein.n_atoms) as W:
            for ts in u.trajectory:
                if ts.time in [1000000, 3000000, 7000000, 9000000]:
                    W.write(protein)

        # Determine RMSD to ref structures
        R_holo = get_rmsd(u, holo_state, core, ["backbone and resid 8:251",
                          beta_flap_group, alpha_flap_group, combo_group])
        R_apo = get_rmsd(u, apo_state, core, ["backbone and resid 8:251",
                          beta_flap_group, alpha_flap_group, combo_group])

        # Determine RMSF by first finding and aligning to the average structure
        calphas, rmsf = get_rmsf(u, f"{ data_path }/protein.gro", core, data_path)

        # Determine the radius of gyration
        r_gyr = get_rgyr(u)

        # Phi and Psi backbone dihedrals
        rama = dihedrals.Ramachandran(u.residues[195:232]).run()
        rama = rama.results.angles
        
        # Chi1 dihedrals
        chi_1s = dict()
        for res in u.residues[195:232]:
            group = res.chi1_selection()
            if group is not None:
                 dihs = dihedrals.Dihedral([group]).run()
                 chi_1s[res.ix] = dihs.results.angles
        
        # Salt-bridge contacts
        sc = get_salt_contacts(u, ref_state, data_path)
        salt_dist = get_bridges(u, data_path)

        # # Hydrogen bonds analysis (intramolecular, not with solvent)
        # hbonds, hbond_count, times = get_hbonds(u, path)
        hpairs = get_hbond_pairs(u, data_path)

        for f, v in zip(np_files, [R_holo, R_apo, rmsf, calphas, r_gyr, rama, chi_1s,
                                   sc, salt_dist, hpairs]): #hbonds, hbond_count, times]):
            np.save(f, v)

    # Make plots for simulation analysis
    # plot_rmsd(R_holo, R_apo, fig_path)
    # plot_rmsf(calphas, rmsf, fig_path)
    # plot_rmsd_time(R_apo, fig_path)
    # plot_rgyr(r_gyr, fig_path)

    # Plots for individual residues: Ramachandran and Chi_1
    # for i, id in enumerate(np.arange(198,232)):
    #     res = u.residues[id]
    #     plot_ramachandran(rama[:,i,0], rama[:,i,1], res, fig_path)
    # for key, chi_1 in chi_1s.items():
    #     res = u.residues[key]
    #     plot_chi1(res, chi_1, fig_path)
    #
    print(hpairs.shape)
    # plot_salt_bridges(sc, fig_path)
    # plot_bridges(salt_dist, r_gyr, fig_path)
    plot_hbond_pairs(hpairs, r_gyr, fig_path)
    #plot_hbonds_count(hbond_count, times, fig_path)

    return None

def get_rmsd(system, reference, alignment, group):
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

    rmsd_arr = R.results.rmsd

    return rmsd_arr

def get_rmsf(u, top, ref_group, path):
    """Determines the RMSF per residue.

    The average structure is calculated and then used as the reference for
    structure alignment, before calculating the rmsf per residue.

    Parameters
    ----------
    u : MDAnalysis.core.universe
        The universe object, with the trajectory loaded in.
    top : str
        The topology file, e.g. xxx.gro.
    ref_group : str
        Atom group selection string of the core residues for the alignment.
    path : str
        Path to the primary working directory.

    Returns
    -------
    resnums : MDAnalysis.core.groups.AtomGroup
        The atom group used for the RMSF calculation consists of all the alpha
        carbons in the protein.
    rmsf : np.ndarray
        The rmsf of the selected atom groups.

    """
    # Perform alignment to the average structure and write to a new traj file.
    protein = u.select_atoms("protein")
    prealigner = align.AlignTraj(u, u, select="protein and name CA",
                                 in_memory=True).run()
    ref_coords = u.trajectory.timeseries(asel=protein).mean(axis=1)
    ref = mda.Merge(protein).load_new(ref_coords[:, None, :],
                              order="afc")
    aligner = align.AlignTraj(u, ref, select=ref_group, in_memory=True).run()
    with mda.Writer(f"{ path }/rmsfit.xtc", n_atoms=u.atoms.n_atoms) as W:
        for ts in u.trajectory:
            W.write(u.atoms)

    from MDAnalysis.analysis.rms import RMSF

    u = mda.Universe(top, f"{ path }/rmsfit.xtc")
    calphas = protein.select_atoms("protein and name CA")

    rmsfer = RMSF(calphas).run()

    return calphas.resnums, rmsfer.results.rmsf

def get_salt_contacts(u, ref_state, path):
    """Runs the Contacts module to get the number of salt contacts.

    Arginine and lysine residues are considered as the basic residues, while
    aspargine and glutamine are classified as acidic residues. A soft cut off
    is used for determining contacts within a radius of 4.5 A.

    Parameters
    ----------
    u : MDAnalysis.core.universe.Universe
        The universe object, with the trajectory loaded in.
    ref_state : MDAnalysis.core.universe.Universe
        The universe object for the reference state.
    path : str
        Path to the primary working directory.

    Returns
    -------
    sc : np.ndarray
        A timeseries of the fraction of retained salt bridges, with respect to
        the reference state.
    """
    sel_basic = "(resname ARG LYS) and (name NH* NZ*)"
    sel_acidic = "(resname ASP GLU) and (name OE* OD*)"
    acidic = ref_state.select_atoms(sel_acidic)
    basic = ref_state.select_atoms(sel_basic)
    salt_contacts = contacts.Contacts(u, select=(sel_acidic, sel_basic),
                    refgroup=(acidic, basic), radius=4.5, method="soft_cut")

    salt_contacts.run()
    sc = salt_contacts.results.timeseries

    return sc

def get_bridges(u, path):
    """Calculates the distance of special salt contacts in the alpha + beta flaps.

    Arginine and lysine residues are considered as the basic residues, while
    aspargine and glutamine are classified as acidic residues.

    Parameters
    ----------
    u : MDAnalysis.core.universe.Universe
        The universe object, with the trajectory loaded in.
    path : str
        Path to the primary working directory.

    Returns
    -------
    distances : np.ndarray
        Trajectory of the four distance arrays. 

    """
    acid_base_pairs = [
        (("resid 208 and name NH*"),("resid 222 and name OE*")), # R208 -- E222
        (("resid 227 and name NZ*"),("resid 213 and name OD*")), # K227 -- D213
        #(("resid 232 and name NZ*"),("resid 149 and name OE*")), # K232 -- E149
        (("resid 202 and name NH*"),("resid 210 and name OE*")), # R202 -- E210
        (("resid 57 and name NZ*"),("resid 200 and name OE*"))] # K57 -- E200

    pairs = []

    for b, a in acid_base_pairs:
        sel_basic = u.select_atoms(b)
        sel_acidic = u.select_atoms(a)
        pairs.append((sel_basic, sel_acidic))

        #dist_pair1 = distance_array(u.coord[ca_pair1_1], u.coord[ca_pair1_2])
    distances = np.zeros((u.trajectory.n_frames, 4))

    # Loop over all frames in the trajectory
    for ts in u.trajectory:
        # Calculate the distances between the four acid-base pairs for this frame
        d1 = distance_array(pairs[0][0].positions, pairs[0][1].positions)
        d2 = distance_array(pairs[1][0].positions, pairs[1][1].positions)
        d3 = distance_array(pairs[2][0].positions, pairs[2][1].positions)
        d4 = distance_array(pairs[3][0].positions, pairs[3][1].positions)
        # Store the distances in the distances array
        distances[ts.frame] = [np.min(d1), np.min(d2), np.min(d3), np.min(d4)]

    return distances

def get_hbonds(u, path):
    """Find the intramolecular hydrogen bonds in the simulation.

    The donor-acceptor distance cutoff is 3 A and the donor-hydrogen-acceptor
    angle cutoff is minimum 150 deg. The function generates two files, one to
    track the types of hbonds formed and one to identify significant and
    persistent bonds.

    Parameters
    ----------
    u : MDAnalysis.core.universe
        The universe object, with the trajectory loaded in.
    path : str
        Path to the primary working directory.

    Returns
    -------
    hbonds : MDAnalysis...HydrogenBondAnalysis
        Each row consists of a particular h-bond observation: [frame,
        donor_index, hydrogen_index, acceptor_index, DA_distance, DHA_angle].
    hbond_count : np.ndarray
        The hbond count as a trajectory.

    """
    hbonds = hydrogenbonds.hbond_analysis.HydrogenBondAnalysis(universe=u)
    hbonds.hydrogens_sel = hbonds.guess_hydrogens("protein")
    hbonds.acceptors_sel = hbonds.guess_acceptors("protein")
    hbonds.run()
    tau_timeseries, timeseries = hbonds.lifetime()


    # these get their own np arrays
    times = hbonds.times
    hbond_count = hbonds.count_by_time()

    # Analyze the bond types
    f = open(f"{ path }/hbond_types.txt", "w")
    f.write("The average number of each type of hydrogen bond formed at each"
            " frame:\n")

    for donor, acceptor, count in hbonds.count_by_type():

        donor_resname, donor_type = donor.split(":")
        n_donors = u.select_atoms(f"resname { donor_resname } " \
                                  f"and type { donor_type }").n_atoms

        # average number of hbonds per donor molecule per frame
        # multiply by two as each hydrogen bond involves two water molecules
        mean_count = 2 * int(count) / (hbonds.n_frames * n_donors)
        if mean_count > 0.1:
            f.write(f"{donor} to {acceptor}: {mean_count:.2f}\n")
    f.close()

    f = open(f"{ path }/significant_hbonds.txt", "w")
    f.write("A descending list of h-bonds formed between atom pairs.\n"\
            "Donor, Hydrogen, Acceptor, Count\n")

    def describe_at(atom):
        a = u.atoms[atom-1]
        return f"{ a.resname }{ a.resid }--{ a.type }"

    # List the most important h-bonds by order of occurence
    c = hbonds.count_by_ids()
    for d, h, a, count in c:
        if count < 10:
            continue
        f.write(f"{ describe_at(d) }\t{ describe_at(h) }\t{ describe_at(a) } "\
                f"\t{ count }\n")
    f.close()

    return hbonds, hbond_count, times

def get_hbond_pairs(u, path):
    """Calculates the distance of special hbond contacts in the alpha + beta flaps.

    Parameters
    ----------
    u : MDAnalysis.core.universe.Universe
        The universe object, with the trajectory loaded in.
    path : str
        Path to the primary working directory.

    Returns
    -------
    distances : np.ndarray
        Trajectory of the four distance arrays. 

    """
    hbond_pairs = [
        (("resid 204 and name OD1"),("(resid 206 or resid 208) and name H")),
        (("resid 231 and name O*"),("resid 209 and name NH*")),
        (("resid 197 and name OD1"),("resid 209 and name N*")),
        (("resid 53 and name H*"),("resid 200 and name O*"))]

    pairs = []

    for a, d in hbond_pairs:
        sel_accept = u.select_atoms(a)
        sel_donor = u.select_atoms(d)
        pairs.append((sel_accept, sel_donor))

    distances = np.zeros((u.trajectory.n_frames, len(pairs)))

    # Loop over all frames in the trajectory
    for ts in u.trajectory:
        # Calculate the distances between the 3 hbond pairs for this frame
        d1 = distance_array(pairs[0][0].positions, pairs[0][1].positions)
        d2 = distance_array(pairs[1][0].positions, pairs[1][1].positions)
        d3 = distance_array(pairs[2][0].positions, pairs[2][1].positions)
        d4 = distance_array(pairs[3][0].positions, pairs[3][1].positions)

        # Store the distances in the distances array
        distances[ts.frame] = [np.min(d1), np.min(d2), np.min(d3), np.min(d3)]

    return distances

def get_rgyr(u):
    """
    """
    r_gyr = []
    protein = u.select_atoms("protein")
    for ts in u.trajectory:
       r_gyr.append((u.trajectory.time, protein.radius_of_gyration()))
    r_gyr = np.array(r_gyr)
    return r_gyr

def plot_rmsd(r_holo, r_apo, path): 
    """Makes a plot of the rmsd against two ref structures with a color bar.

    Parameters
    ----------
    r_holo : MDAnalysis.analysis.rms.RMSD or nparray
        A time series of the rmsd with the holo crystal structure as reference.
    r_apo : MDAnalysis.analysis.rms.RMSD or nparray
        A time series of the rmsd with the apo crystal structure as reference.
    path : str
        Path to the primary working directory.

    Returns
    -------
    None.

    """
    fig, ax = plt.subplots(constrained_layout=True, figsize=(12,8))

    d = ax.scatter(r_holo[:,3], r_apo[:,3], c=np.arange(0,len(r_holo[:,0])),
                cmap="cividis", label=r"Full backbone", marker="X")
    d = ax.scatter(r_holo[:,4], r_apo[:,4], c=np.arange(0,len(r_holo[:,0])),
                cmap="cividis", label=r"$\beta$-flap", marker="o")
    d = ax.scatter(r_holo[:,5], r_apo[:,5], c=np.arange(0,len(r_holo[:,0])),
                cmap="cividis", label=r"$\alpha$-flap", marker="D")

    # Colormap settings
    cbar = plt.colorbar(d)
    cbar.set_label(r'Time [$\mu$s]', fontsize=28, labelpad=10)
    cbar.ax.yaxis.set_ticks(np.arange(0,len(r_holo[:,0]),1000))
    cticks = list(map(lambda x: str(x/1000).split(".")[0],
                      np.arange(0,len(r_holo[:,0]),1000)))
    cbar.ax.yaxis.set_ticklabels(cticks)
    cbar.ax.tick_params(labelsize=16, direction='out', width=2, length=5)
    cbar.outline.set_linewidth(2)

    # Plot settings
    ax.tick_params(axis='y', labelsize=18, direction='in', width=2, \
                    length=5, pad=10)
    ax.tick_params(axis='x', labelsize=18, direction='in', width=2, \
                    length=5, pad=10)
    for i in ["top","bottom","left","right"]:
        ax.spines[i].set_linewidth(2)
    ax.grid(True)
    ax.set_xlabel(r"RMSD to holo state ($\AA$)", labelpad=5, \
                    fontsize=28)
    ax.set_ylabel(r"RMSD to apo state ($\AA$)", labelpad=5, fontsize=28)
    _, xmax = ax.get_xlim()
    _, ymax = ax.get_ylim()
    ax.set_xlim(0,xmax)
    ax.set_ylim(0,ymax)

    plt.legend(fontsize=24)

    plt.savefig(f"{ path }/rmsd.pdf")
    plt.close()

    return None

def plot_rmsf(calphas, rmsf, path):
    """Makes an RMSF plot.

    Parameters
    ----------
    resnums : MDAnalysis.core.groups.AtomGroup
        The atom group used for the RMSF calculation consists of all the alpha
        carbons in the protein.
    rmsf : np.ndarray
        The rmsf of the selected atom groups.
    path : str
        Path to the primary working directory.

    Returns
    -------
    None.

    """
    fig, ax = plt.subplots(constrained_layout=True, figsize=(12,4))
    resids = list(map(lambda x : x + 544, calphas))
    plt.plot(calphas, rmsf, lw=3, color="#47abc4")

    # Plot settings
    ax.tick_params(axis='y', labelsize=18, direction='in', width=2, \
                    length=5, pad=10)
    ax.tick_params(axis='x', labelsize=18, direction='in', width=2, \
                    length=5, pad=10)
    for i in ["top","bottom","left","right"]:
        ax.spines[i].set_linewidth(2)
    ax.grid(True)
    ax.set_xlabel(r"Residue number", labelpad=5, fontsize=28)
    ax.set_ylabel(r"RMSF ($\AA$)", labelpad=5, fontsize=28)
    bottom, top = ax.get_ylim()
    ax.vlines([195,217.5], bottom, top, linestyles="dashed", lw=3,
              colors="#FF1990")
    ax.vlines([219.5,231], bottom, top, linestyles="dashed", lw=3,
              colors="#dba61f")
    ax.set_ylim(-1,10)

    plt.savefig(f"{ path }/rmsf.pdf")
    plt.close()

    return None

def plot_rmsd_time(r_apo, path):
    """
    """
    fig, ax = plt.subplots(constrained_layout=True, figsize=(24,8))
    time = r_apo[:,0]
    #plt.plot(time, r_apo[:,2], lw=2, color="#02ab64", alpha=0.8,
    #         label="core")
    plt.plot(time[::10], r_apo[:,5][::10], lw=3, color="#02ab64", alpha=0.8,
             label="alpha flap")
    plt.plot(time[::10], r_apo[:,4][::10], lw=3, color="#dba61f", alpha=0.8,
             label="beta flap")
    plt.plot(time[::10], r_apo[:,3][::10], lw=3, color="#47abc4", alpha=0.8,
             label="full backbone")

    # Plot settings
    ax.tick_params(axis='y', labelsize=32, direction='in', width=2, \
                    length=5, pad=10)
    ax.tick_params(axis='x', labelsize=32, direction='in', width=2, \
                    length=5, pad=10)
    xticks = np.arange(0,len(time),1000)
    ax.set_xticks(xticks)
    ax.set_xticklabels(list(map(lambda x : str(x/1000).split(".")[0], xticks)))
    for i in ["top","bottom","left","right"]:
        ax.spines[i].set_linewidth(2)
    ax.grid(True)
    ax.set_xlabel(r"time ($\mu s$)", labelpad=5, fontsize=38)
    ax.set_ylabel(r"RMSD ($\AA$)", labelpad=5, fontsize=38)
    _, ymax = ax.get_ylim()
    ax.set_ylim(0,ymax)
    plt.legend(fontsize=28)

    plt.savefig(f"{ path }/rmsd_time.pdf", dpi=300)
    plt.close()

    return None

def plot_ramachandran(phis, psis, res, path):
    """Make a Ramachandran plot for the given residues.

    Parameters
    ----------
    dihs_phi : np.ndarray
        A timeseries of the phi dihedrals for the residue group.
    dihs_psi : np.ndarray
        A timeseries of the psi dihedrals for the residue group.
    resnum : MDAnalysis.core.groups.Residue
        The residue object for the ramachandran plot.
    path : str
        Path to the primary working directory.

    Returns
    -------
    None.

    """
    fig, ax = plt.subplots(constrained_layout=True, figsize=(12,8))

    d = ax.scatter(phis, psis, c=np.arange(0,len(phis)), cmap="cividis",
                   label=f"{ res.resname } { res.resid }")

    # Colormap settings
    cbar = plt.colorbar(d)
    cbar.set_label(r'Time [$\mu$s]', fontsize=28, labelpad=10)
    cbar.ax.yaxis.set_ticks(np.arange(0,len(psis),1000))
    cticks = list(map(lambda x: str(x/1000).split(".")[0],
                      np.arange(0,len(psis),1000)))
    cbar.ax.yaxis.set_ticklabels(cticks)
    cbar.ax.tick_params(labelsize=16, direction='out', width=2, length=5)
    cbar.outline.set_linewidth(2)

    # Plot settings
    ax.tick_params(axis='y', labelsize=18, direction='in', width=2, \
                    length=5, pad=10)
    ax.tick_params(axis='x', labelsize=18, direction='in', width=2, \
                    length=5, pad=10)
    for i in ["top","bottom","left","right"]:
        ax.spines[i].set_linewidth(2)
    ax.grid(True)
    ax.set_xlabel(r"$\Phi$", labelpad=5, \
                    fontsize=28)
    ax.set_ylabel(r"$\Psi$", labelpad=5, fontsize=28)
    plt.xlim([-180, 180])
    plt.ylim([-180, 180])
    plt.legend(fontsize=20)

    # Save plot to file
    plot_path = f"{ path }/ramachandran"
    if not os.path.exists(plot_path):
       os.makedirs(plot_path)
    plt.savefig(f"{ plot_path }/res_{ res.resid }.pdf")
    plt.close()

    return None

def plot_chi1(res, chi_1, path):
    """Make a chi_1 plot for the given residue.

    Parameters
    ----------
    res : MDAnalysis.core.groups.Residue
        The residue object for the chi_1 plot.
    chi_1 : np.ndarray
        A timeseries of the chi 1 dihedrals for the residue group.
    path : str
        Path to the primary working directory.

    Returns
    -------
    None.

    """
    fig, ax = plt.subplots(constrained_layout=True, figsize=(12,8))

    d = ax.hist(chi_1, bins=180, density=True, color="#b21856",
                label=f"{ res.resname } { res.resid }")

    # Plot settings
    ax.tick_params(axis='y', labelsize=18, direction='in', width=2, \
                    length=5, pad=10)
    ax.tick_params(axis='x', labelsize=18, direction='in', width=2, \
                    length=5, pad=10)
    for i in ["top","bottom","left","right"]:
        ax.spines[i].set_linewidth(2)
    ax.grid(True)
    ax.set_xlabel(r"$\chi_1$ (°)", labelpad=5, \
                    fontsize=28)
    ax.set_ylabel(r"Density", labelpad=5, fontsize=28)
    plt.xlim([-180, 180])
    plt.legend(fontsize=20)

    plot_path = f"{ path }/chi_1s"
    if not os.path.exists(plot_path):
       os.makedirs(plot_path)
    plt.savefig(f"{ plot_path }/res_{ res.resid }.pdf")
    plt.close()

    return None

def plot_salt_bridges(sc, path):
    """Make a plot for the fraction of salt contacts compared to the reference.

    A soft cut-off from 4 A was used to determine the fraction of salt bridges
    retained from the reference structure. This includes bridges between the
    charged residues, i.e. N in ARG and LYS with O in ASP and GLU.

    Parameters
    ----------
    sc : np.ndarray
        A timeseries of the fraction of salt bridge contacts, relative to the
        reference structure.
    path : str
        Path to the primary working directory.

    Returns
    -------
    None.

    """

    print(sc.shape)
    fig, ax = plt.subplots(constrained_layout=True, figsize=(12,8))
    ave_contacts = np.mean(sc[:, 1])
    print(f"average contacts = { ave_contacts }")
    sc = sc[::40]

    ax.scatter(sc[:, 0], sc[:, 1], s=150, color="#00C27B")

    # Plot settings
    ax.tick_params(axis='y', labelsize=18, direction='in', width=2, \
                    length=5, pad=10)
    ax.tick_params(axis='x', labelsize=18, direction='in', width=2, \
                    length=5, pad=10)
    for i in ["top","bottom","left","right"]:
        ax.spines[i].set_linewidth(2)
    ax.grid(True)
    ax.set_xlabel(r"Frame", labelpad=5, fontsize=24)
    ax.set_ylabel(r"Fraction of Native Contacts", labelpad=5, fontsize=24)
    plt.ylim([0, 1])

    plt.savefig(f"{ path }/salt_contacts.pdf")
    plt.close()

    return None

def plot_hbonds_count(hbond_count, times, path):
    """Make a plot for the number of hbonds at each time step.

    Parameters
    ----------
    hbond_count : np.ndarray
        A trajectory of the hydrogen bond count.
    times : np.ndarray
        The trajectory times as an array in ps.
    path : str
        Path to the primary working directory.

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

def plot_rgyr(r_gyr, path):
    fig, ax = plt.subplots(constrained_layout=True, figsize=(24,8))

    time = r_gyr[:,0] / 1000
    plt.plot(time[::10], r_gyr[:,1][::10], "--", lw=3, color="#02ab64", label=r"$R_G$")

    # Plot settings
    ax.tick_params(axis='y', labelsize=32, direction='in', width=2, \
                    length=5, pad=10)
    ax.tick_params(axis='x', labelsize=32, direction='in', width=2, \
                    length=5, pad=10)
    xticks = np.arange(0,len(time),1000)

    ax.set_xticks(xticks)
    x_labels = list(map(lambda x : str(x/1000).split(".")[0], xticks))
    ax.set_xticklabels(x_labels)
    for i in ["top","bottom","left","right"]:
        ax.spines[i].set_linewidth(2)
    ax.grid(True)
    ax.set_xlabel(r"time ($\mu s$)", labelpad=5, fontsize=38)
    ax.set_ylabel(r"Radius of Gyration $R_G$ ($\AA$)", labelpad=5, fontsize=38)
    plt.legend(fontsize=28)

    plt.savefig(f"{ path }/rad_gyration.pdf", dpi=300)
    plt.close()

    return None

def plot_bridges(salt_dist, r_gyr, path):
    fig, ax = plt.subplots(constrained_layout=True, figsize=(12,4))

    time = r_gyr[:,0] / 1000
    stride = 10
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
    labels = [r"$\alpha - \beta \:$ R208$-$E222",
              r"$\alpha - \beta \:$ K227$-$D213",
              #r"$\alpha - pocket \:$ K232-E149",
              r"$\beta - \beta \:$ R202$-$E210",
              r"Pocket$-\beta$ K57$-$E200"]

    for i in range(4):

        plt.plot(time[::stride], salt_dist[::stride,i], "-", lw=3, color=colors[i], 
            label=labels[i], alpha=0.8,
            path_effects=[pe.Stroke(linewidth=5, foreground='#595959'), pe.Normal()])

    ax.axhline(y=4.5, linestyle='--', lw=3, color='red', alpha=0.5,
        path_effects=[pe.Stroke(linewidth=5, foreground='#595959'), pe.Normal()])

    # Plot settings
    ax.tick_params(axis='y', labelsize=18, direction='in', width=2, \
                    length=5, pad=10)
    ax.tick_params(axis='x', labelsize=18, direction='in', width=2, \
                    length=5, pad=10)
    xticks = np.arange(0,len(time),1000)
    ax.set_xticks(xticks)
    ax.set_xticklabels(list(map(lambda x : str(x/1000).split(".")[0], xticks)))
    ax.set_xlabel(r"Time ($\mu s$)", labelpad=5, fontsize=24)
    for i in ["top","bottom","left","right"]:
        ax.spines[i].set_linewidth(2)
    ax.grid(True)
    ax.set_ylabel("Distance ($\AA$)", labelpad=5, fontsize=24)
    plt.ylim([0, 40])
    _, xmax = ax.get_xlim()
    ax.set_xlim(0,xmax)
    plt.legend(fontsize=24, ncol=2)

    plt.savefig(f"{ path }/salt_bridges.pdf", dpi=300)
    plt.close()

    return None

def plot_hbond_pairs(hpairs, r_gyr, path):
    fig, ax = plt.subplots(constrained_layout=True, figsize=(12,4))

    time = r_gyr[:,0] / 1000
    stride = 10
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
    labels = [r"N204$-$R208", r"S231$-$R209",
              r"N197$-$R209", r"N53$-$E200"]

    print(hpairs[::1000,3])

    for i in range(4):

        plt.plot(time[::stride], hpairs[::stride,i], "-", lw=3, color=colors[i], 
            label=labels[i], alpha=0.8,
            path_effects=[pe.Stroke(linewidth=5, foreground='#595959'), pe.Normal()])

    ax.axhline(y=4.5, linestyle='--', lw=3, color='red', alpha=0.5,
        path_effects=[pe.Stroke(linewidth=5, foreground='#595959'), pe.Normal()])

    # Plot settings
    ax.tick_params(axis='y', labelsize=18, direction='in', width=2, \
                    length=5, pad=10)
    ax.tick_params(axis='x', labelsize=18, direction='in', width=2, \
                    length=5, pad=10)
    xticks = np.arange(0,len(time),1000)
    ax.set_xticks(xticks)
    ax.set_xticklabels(list(map(lambda x : str(x/1000).split(".")[0], xticks)))
    ax.set_xlabel(r"Time ($\mu s$)", labelpad=5, fontsize=24)
    for i in ["top","bottom","left","right"]:
        ax.spines[i].set_linewidth(2)
    ax.grid(True)
    ax.set_ylabel("Distance ($\AA$)", labelpad=5, fontsize=24)
    plt.ylim([0, 30])
    _, xmax = ax.get_xlim()
    ax.set_xlim(0,xmax)
    plt.legend(fontsize=24, ncol=2)

    plt.savefig(f"{ path }/hpairs.pdf", dpi=300)
    plt.close()

    return None

if __name__ == '__main__':
    main(sys.argv)
