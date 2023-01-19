import sys
import numpy as np
import os
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis import rms, align, dihedrals, contacts, distances, \
                                hydrogenbonds
import pandas as pd

def get_rmsd(system, reference, alignment, rmsd_group):
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
    rmsd_group : MDAnalysis.core.groups.AtomGroup
        The group for the RMSD, e.g. the beta flap residues.

    Returns
    -------
    rmsd_arr : np.ndarray
        A timeseries of the RMSD against the given reference, for the given
        atom group.
    """
    R = rms.RMSD(system,
                 reference,  # reference universe or atomgroup
                 select=alignment,  # group to superimpose and calculate RMSD
                 groupselections=[rmsd_group])  # groups for RMSD
    R.run()

    rmsd_arr = R.results.rmsd

    return rmsd_arr

def get_rmsf(u, top, path):
    """Determines the RMSF per residue.

    The average structure is calculated and then used as the reference for
    structure alignment, before calculating the rmsf per residue.

    Parameters
    ----------
    u : MDAnalysis.core.universe
        The universe object, with the trajectory loaded in.
    top : str
        The topology file, e.g. xxx.gro.
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
    aligner = align.AlignTraj(u, ref, select="protein and name CA",
                      in_memory=True).run()
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

def plot_rmsd(rmsd, path, group="selection", ref="reference structure"):
    """Makes a plot of the rmsd over the trajectory.

    Parameters
    ----------
    rmsd : MDAnalysis.analysis.rms.RMSD or nparray
        A time series of the rmsd against a known reference structure.
    path : str
        Path to the primary working directory.
    group : str

    Returns
    -------
    None.

    """
    fig, ax = plt.subplots(constrained_layout=True, figsize=(12,8))

    ax.plot(rmsd[:,1]/1e6, rmsd[:,2], "k-", label="all")
    ax.plot(rmsd[:,1]/1e6, rmsd[:,3], ls=":", color="r", label=group)

    # Plot settings
    ax.tick_params(axis='y', labelsize=18, direction='in', width=2, \
                    length=5, pad=10)
    ax.tick_params(axis='x', labelsize=18, direction='in', width=2, \
                    length=5, pad=10)
    for i in ["top","bottom","left","right"]:
        ax.spines[i].set_linewidth(2)
    ax.grid(True)
    ax.set_xlabel(r"time ($\mu$s)", labelpad=5, \
                    fontsize=28)
    ax.set_ylabel(r"RMSD to the {} ($\AA$)".format(ref),
                  labelpad=5, fontsize=28)
    plt.legend(fontsize=18)

    if ref == "reference structure":
        ref = "ref"
    else:
        ref = ref.replace(" ", "_")
    plt.savefig(f"{ path }/rmsd_to_{ ref }.png")
    plt.close()

    return None

def plot_rmsds(r_holo, r_apo, path):
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
                cmap="cividis")

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
    ax.set_xlabel(r"RMSD to bound state ($\AA$)", labelpad=5, \
                    fontsize=28)
    ax.set_ylabel(r"RMSD to apo state ($\AA$)", labelpad=5, fontsize=28)

    plt.savefig(f"{path}/../figures/rmsd_beta-flap.png")
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
    resids = list(map(lambda x : x + 542, calphas))
    plt.plot(resids, rmsf, lw=3, color="#b21856")

    # Plot settings
    ax.tick_params(axis='y', labelsize=18, direction='in', width=2, \
                    length=5, pad=10)
    ax.tick_params(axis='x', labelsize=18, direction='in', width=2, \
                    length=5, pad=10)
    for i in ["top","bottom","left","right"]:
        ax.spines[i].set_linewidth(2)
    ax.grid(True)
    ax.set_xlabel(r"Residue number", labelpad=5, \
                    fontsize=28)
    ax.set_ylabel(r"RMSF ($\AA$)", labelpad=5, fontsize=28)
    bottom, top = ax.get_ylim()
    ax.vlines([743,776], bottom, top, linestyles="dashed", alpha=0.6, lw=3,
              colors="r")

    plt.savefig(f"{ path }/../figures/rmsf.png")
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
    plot_path = f"{ path }/../figures/ramachandran"
    if not os.path.exists(plot_path):
       os.makedirs(plot_path)
    plt.savefig(f"{ plot_path }/res_{ res.resid }.png")
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

    plot_path = f"{ path }/../figures/chi_1s"
    if not os.path.exists(plot_path):
       os.makedirs(plot_path)
    plt.savefig(f"{ plot_path }/res_{ res.resid }.png")
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

    plt.savefig(f"{ path }/../figures/salt_contacts.png")
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

    plt.savefig(f"{ path }/../figures/hbonds_count.png")
    plt.close()

    return None
