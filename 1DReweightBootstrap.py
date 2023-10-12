import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import plumed
import wham
import pandas as pd
import MDAnalysis as mda
import argparse
from MDAnalysis.analysis import align
import subprocess
import glob

def main(argv):

    try:
        parser = argparse.ArgumentParser()

        parser.add_argument("-p", "--path",
                            action = "store",
                            dest = "path",
                            default = None,
                            help = """Set path to the data directory.""")
        parser.add_argument("-v", "--collective-variable",
                            action = "store",
                            dest = "cv",
                            default = "closed",
                            help = """Collective variable used for biasing, i.e. "open" or "biased".""")
        parser.add_argument("-f", "--figpath",
                            action = "store",
                            dest = "fig_path",
                            default = None,
                            help = """Set a path destination for the figure.""")  
        parser.add_argument("-b", "--blocks",
                            action = "store",
                            dest = "nblocks",
                            default = 50,
                            help = """Number of blocks to use in bootstrap analysis.""")            
        parser.add_argument("-r", "--recalc",
                            action = "store_true",
                            dest = "recalc",
                            default = False,
                            help = """Chose whether the reweighting should be recomputed.""")
        args = parser.parse_args()

    except argparse.ArgumentError:
        print("Command line arguments are ill-defined, please check the arguments")
        raise

    global path_head, data_path, nb, cv, nw, kBT

    # Set path variables etc.
    path_head = "/home/lf1071fu/project_b3"
    data_path = args.path
    fig_path = args.fig_path
    recalc = args.recalc
    nb = int(args.nblocks) # number of blocks for dividing data into blocks for bootstrapping
    cv = args.cv # descriptor for collective variable used for biasing
    nw = 20 # number of windows (number of simulations) 
    bs = 100 # number of bootstraps
    kBT = 310 * 8.314462618 * 0.001 # use kJ/mol here
    if not data_path:
        data_path = os.getcwd()
    if not fig_path:
        fig_path = data_path
    
    # Reference quantities (non-essential)
    vec_open, vec_closed = get_ref_vecs() # reference values of the collective variables
    restraint_pts = get_restraint_loc(nw, vec_open, vec_closed, cv) # target values for window biases 

    np_ave_files = [f"{ data_path }/open_ave.npy", f"{ data_path }/open_sq_ave.npy", f"{ data_path }/closed_ave.npy",
                    f"{ data_path }/closed_sq_ave.npy"]
    bs_path = f"{ data_path }/bootstrap_files"
    if not os.path.exists(bs_path):
        os.makedirs(bs_path)

    if not all(list(map(lambda x : os.path.exists(x), np_ave_files))) or recalc: 

        # Get the biasing data from COLVAR files
        time, odot, cdot, bias, fpw = get_colvar_data()

        # Reshape the bias into blocks that can be used for bootstrapping
        # shape: (blocks x frames per block x windows)
        bb = bias.reshape((nb, fpw // nb, nw))
        time, odot, cdot = [n.reshape((nb, fpw // nb)) for n in [time, odot, cdot]]

        # Initialize arrays for the histogram data
        o_ave, o_ave_sq = np.zeros(601), np.zeros(601)
        c_ave, c_ave_sq = np.zeros(601), np.zeros(601)

        # Construct bootstraps and determine reweighted histograms
        for i in range(bs):
            print(i)

            # Choose random blocks for bootstap
            c=np.random.choice(nb, nb)

            # Get the log weights for reweighting with WHAM 
            w=wham.wham(bb[c,:,:].reshape((-1, nw)), T=kBT)

            # Write the logweights to a pandas table for reweighting
            colvar_weights = pd.DataFrame(data={"time" : time[c,:].flatten(), 
                                "openvec" : odot[c,:].flatten(), 
                                "closedvec" : cdot[c,:].flatten(), 
                                "logweights" : w["logW"]})

            # Use plumed to make rweighted histograms
            make_ith_histogram(colvar_weights, bs_path, i)

            # Load in histogram data
            hist_open = plumed.read_as_pandas(f"{ bs_path }/rhist_odot_{ i }.dat").replace([np.inf, 
                        -np.inf], np.nan).dropna()
            hist_closed = plumed.read_as_pandas(f"{ bs_path }/rhist_cdot_{ i }.dat").replace([np.inf, 
                        -np.inf], np.nan).dropna()

            # Use the ith histogram to estimate average probability densities
            o_ave += hist_open.hhodotr
            o_ave_sq += hist_open.hhodotr*hist_open.hhodotr
            c_ave += hist_closed.hhcdotr
            c_ave_sq += hist_closed.hhcdotr*hist_closed.hhcdotr

        for file, arr in zip(np_ave_files, [o_ave, o_ave_sq, c_ave, c_ave_sq]):
            np.save(file, arr)
    
    # Load calculated values from pandas tables and numpy arrays
    else:

        hist_open = plumed.read_as_pandas(f"{ bs_path }/"
                        "rhist_odot_1.dat").replace([np.inf, -np.inf], 
                        np.nan).dropna()
        hist_closed = plumed.read_as_pandas(f"{ bs_path }/"
                        "rhist_cdot_1.dat").replace([np.inf, -np.inf], 
                        np.nan).dropna()

        o_ave = np.load(np_ave_files[0], allow_pickle=True)
        o_ave_sq = np.load(np_ave_files[1], allow_pickle=True)
        c_ave = np.load(np_ave_files[2], allow_pickle=True)
        c_ave_sq = np.load(np_ave_files[3], allow_pickle=True)

    # Calculate free energy surface from histogram averages and 
    # free energy errors from the histograms, see expressions in ex-5 
    # https://www.plumed.org/doc-v2.8/user-doc/html/masterclass-21-2.html
    average_c, average_o = c_ave / bs, o_ave / bs
    var_c = (1 / (bs - 1)) * ( c_ave_sq / bs - average_c * average_c ) 
    var_o = (1 / (bs - 1)) * ( o_ave_sq / bs - average_o * average_o ) 
    fes_c, fes_o = - kBT * np.log(average_c), - kBT * np.log(average_o)
    error_c, error_o = np.sqrt( var_c ), np.sqrt( var_o )
    ferr_c, ferr_o = kBT * np.log(error_c / average_c), kBT * np.log(np.abs(error_o / average_o))

    # Make FES plots with error bars
    plot_fes(hist_open.odot, fes_o, ferr_o, fig_path, "open")
    plot_fes(hist_closed.cdot, fes_c, ferr_c, fig_path, "closed")

    return None

def get_restraint_loc(num_us, vec_open, vec_closed, conform="closed"):
    """Determines restraint positions for the umbrella windows.

    Parameters
    ----------
    num_us : int
        Number of umbrella windows.
    vec_open : np.ndarray
        The beta vector for the reference open state in Angstrom.
    vec_closed : np.ndarray
        The beta vector for the reference closed state in Angstrom.
    conform : str
        Descriptor for the reference conformation used in restraints,
        i.e. "open" or "closed".
    
    Returns
    -------
    restraint_pts : np.ndarray
        Positions of the restraints for the beta vec dot products.

    """

    def three_point_function(p1, p2, p3):
        """Determines the coefficients of a 2 deg polynomial, passing 
        through 3 points.
        """
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        A = np.array([[x1**2, x1, 1], [x2**2, x2, 1], [x3**2, x3, 1]])
        b = np.array([y1, y2, y3])
        coeffs = np.linalg.solve(A, b)
        return coeffs

    p1 = (np.dot(vec_closed, vec_open), np.dot(vec_closed, vec_closed))
    p2 = (3.5, 3.5)
    p3 = (np.dot(vec_open, vec_open), np.dot(vec_open, vec_closed))

    f = three_point_function(p1, p2, p3)

    if conform == "open":

        restraint_pts = np.zeros((num_us,2))
        restraint_pts[:,0] = np.linspace(p1[0],p3[0],20)
        restraint_pts[:,1] = [f[0]*x**2 + f[1]*x + f[2] for x in restraint_pts[:,0]] 
    else:

        restraint_pts = np.zeros((num_us-1,2))
        restraint_pts[:,1] = np.linspace(p1[1], p3[1], num_us-1)
        restraint_pts[:,0] = [np.roots([f[0],f[1],f[2] - y])[0] for y in restraint_pts[:,1]]
        restraint_pts = np.vstack([np.array(p1), restraint_pts])

    return restraint_pts

def get_ref_vecs():
    """Determines the beta vectors for the reference state.

    Structures are aligned with the same reference structure used in the biasing
    during simulation.

    Parameters
    ----------
    None.

    Returns
    -------
    vec_open : np.ndarray
        The beta vector for the reference open state in Angstrom.
    vec_closed : np.ndarray
        The beta vector for the reference closed state in Angstrom.

    """
    path_head = "/home/lf1071fu/project_b3"
    struct_path = f"{ path_head }/structures"
    beta_vec_path =f"{ path_head }/simulate/umbrella_sampling/beta_vec_open/nobackup"

    # Load in relevant reference structures
    # NB: PDB uses units of nm
    open_state = mda.Universe(f"{ struct_path }/open_ref_state.pdb", length_unit="nm")
    closed_state = mda.Universe(f"{ struct_path }/closed_ref_state.pdb", length_unit="nm")
    ref_state = mda.Universe(f"{ struct_path }/alignment_ref.pdb", length_unit="nm")
    open_data = f"{ path_head }/simulate/open_conf/data"

    # Load in universe objects for the simulation and the reference structures
    top = f"{ open_data }/topol.top"

    core_res, core = get_core_res()

    # Align the traj and ref states to one structure
    align.AlignTraj(open_state, ref_state, select=core, in_memory=True).run()
    align.AlignTraj(closed_state, ref_state, select=core, in_memory=True).run()

    r1, r2 = 206, 215
    r1_open = open_state.select_atoms(f"name CA and resnum { r1 }").positions[0]
    r2_open = open_state.select_atoms(f"name CA and resnum { r2 }").positions[0]
    vec_open = r2_open/10 - r1_open/10
    # mag_open = np.linalg.norm(vec_open)
    print(vec_open)
    r1_closed = closed_state.select_atoms(f"name CA and resnum { r1 }").positions[0]
    r2_closed = closed_state.select_atoms(f"name CA and resnum { r2 }").positions[0]
    vec_closed = r2_closed/10 - r1_closed/10
    # mag_closed = np.linalg.norm(vec_closed)
    print(vec_closed)

    return vec_open, vec_closed

def get_core_res(recalc=False):
    """Finds the core residues which are immobile across the conformational states.

    Uses data from the combined simulation of the closed states open and closed simulations,
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
    core_res_path = f"{ path_head }/structures"
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

def get_colvar_data():
    """Get relevant data for the collective variables and restraint quantities.

    Returns
    -------
    time : np.ndarray
        The time of the concatenated trajectory in picoseconds.
    odot : np.ndarray
        The concatenated trajectory of the open vector dot product, dim = (nframes).
    cdot : np.ndarray
        The concatenated trajectory of the closed vector dot product, dim = (nframes).
    bias : np.ndarray
        The bias applied to the full, concatenated trajectory for each window, 
        dim = (n-frames, n-windows).
    fpw : int
        The number of frames per simulation window.
    """
    if cv == "open":
        columns = ["time", "openvec", "closedvec", "opentheta", "closedtheta", "bias", "force"]
    elif cv == "closed":
        columns = ["time", "openvec", "closedvec", "bias", "force"]
    else:
        print("""ERROR: Invalid name for the collective variable descriptor. (Try "open" or "closed").""")
        sys.exit(1)
    col = []
    for i in range(nw):
        col.append(pd.read_csv(f"{ data_path }/COLVAR_" + str(i+1)+".dat", delim_whitespace=True, 
                    comment='#', names=columns))

    fpw = len(col[0]["bias"]) - 1 # frames per window
    bias = np.zeros((fpw,nw))
    for i in range(nw):
        bias[:,i] = col[i]["bias"].iloc[1:] # [-len(bias):] 
    time = np.array(col[0]["time"].iloc[1:])
    odot = np.array(col[0]["openvec"].iloc[1:])
    cdot = np.array(col[0]["closedvec"].iloc[1:])

    return time, odot, cdot, bias, fpw

def make_ith_histogram(colvar_weights, bs_path, i):
    """Use plumed HISTOGRAM function to obtain reweighted histograms for the bootstrap.

    Parameters
    ----------
    colvar_weights : pd.DataFrame
        Dataframe containing the collective variables and logweights.
    bs_path : str
        Path to directory for bootstraping data.
    i : int
        Integer for the bootstrap.

    Returns
    -------
    None.

    """
    plumed.write_pandas(colvar_weights, f"{ bs_path }/colvar_weights_{ i }.dat")

    with open(f"{ bs_path }/colvar_histograms_{ i }.dat","w") as f:
        print(f"""
    # vim:ft=plumed
    odot: READ FILE={ bs_path }/colvar_weights_{ i }.dat VALUES=openvec IGNORE_TIME
    cdot: READ FILE={ bs_path }/colvar_weights_{ i }.dat VALUES=closedvec IGNORE_TIME
    lw: READ FILE={ bs_path }/colvar_weights_{ i }.dat VALUES=logweights IGNORE_TIME

    hhodotr: HISTOGRAM ARG=odot GRID_MIN=0 GRID_MAX=6 GRID_BIN=600 BANDWIDTH=0.05 NORMALIZATION=ndata LOGWEIGHTS=lw
    DUMPGRID GRID=hhodotr FILE={ bs_path }/rhist_odot_{ i }.dat

    hhcdotr: HISTOGRAM ARG=cdot GRID_MIN=0 GRID_MAX=6 GRID_BIN=600 BANDWIDTH=0.05 NORMALIZATION=ndata LOGWEIGHTS=lw
    DUMPGRID GRID=hhcdotr FILE={ bs_path }/rhist_cdot_{ i }.dat

    """, file=f)

    subprocess.run(("plumed driver --noatoms --plumed {}/colvar_histograms_{}.dat --kt {}").format(bs_path, i, kBT), shell=True)

    return None

def plot_fes(cv_bins, fes, ferr, fig_path, coord):
    """Plot the free energy surface for the CV with error bars.

    Parameters
    ----------
    cv_bins : np.ndarray
        The collective variable binning positions from the histograms.
    fes : np.ndarray
        The free energy of the collective variable in kJ / mol.
    ferr : np.ndarray
        The error in the free energy in kJ / mol.
    fig_path : str
        The path for storing in figure image.
    coord : str
        The reaction coordinate for the free energy surface.

    Returns
    -------
    None.

    """
    # Make main plot and error bar plot.
    fig, ax = plt.subplots(constrained_layout=True, figsize=(12,8))
    plt.plot(cv_bins, fes)
    plt.fill_between(cv_bins, fes - ferr, fes + ferr, alpha=0.5)

    # Plot labels
    if coord == "closed":
        ax.set_xlabel(r"$\vec{\upsilon} \cdot \vec{\upsilon}_{closed}$ (nm$^2$)", labelpad=5, fontsize=24)
        ax.set_ylabel(r"$F(\vec{\upsilon} \cdot \vec{\upsilon}_{closed})$ (kJ / mol)", labelpad=5, fontsize=24)
    elif coord == "open":
        ax.set_xlabel(r"$\vec{\upsilon} \cdot \vec{\upsilon}_{open}$ (nm$^2$)", labelpad=5, fontsize=24)
        ax.set_ylabel(r"$F(\vec{\upsilon} \cdot \vec{\upsilon}_{open})$ (kJ / mol)", labelpad=5, fontsize=24)

    plt.savefig(f"{ fig_path }/fes_{ coord }_nb{ nb }.png", dpi=300)
    plt.show()

    plt.close()

if __name__ == '__main__':
    main(sys.argv)
