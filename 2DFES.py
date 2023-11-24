import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import plumed
import wham
import MDAnalysis as mda
from MDAnalysis.analysis import align
import subprocess    
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.tri import Triangulation
from scipy.signal import find_peaks
import argparse
import pandas as pd

def main(argv):

    try:
        parser = argparse.ArgumentParser()

        parser.add_argument("-p", "--path",
                            action = "store",
                            dest = "path",
                            default = ("/home/lf1071fu/project_b3/simulate/"
                                        "umbrella_sampling/2d_fes/nobackup"),
                            help = """Set path to the data directory.""")        
        parser.add_argument("-a", "--angle",
                            action = "store_true",
                            dest = "angle_coord",
                            default = False,
                            help = """Chose whether the reaction coordinate should be converted into an angle.""")
        parser.add_argument("-f", "--figpath",
                            action = "store",
                            dest = "fig_path",
                            default = ("/home/lf1071fu/project_b3/figures/umbrella"
                                        "/2dfes_apo"),
                            help = """Set a path destination for the figure.""")  
        parser.add_argument("-b", "--blocks",
                            action = "store",
                            dest = "nblocks",
                            default = 100,
                            help = """Number of blocks to use in bootstrap analysis.""")            
        parser.add_argument("-r", "--recalc",
                            action = "store_true",
                            dest = "recalc",
                            default = False,
                            help = """Chose whether the reweighting should be recomputed.""")
        parser.add_argument("-n", "--nobootstrap",
                            action = "store_true",
                            dest = "no_bootstrap",
                            default = False,
                            help = """Compute the free energy surface without error analysis.""")

        args = parser.parse_args()

    except argparse.ArgumentError:
        print("Command line arguments are ill-defined, please check the arguments")
        raise

    global home, path_head, nb, cv, nw, kBT


    # Path variables
    home = args.path
    fig_path = args.fig_path
    path_head = "/home/lf1071fu/project_b3"
    struct_path = f"{ path_head }/structures"

    # Other variables
    angle_coord = args.angle_coord
    recalc = args.recalc
    no_bs = args.no_bootstrap
    nb = int(args.nblocks) # number of blocks for dividing data into blocks for bootstrapping
    nw = 181 # number of windows (number of simulations) 
    bs = 10 #200 # number of bootstraps
    kBT = 310 * 8.314462618 * 0.001 # use kJ/mol here

    # Helpful objects for structure alignment
    core_res, core = get_core_res(path_head)
    ref_state = mda.Universe(f"{ struct_path }/alignment_ref.pdb", length_unit="nm")

    # plot_1dfes("fes_catr.dat", "open", fig_path)
    # plot_1dfes("fes_catr.dat", "closed", fig_path)

    vec_open, vec_closed = get_ref_vecs(struct_path, core, ref_state)

    np_ave_files = [f"{ home }/hist_ave.npy", f"{ home }/hist_sq_ave.npy", f"{ home }/bin_counter.npy"]
    bs_path = f"{ home }/bootstrap_files"
    if not os.path.exists(bs_path):
        os.makedirs(bs_path)

    if no_bs:

        fes = get_no_bs_fes(bs_path)

        # Make the 2D FES plot
        plot_2dfes(fes.odot, fes.cdot, fes.ffr, None)

    elif not all(list(map(lambda x : os.path.exists(x), np_ave_files))) or recalc: 

        # # Get the biasing data from COLVAR files
        time, odot, cdot, bias, fpw = get_colvar_data()

        print(bias)
        print(f"SIZE OF BIAS { np.shape(bias) }")

        # Reshape the bias into blocks that can be used for bootstrapping
        bb = bias.reshape((nw, nb, fpw // nb))
        time, odot, cdot = [n.reshape((nb, fpw // nb)) for n in [time, odot, cdot]]

        # Initialize arrays for the histogram data
        ave, ave_sq, count = np.zeros(151**2), np.zeros(151**2), np.zeros(151**2)

        # # Construct bootstraps and determine reweighted histograms
        for i in range(bs):
            print(i)

            # Choose random blocks for bootstap
            c=np.random.choice(nb, nb)

            pfes_file = f"{ bs_path }/fes_catr_{ i }.dat"
            hist_file = f"{ bs_path }/hist_catr_{ i }.dat"

            if not os.path.exists(pfes_file) or recalc:

                # Get the log weights for reweighting with WHAM 
                w = wham.wham(bb[:,c,:].reshape((-1, nw)), T=kBT)

                # Write the logweights to a pandas table for reweighting
                colvar_weights = pd.DataFrame(data={"time" : time[c,:].flatten(), 
                                    "openvec" : odot[c,:].flatten(), 
                                    "closedvec" : cdot[c,:].flatten(), 
                                    "logweights" : w["logW"]})

                # Use plumed to make reweighted histograms
                make_ith_histogram(colvar_weights, bs_path, i)

            # Load in free energy estimates determined via plumed
            pfes = plumed.read_as_pandas(pfes_file).replace([np.inf, -np.inf], np.nan)

            # Use the ith histogram to estimate average probability densities
            # mask = ~np.isnan(pfes.ffr) # some bins will be NAN
            # ave = np.nansum([ave, pfes.ffr], axis=0) 
            # ave_sq = np.nansum([ave_sq, pfes.ffr*pfes.ffr], axis=0)
            # count[mask] += 1 # only bins with a bootstrap estimate should have the count increased

            ffr = pfes.ffr.astype(float)
            ave += ffr
            ave_sq += ffr*ffr

        for file, arr in zip(np_ave_files, [ave, ave_sq, count]):
            np.save(file, arr)

    # Load calculated values from pandas tables and numpy arrays
    else:

        # Get bin positions for the CVs
        pfes = plumed.read_as_pandas(f"{ bs_path }/"
                        "fes_catr_0.dat").replace([np.inf, -np.inf], np.nan) #.dropna()

        # Load the sum of the free energy esimates from each bin
        ave = np.load(np_ave_files[0], allow_pickle=True)
        ave_sq = np.load(np_ave_files[1], allow_pickle=True)
        count = np.load(np_ave_files[2], allow_pickle=True)

    # Calculate free energy surface from histogram averages and 
    # free energy errors from the histograms, see expressions in ex-5 
    # https://www.plumed.org/doc-v2.8/user-doc/html/masterclass-21-2.html
    ave = ave / bs
    # fes = convert_fes(hist.odot, hist.cdot, ave)

    var = (1 / (count - 1)) * ( ave_sq / count - ave * ave ) 
    fes = - kBT * np.log(ave)
    error = np.sqrt( var )
    ferr = error / ave

    # Make the 2D FES plot
    plot_2dfes(pfes.odot, pfes.cdot, fes, ferr)

    # Make the error estimate plot for the 2D FES

    return None

def plot_2dfes(open_bins, closed_bins, fes, ferr):
    """Plots the 2D FES as a colorbar + contour map.

    open_bins : np.ndarray
        Bin positions for the beta vec dot product with the open ref.
    closed_bins : np.ndarray
        Bin positions for the beta vec dot product with the closed ref.
    fes_o : np.ndarray
        Free energies at the grid positions.
    ferr : np.ndarray
        Errors in the free energies at the grid positions. 
    """
    fig, ax = plt.subplots(constrained_layout=True, figsize=(12,8))

    mask = ((-1e2 < fes) & (fes < 1e2))

    x, y = open_bins[mask], closed_bins[mask]
    z = fes[mask]
    # z, err = fes[mask], ferr[mask]

    d = ax.scatter(x, y, c=z, cmap=plt.cm.viridis)

    # if angle_coord:
    #     fes_r=plumed.read_as_pandas("fes_theta_catr.dat").replace([np.inf, -np.inf], np.nan).dropna()
    #     x = [ x for x, z in zip(fes_r.theta1, fes_r.ffthetar) if -80 < z < 80 ]
    #     y = [ y for y, z in zip(fes_r.theta2, fes_r.ffthetar) if -80 < z < 80 ]
    #     zmin_g = np.min(np.array([f for f in fes_r.ffthetar if f > -80]))
    #     zmin_l = np.min(np.array([f for x, f in zip(fes_r.theta1, fes_r.ffthetar) if ((f > -80) and (x < 0.75))]))
    #     z = [ z - zmin_g for z in fes_r.ffthetar if -80 < z < 80 ]

    # else:
    # fes_r=plumed.read_as_pandas("fes_catr.dat").replace([np.inf, -np.inf], np.nan).dropna()
    # x = [ x for x, z in zip(fes_r.opendot, fes_r.ffr) if -80 < z < 80 ]
    # y = [ y for y, z in zip(fes_r.closeddot, fes_r.ffr) if -80 < z < 80 ]
    # zmin_g = np.min(np.array([f for f in fes_r.ffr if f > -80]))
    # zmin_l = np.min(np.array([f for x, f in zip(fes_r.opendot, fes_r.ffr) if ((f > -80) and (x > 4))]))
    # z = [ z - zmin_g for z in fes_r.ffr if -80 < z < 80 ]
    # print(f"Global min: { zmin_g } kJ/mol\tLocal min: { zmin_l } kJ/mol")
    plt.show()
    sys.exit(1)
    
    tri = Triangulation(x, y)

    # Create contour lines on the XY plane using tricontour
    contours = ax.tricontour(tri, z, cmap="inferno")

    # Colormap settings
    cbar = plt.colorbar(d)
    # if angle_coord:
    #     cbar.set_label(r'$F(\theta_{open}, \theta_{closed})$ (kJ / mol)', fontsize=24, labelpad=10)
    # else: 
    cbar.set_label(r'$F(\vec{\upsilon} \cdot \vec{\upsilon}_{open}, \vec{\upsilon} \cdot \vec{\upsilon}_{closed})$ (kJ / mol)', fontsize=28, labelpad=10)
    cbar.ax.tick_params(labelsize=18, direction='out', width=2, length=5)
    cbar.outline.set_linewidth(2)
    #ax.clabel(contours, inline=1, fontsize=20)


    # Find minima
    # minima_indices, _ = find_peaks(-np.array(z), prominence=70)
    # mins = np.array(minima_indices)
    # xmins = [x[i] for i in mins]
    # ymins = [y[i] for i in mins]

    # Add both reference positions
    if angle_coord:
        ax.scatter(calc_theta(vec_open, vec_open), calc_theta(vec_open, vec_closed), 
                label="Open ref.", marker="X", alpha=1, edgecolors="#404040", 
                s=550, lw=3, color="#EAAFCC")
        ax.scatter(calc_theta(vec_open, vec_closed), 0, label="Closed ref.", 
                marker="X", alpha=1, edgecolors="#404040", s=550, lw=3, color="#A1DEA1")
    else:
        ax.scatter(np.dot(vec_open, vec_open), np.dot(vec_open, vec_closed), 
                label="Open ref.", marker="X", alpha=1, edgecolors="#404040", 
                s=550, lw=3, color="#EAAFCC")
        ax.scatter(np.dot(vec_open, vec_closed), np.dot(vec_closed, vec_closed), 
                label="Closed ref.", marker="X", alpha=1, edgecolors="#404040", 
                s=550, lw=3, color="#A1DEA1")

    ax.tick_params(axis='y', labelsize=18, direction='in', width=2, \
                    length=5, pad=10)
    ax.tick_params(axis='x', labelsize=18, direction='in', width=2, \
                    length=5, pad=10)
    for i in ["top","bottom","left","right"]:
        ax.spines[i].set_linewidth(2)
    if angle_coord:
        ax.set_xlabel(r"$\theta_{open}$ (rad)", labelpad=5, fontsize=24)
        ax.set_ylabel(r"$\theta_{closed}$ (rad)", labelpad=5, fontsize=24)
    else: 
        ax.set_xlabel(r"$\vec{\upsilon} \cdot \vec{\upsilon}_{open}$ (nm$^2$)", labelpad=5, fontsize=24)
        ax.set_ylabel(r"$\vec{\upsilon} \cdot \vec{\upsilon}_{closed}$ (nm$^2$)", labelpad=5, fontsize=24)
    _, xmax = ax.get_xlim()
    _, ymax = ax.get_ylim()
    ax.set_xlim(0,xmax)
    ax.set_ylim(0,ymax)
    plt.legend(fontsize=24)

    if angle_coord:
        plt.savefig(f"{ fig_path }/2dfes_angles.png", dpi=300)
    else:
        plt.savefig(f"{ fig_path }/2dfes.png", dpi=300)

def get_core_res(path_head):
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
    core_res_path = f"{ path_head }/structures"
    core_res = np.load(f"{ core_res_path }/core_res.npy")

    aln_str = "protein and name CA and ("
    core_open = [f"resid {i} or " for i in core_res]
    core_closed = [f"resid {i + 544} or " for i in core_res]
    core = aln_str + "".join((core_open + core_closed))[:-4] + ")"

    return core_res, core

def get_ref_vecs(struct_path, core, ref_state):
    """Calculates the beta-vectors for the reference structures.

    Parameters
    ----------
    struct_path : str
        Path to the structure directory, i.e. project_b3/structures
    core : str 
        The core residues used for alignment throughout this project.
    ref_state: MDAnalysis.Universe
        The reference structure object.

    Returns
    -------
    vec_open : np.ndarray
        The beta vector in Angstrom for the aligned open reference state.
    vec_closed : np.ndarray
        The beta vector in Angstrom for the aligned closed reference state.

    """
    # Load in relevant reference structures
    open_ref = mda.Universe(f"{ struct_path }/open_ref_state.pdb", length_unit="nm")
    closed_ref = mda.Universe(f"{ struct_path }/closed_ref_state.pdb", length_unit="nm")

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

    return vec_open, vec_closed

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

def plot_1dfes(dat_path, conform, fig_path):
    # Plot FES wrt the open state dot product
    # fes_hdot=plumed.read_as_pandas(dat_path).replace([np.inf, -np.inf], np.nan).dropna()
    # plt.plot(fes_hdot.hdot,fes_hdot.ffhdot,label="original")
    fig, ax = plt.subplots(constrained_layout=True, figsize=(12,8))
    # fes_opendotb=plumed.read_as_pandas("fes_opendot_cat.dat").replace([np.inf, -np.inf], np.nan).dropna()
    # ax.plot(fes_opendotb.opendot,fes_opendotb.ffopendot,label="biased")
    fes = plumed.read_as_pandas(dat_path).replace([np.inf, -np.inf], np.nan).dropna()
    if conform == "open":
        ax.plot(fes.opendot,fes.ffopendotr,label="reweighted")
        ax.set_xlabel(r"$\vec{\upsilon} \cdot \vec{\upsilon}_{open}$ (nm$^2$)", labelpad=5, fontsize=24)
        ax.set_ylabel(r"$F(\vec{\upsilon} \cdot \vec{\upsilon}_{open})$ (kJ / mol)", labelpad=5, fontsize=24)
    elif conform == "closed":
        ax.plot(fes.closeddot,fes.ffcloseddotr,label="reweighted")
        ax.set_xlabel(r"$\vec{\upsilon} \cdot \vec{\upsilon}_{closed}$ (nm$^2$)", labelpad=5, fontsize=24)
        ax.set_ylabel(r"$F(\vec{\upsilon} \cdot \vec{\upsilon}_{closed})$ (kJ / mol)", labelpad=5, fontsize=24)

    # Plot settings
    ax.tick_params(axis='y', labelsize=18, direction='in', width=2, \
                    length=5, pad=10)
    ax.tick_params(axis='x', labelsize=18, direction='in', width=2, \
                    length=5, pad=10)
    for i in ["top","bottom","left","right"]:
        ax.spines[i].set_linewidth(2)
    ax.grid(True)
    plt.legend(fontsize=18)
    _, xmax = ax.get_xlim()
    _, ymax = ax.get_ylim()
    ax.set_xlim(0,xmax)
    ax.set_ylim(0,ymax)

    plt.savefig(f"{ fig_path }/fes_{ conform }.png", dpi=300)
    plt.close()

    return None

def get_colvar_data(recalc=False):
    """Get relevant data for the collective variables and restraint quantities.

    Parameters
    ----------
    recalc : bool
        Redetermine the collective variable from plumed files rather than loading
        arrays from as numpy files.

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
    cv_bias_files = [f"{ home }/timeseries.npy", f"{ home }/opendot_concat.npy", 
                     f"{ home }/closeddot_concat.npy", f"{ home }/bias_concat.npy"]

    if not all(list(map(lambda x : os.path.exists(x), cv_bias_files))) or recalc: 

        columns = ["time", "opendot", "closeddot", "theta1", "theta2", "bias", "force"]
        col = []
        for i in range(nw):
            col.append(pd.read_csv(f"{ home }/COLVAR_" + str(i+1)+".dat", delim_whitespace=True, 
                        comment='#', names=columns))

        fpw = len(col[0]["bias"]) - 1 # frames per window
        bias = np.zeros((fpw,nw))
        for i in range(nw):
            bias[:,i] = col[i]["bias"].iloc[1:] # [-len(bias):] 
        time = np.array(col[0]["time"].iloc[1:])
        odot = np.array(col[0]["opendot"].iloc[1:])
        cdot = np.array(col[0]["closeddot"].iloc[1:])

        for file, arr in zip(cv_bias_files, [time, odot, cdot, bias]):
            np.save(file, arr)

    else:

        time = np.load(cv_bias_files[0], allow_pickle=True)
        odot = np.load(cv_bias_files[1], allow_pickle=True)
        cdot = np.load(cv_bias_files[2], allow_pickle=True)
        bias = np.load(cv_bias_files[3], allow_pickle=True)
        fpw = len(odot)

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

    hhr: HISTOGRAM ARG=odot,cdot GRID_MIN=0,0 GRID_MAX=6,6 GRID_BIN=150,150 BANDWIDTH=0.05,0.05 LOGWEIGHTS=lw
    DUMPGRID GRID=hhr FILE={ bs_path }/hist_catr_{ i }.dat
    ffr: CONVERT_TO_FES GRID=hhr 
    DUMPGRID GRID=ffr FILE={ bs_path }/fes_catr_{ i }.dat

    """, file=f)

    subprocess.run(("plumed driver --noatoms --plumed {}/colvar_histograms_{}.dat --kt {}").format(bs_path, i, kBT), shell=True)

    return None

def get_no_bs_fes(bs_path):
    """Use all the data to estimate the free energy surface, without bootstrapping.

    Parameters
    ----------
    bs_path : str
        Path to bootstrapping calculations directory.

    Returns
    -------
    fes : pd.DataFrame
        Pandas table with collective variable data and free energy estimates in kJ/mol.

    """
    fes_dat = f"{ bs_path }/fes_catr_fulldata.dat"

    if not os.path.exists(fes_dat):

        # # Get the biasing data from COLVAR files
        time, odot, cdot, bias, fpw = get_colvar_data()

        bb = bias.reshape((nb, fpw // nb, nw))
        time, odot, cdot = [n.reshape((nb, fpw // nb)) for n in [time, odot, cdot]]

        c = np.arange(0,100)

        # Get the log weights for reweighting with WHAM 
        w = wham.wham(bb[c,:,:].reshape((-1, nw)), T=kBT)
        
        # Write the logweights to a pandas table for reweighting
        colvar_weights = pd.DataFrame(data={"time" : time[c,:].flatten(), 
                            "openvec" : odot[c,:].flatten(), 
                            "closedvec" : cdot[c,:].flatten(), 
                            "logweights" : w["logW"]})
        
        # Use plumed to make reweighted histograms
        make_ith_histogram(colvar_weights, bs_path, "fulldata")

    fes = plumed.read_as_pandas(fes_dat).replace([np.inf, -np.inf], np.nan).dropna()

    return fes

if __name__ == '__main__':
    main(sys.argv)