import sys
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis import align, pca
import pandas as pd
import seaborn as sns
import nglview as nv

def main(argv):

    # Add command line arg to control group selection
    try:
        parser = argparse.ArgumentParser()

        parser.add_argument("-g", "--group",
    						action = "store",
    						dest = "group",
    						default = "beta-flap",
    						help = "Choose the subgroup for PCA. Options = backbone, "\
                                    "alpha-flap, beta-flap, alpha-beta-flap. "\
                                    "Default = beta-flap")
        parser.add_argument("-c", "--conform",
                            action = "store",
                            dest = "conform",
                            default = "holo",
                            help = """Chose a conformer for analysis. I.e. "holo" or "apo".""")
        args = parser.parse_args()

    except argparse.ArgumentError:
    	print("Command line arguments are ill-defined, please check the arguments")
    	raise

    # Assign group selection from argparse
    group = args.group
    conform = args.conform

    # Set up paths
    path_head = "/home/lf1071fu/project_b3"
    if conform == "holo": 
        path = f"{ path_head }/simulate/holo_conf/data"
        fig_path = f"{ path_head }/figures/holo/pca"
    elif conform == "apo":
        path = f"{ path_head }/simulate/apo_conf/initial_10us"
        fig_path = f"{ path_head }/figures/apo/pca"
    else:
        print("Select either holo or apo conformer using the command line arguement '-c'.")
        sys.exit(1)
    struct_path = f"{ path_head }/structures"

    # # Find low rmsf residues for alignment fitting
    # core_res = np.load(f"{ path }/core_res.npy")
    # aln_str = "protein and name CA and ("
    # core_holo = [f"resid {i} or " for i in core_res]
    # core_apo = [f"resid {i + 544} or " for i in core_res]
    # core = aln_str + "".join((core_holo + core_apo))[:-4] + ")"

    # Set up selection strings for atom groups, using all atoms in the selected residues
    # beta_flap = "backbone and (resnum 195-231 or resnum 740-776)"
    if group == "alpha-flap":
        flap = "resid 219-231 or resid 763-775"
    elif group == "backbone":
        flap = "resid 8-251 or resid 552-795"
    elif group == "alpha-beta-flap": 
        flap = "resid 195-231 or resid 739-775"
    else:
        # The beta-flap group (default)
        flap = "backbone and (resid 195-218 or resid 739-762)"
    ref_backbone = "resid 8-251 or resid 552-795"

    # Load in relevant reference structures
    if conform == "holo":
        ref_state = mda.Universe(f"{ struct_path }/holo_state.pdb")
    else:
        ref_state = mda.Universe(f"{ struct_path }/apo_state.pdb")

    # Load in and align the traj
    top = f"{ path }/topol.top"
    traj = f"{ path }/fitted_traj.xtc"
    u = mda.Universe(top, traj, topology_format="ITP", dt=1000)
    # align.AlignTraj(u, ref_state, select=f"name CA and ({ref_backbone})", in_memory=True).run()
    align.AlignTraj(u, u.select_atoms('name CA'), select="name CA", in_memory=True).run()
    uf = u.select_atoms(flap)

    p_components, cumulated_variance, transformed, mean = get_pc_data(u, \
                                                    flap, uf, group, path)

    print(cumulated_variance)

    # n_pcs = np.where(pc.results.cumulated_variance > 0.95)[0][0]
    # print(f"The first { n_pcs } principal components explain at least 95% of the "\
    #        "total variance.\n")

    plot_eigvals(cumulated_variance, group, os.getcwd())

    plot_3PC(transformed, group, os.getcwd())

    for i in range(3):
        pc_min_max(transformed, u, group, i, os.getcwd())

    for i in range(3):
        visualize_PC(p_components, transformed, mean, i, uf, group, os.getcwd())

    return None

def get_pc_data(u, flap, atom_group, group, path):
    """
    """
    pc_files = [f"{ path }/p_components_{ group }.npy", 
                f"{ path }/cumul_vaviance_{ group }.npy",
                f"{ path }/transformed_{ group }.npy",
                f"{ path }/mean_{ group }.npy"]

    if False: #all(list(map(lambda x : os.path.exists(x), pc_files))):

        p_components = np.load(pc_files[0], allow_pickle=True)
        cumulated_variance = np.load(pc_files[1], allow_pickle=True)
        transformed = np.load(pc_files[2], allow_pickle=True)
        mean = np.load(pc_files[3], allow_pickle=True)

    else:

        pc = pca.PCA(u, select=flap, align=True, n_components=50).run()
        p_components = pc.results.p_components
        cumulated_variance = pc.results.cumulated_variance

        # Transforms the atom group into weights over each principal component
        # Here, weights of the first 3 components; shape --> (n_frames, n_PCs) 
        transformed = pc.transform(atom_group, n_components=3)

        mean = pc.mean.flatten()

        np.save(pc_files[0], p_components)
        np.save(pc_files[1], cumulated_variance)
        np.save(pc_files[2], transformed)
        np.save(pc_files[3], mean)

    return p_components, cumulated_variance, transformed, mean

def plot_eigvals(cumulated_variance, group, fig_path):
    """Make a plot of the cummulative variance.

    The relative contribution to the overall variance by each principle
    component is shown for the first 25 PCs. This should simply demonstrate the
    number of relevant PCs in the dimensionality reduction.

    Parameters
    ----------
    cumulated_variance : np.1darray
        The cumulative variance, calculated from PCA eigenvalues.
    group : str
        The name of the subgroup selected for PCA.
    path : str
        The path to the directory where the figure should be saved.

    Returns
    -------
    None.

    """
    fig, ax = plt.subplots(constrained_layout=True)
    filled_marker_style = dict(marker='o', markersize=10, linestyle="-", lw=3,
                               markeredgecolor='#A31130')
    ax.plot(cumulated_variance[:25], color="#FF6666",
            **filled_marker_style)

    # Plot settings
    ax.tick_params(axis='y', labelsize=16, direction='in', width=2, \
                    length=5, pad=10)
    ax.tick_params(axis='x', labelsize=16, direction='in', width=2, \
                    length=5, pad=10)
    for i in ["top","bottom","left","right"]:
        ax.spines[i].set_linewidth(2)
    ax.grid(True)
    ax.set_xlabel(r"Principle component", labelpad=5, fontsize=16)
    ax.set_ylabel(r"Cumulative variance", labelpad=5, fontsize=16)

    # Save fig
    plt.savefig(f"{ fig_path }/pca_scree_{ group }.png")
    plt.close()

    return None

def plot_3PC(transformed, group, fig_path):
    """Plot the first 3 principle components against each other. 

    Parameters
    ----------
    transformed : np.ndarray
        Trajectory projected onto the first 3 principle components.
    group : str
        The name of the subgroup selected for PCA.
    path : str
        The path to the directory where the figure should be saved.

    Returns
    -------
    None. 

    """
    df = pd.DataFrame(transformed, columns=['PC{}'.format(i+1) for i in range(3)])

    df["Time (ns)"] = df.index

    fig, axes2d = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True, 
                               constrained_layout=True)

    for i, row in enumerate(axes2d):
        for j, cell in enumerate(row):
            ts = df["Time (ns)"]
            cb = cell.scatter(df[f"PC{i+1}"], df[f"PC{j+1}"], c= ts, 
                              cmap="YlGnBu")
            if i == len(axes2d) - 1:
                cell.set_xlabel(f"PC {j+1}", fontsize=16)
            if j == 0:
                cell.set_ylabel(f"PC {i+1}", fontsize=16)
            if j == 2:
                cbar = fig.colorbar(cb, ax=cell)
                cbar.set_label("Time ($\mu$s)", fontsize=16)
                # cticks = list(map(lambda x: str(x/1000).split(".")[0],
                #       np.arange(0,len(ts),5000)))
                cbar.set_ticks(np.arange(0,len(ts),5000))
                ticks_loc = cbar.get_ticks().tolist()
                cbar.ax.yaxis.set_ticklabels([str(x/1000).split(".")[0] \
                                              for x in ticks_loc])
                cbar.ax.tick_params(labelsize=16, direction='out', width=2, length=5)
                cbar.outline.set_linewidth(2)
            cell.tick_params(axis='y', labelsize=16, direction='in', width=2,
                            length=5, pad=10)
            cell.tick_params(axis='x', labelsize=16, direction='in', width=2,
                            length=5, pad=10)
            for n in ["top","bottom","left","right"]:
                cell.spines[n].set_linewidth(2)

    # Save fig
    plt.savefig(f"{ fig_path }/pca_first3_{ group }.png")
    plt.close()

    return None

def pc_min_max(transformed, u, group, rank, fig_path):
    """
    """
    pc = transformed[:,rank - 1]
    protein = u.select_atoms("protein")

    min_ind = np.argmin(pc)
    u.trajectory[min_ind]
    protein.write(f"PC{ rank }_min_{ group }.pdb")

    max_ind = np.argmax(pc)
    u.trajectory[max_ind]
    protein.write(f"PC{ rank }_max_{ group }.pdb")

    return None

def visualize_PC(p_components, transformed, mean, rank, atom_group, group, fig_path):
    """
    """
    pc1 = p_components[:, rank - 1]
    print(f"pc1 shape : { np.shape(pc1)}")
    trans1 = transformed[:, rank - 1]
    print(f"trans1 shape : { np.shape(trans1)}")
    projected = np.outer(trans1, pc1) + mean
    coordinates = projected.reshape(len(trans1), -1, 3)

    proj1 = mda.Merge(atom_group)
    proj1.load_new(coordinates, order="fac")

    with mda.Writer(f"{ fig_path }/pc_{ group }{ rank }.xtc", atom_group.n_atoms) as W:
        for ts in proj1.trajectory:
            if ts.frame % 10 == 0:
                W.write(proj1)

    with mda.Writer(f"{ fig_path }/pc_{ group }{ rank }.gro") as W:
        W.write(proj1.atoms)

    return None

if __name__ == '__main__':
    main(sys.argv)