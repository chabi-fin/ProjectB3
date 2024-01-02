import sys
import numpy as np
import os
import argparse
import subprocess
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array
sys.path.insert(0, "/home/lf1071fu/project_b3/ProjectB3")
import config.settings as c
from tools import utils, traj_funcs
from biased.NewWindowConfs import add_colvar_data, cv_min_dist
from VectorCoordCombo import three_point_function, get_ref_vecs
import pandas as pd
from Bio import PDB

def main(argv):

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("-m", "--mutant",
                            action = "store",
                            dest = "mutant",
                            default = None,
                            help = ("Select the mutation experiment, "
                                "native, K57G, E200G or double_mut."))
        args = parser.parse_args()

    except argparse.ArgumentError:
        print("Command line arguments are ill-defined, please check the "
            "arguments")
        raise

    # Set command line arguement variable
    mut = args.mutant

    # Define four mutation experiments
    experiment = {"native" : [], "K57G" : [57], "E200G" : [200],
                "double_mut" : [57, 200]}
    if mut not in experiment.keys():
        print(f"Experiment '{ mut }' not a valid choice. Select a valid " 
            "input for mutation experiment: native, K57G, E200G or "
            "double-mut.")
        sys.exit(1)

    # Set the directory for the experiment
    workdir = f"{ c.data_head }/umbrella/salt_bridge/{ mut }"
    print("Working dir :", workdir)

    # Load in universe objects for the simulation and the reference
    # structures
    open_ref = mda.Universe(f"{ c.struct_head }/open_ref_state.pdb")
    closed_ref = mda.Universe(f"{ c.struct_head }/closed_ref_state.pdb")

    # Get the reference salt bridge distances of the alpha carbons for 
    # open and closed conformations
    res_200_sel = ("resid 200 and name CA")
    res_57_sel = ("resid 57 and name CA")
    open_dist = distance_array(open_ref.select_atoms(res_57_sel).positions, 
                    open_ref.select_atoms(res_200_sel).positions)[0][0]
    closed_dist = distance_array(closed_ref.select_atoms(res_57_sel).positions, 
                    closed_ref.select_atoms(res_200_sel).positions)[0][0]

    # Use evenly spaced 57CA--200CA distances to define restraint points
    # for umbrella windows
    num_us = 30
    restraint_pts = np.linspace(closed_dist, open_dist, num=num_us, 
        endpoint=True)

    # Make a table of windows and salt-bridge restraint points
    # Divide by 10 to converst distance from AA (MDAnalysis) to nm (plumed)
    df = pd.DataFrame({"Window" : np.arange(1,31),
                       "RestraintPoint" : restraint_pts / 10})
    df = df.assign(NearestWindow=0, NearestRun=0, NearestFrame=0)

    # Make a table of sampled beta-vec values to select starting confoms
    # which lie along the known transition pathway
    conf_path = f"{ c.data_head }/umbrella/holo_state/nobackup/window_data"
    df_cat = get_df_cat(conf_path) 

    # Get 2d points which very roughly approximates
    # the conformational transition path in the beta-vec space
    sample_points = get_sample_points(num_us, workdir)
    print("Getting initial conforms near the points in the beta-vector"
            " space (Open, Closed) :\n", sample_points)

    # Iterate over all the path points to find the closest conform from 
    # the sampled windows (w) to apply to the salt-bridge windows (window)
    for window, g in zip(df["Window"], sample_points):
        d, d_ind = cv_min_dist(g, (df_cat["opendot"], 
                                    df_cat["closeddot"]))
        
        # Record the trajectory and time frame to access a good initial 
        # conformation for each salt-bridge window
        w, r, t = df_cat.iloc[d_ind][["window", "run", "time"]]
        df.loc[df["Window"] == window, "NearestWindow"] = w
        df.loc[df["Window"] == window, "NearestRun"] = r
        df.loc[df["Window"] == window, "NearestFrame"] = t

    # Save DataFrame containing the initial conform data
    print(f"DF PTS, { df.shape }\n", df, "\n")
    utils.save_df(df, f"{ workdir }/salt_bridge_restraints.csv")

    # Get the plumed template file
    with open("plumed.dat", "r") as f:
        plumed_lines = f.readlines()  

    # Set up the plumed file and the initial conform as a pdb in the 
    # directories for each window
    for _, row in df.iterrows():

        # Select which residues should be mutated to GLY
        mutate = experiment[mut]
        
        # Prepare initial conformations and plumed files for the 
        # particular mutation experiment
        set_up_window(row, plumed_lines, mutate, workdir, conf_path)
        
def set_up_window(row, plumed_lines, mutate, workdir, conf_path):
    """
    """
    w = int(row["Window"])

    # Prepare paths
    destination = f"{ workdir }/window{ w }"
    os.makedirs(destination, exist_ok=True)
    plumed_wfile = f"{ destination }/plumed_{ w }.dat"
    extracted_out = f"{ destination }/extracted_conform.pdb"
    print("extracted out", extracted_out)

    # Get paths for extracting sampled conformation
    nw = str(int(row["NearestWindow"]))
    r = str(int(row["NearestRun"]))
    traj = f"{ conf_path }/window{ nw }/run{ r }/fitted_traj.xtc"
    top = f"{ conf_path }/window{ nw }/run{ r }/w{ nw }_r{ r }.tpr"
    time_frame = int(row["NearestFrame"])

    if not os.path.exists(traj):
        print(f"Missing path: { traj } for window { w }")
        return None

    # Define the gromacs command
    gmx = ["echo", "1", "|", "gmx22", "trjconv", "-f", 
        traj, "-s", top, "-o", extracted_out, "-b", 
        str(time_frame - 1000), "-dump", str(time_frame), "-nobackup"]
    print("\n", " ".join(gmx), "\n")
    
    # Use gromacs subprocess to extract the conformation at the 
    # desired time
    process = subprocess.Popen(" ".join(gmx), 
                                stdin=subprocess.PIPE, 
                                stdout=subprocess.PIPE,
                                shell=True, text=True)

    # Pass input to the GROMACS command to use protein only
    stdout, stderr = process.communicate("1\n")
    print("Output:", stdout)
    print("Error:", stderr)

    # Edit conformations to produce the desired mutant
    # Load the PDB file
    pdb_parser = PDB.PDBParser(QUIET=True)
    structure = pdb_parser.get_structure("mutated", extracted_out)
    for res in mutate:
        structure = mutate_to_glycine(structure, res)

    # Write the mutated structure to a new PDB file
    mutated_pdb = f"{ destination }/mutated.pdb"
    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(mutated_pdb)
    fix_gly_names(mutated_pdb)

    # Get the atom numbers for CA 57 and CA 200
    u = mda.Universe(mutated_pdb)
    ind_57 = u.select_atoms("resid 57 and name CA")[0].ix
    ind_200 = u.select_atoms("resid 200 and name CA")[0].ix

    # Substitute restraint values into the windows plumed file
    plumed_wlines = []
    for line in plumed_lines:
        if "RESTRAINT_SALT" in line:
            line = line.replace("RESTRAINT_SALT", 
                                str(np.round(row["RestraintPoint"], 3)))
        if "ATOM57" in line:
            line = line.replace("ATOM57", str(ind_57 + 1))
        if "ATOM200" in line:
            line = line.replace("ATOM200", str(ind_200 + 1))
        if "COLVAR_WINDOW" in line:
            line = line.replace("COLVAR_WINDOW", "COLVAR_" + str(w))
        plumed_wlines.append(line)

    # Write out the plumed file for the window
    with open(plumed_wfile, "w") as f:
        f.writelines(plumed_wlines)

    return 1

def mutate_to_glycine(structure, residue_id):
    """
    """
    gly_names = ["N", "H", "CA", "HA1", "HA2", "C", "O"]
    # Find the specified residue
    for residue in structure[0]["A"]:
        if residue.id[1] == residue_id:
            # Mutate the residue to glycine
            residue.resname = "GLY"
            residue.id = (" ", residue_id, " ")

            # Remove atoms extraneous to glycine
            removal_atoms = []
            for atom in residue.get_atoms():
                # Modify the atom name if needed
                if atom.name == "HA": 
                    atom.name = "HA1"
                elif atom.name == "CB":
                    atom.name = "HA2"
                if not any([atom.name == a for a in gly_names]):
                    removal_atoms.append(atom.name)

            for atom in removal_atoms:
                residue.detach_child(atom)

    return structure

def fix_gly_names(pdb):
    # Get the plumed template file
    with open(pdb, "r") as f:
        pdb_lines = f.readlines()  

    # Fix the atom names of glycine
    pdb_new_lines = []
    for line in pdb_lines:
        if "HA  GLY A  57" in line:
            line = line.replace("HA  GLY A  57", "HA1 GLY A  57")
        if "CB  GLY A  57" in line:
            line = line.replace("CB  GLY A  57", "HA2 GLY A  57")
        if "HA  GLY A 200" in line:
            line = line.replace("HA  GLY A 200", "HA1 GLY A 200")
        if "CB  GLY A 200" in line:
            line = line.replace("CB  GLY A 200", "HA2 GLY A 200")
        pdb_new_lines.append(line)

    # Write out the pdb file
    with open(pdb, "w") as f:
        f.writelines(pdb_new_lines)

def get_sample_points(num_us, workdir):
    """Finds relevant points in the beta-vec space. 

    Sketches a polynomial which very roughly approximates the 
    conformational transition path in the beta-vec space and draws a 
    point for each window, evenly spaced wrt the x-axis.

    Parameters
    ----------
    num_us : int
        The number of umbrella windows.
    workdir : str
        Path to the working dir. 
    
    Returns
    -------
    sample_points : np.ndarray()
        The 2D points along the approximate transition path.

    """
    _, vec_open, vec_closed = get_ref_vecs(c.struct_head, 206, 215)
    p1 = (np.dot(vec_closed, vec_open), np.dot(vec_closed, vec_closed))
    p2 = (3.6, 3.6)
    p3 = (np.dot(vec_open, vec_open), np.dot(vec_open, vec_closed))

    # coefficients for a 3d polynomial 
    f = three_point_function(p1, p2, p3)

    sample_points = np.zeros((num_us,2))
    sample_points[:,0] = np.linspace(p1[0], p3[0], num_us)
    sample_points[:,1] = [f[0]*x**2 + f[1]*x + f[2] for x 
                                in sample_points[:,0]] 

    # Make a plot of the points
    plot_rxn_coord(sample_points, workdir, "initial_conforms")

    return sample_points

def plot_rxn_coord(samples, fig_path, state):
    """Makes a plot to show where samplec conformations lie in beta-vec space.

    This is a very rough approximation of the path taken according to the 
    beta-flap rotation in transitioning between the open and closed 
    conformations.

    Parameters
    ----------
    samples : np.ndarray()
        The 2D points along the approximate transition path.
    fig_path : str
        Directory where the figure should be stored.
    state : str
        A label for naming the figure file.
    """
    # Initialize figure
    fig, ax = plt.subplots(constrained_layout=True)

    # Add reference positions
    _, vec_open, vec_closed = get_ref_vecs(c.struct_head, 206, 215)
    ax.scatter(np.dot(vec_open, vec_open), np.dot(vec_open, vec_closed), 
                label="Open ref.", marker="X", color=c.closed_color, 
                edgecolors="#404040", s=550,lw=3)
    ax.scatter(np.dot(vec_open, vec_closed), np.dot(vec_closed, vec_closed), 
                label="Closed ref.", marker="X", color=c.open_color, 
                edgecolors="#404040", s=550, lw=3)

    # Add restraint points
    ax.scatter(samples[:,0], samples[:,1], marker="o",
                label="Extract conformation", color="#949494", 
                edgecolors="#EAEAEA", lw=3, s=150)

    # Plot settings
    ax.set_xlabel(r"$\vec{\upsilon} \cdot \vec{\upsilon}_{open}$ (nm$^2$)",
        labelpad=5, fontsize=24)
    ax.set_ylabel(r"$\vec{\upsilon} \cdot \vec{\upsilon}_{closed}$ (nm$^2$)", 
        labelpad=5, fontsize=24)
    plt.legend(fontsize=18)

    utils.save_figure(fig, f"{ fig_path }/{ state }_beta_vec.png")
    plt.close()

def get_df_cat(conf_path):
    """Makes a table of concatenated sim data, at 1 ns intervals.

    Parameters
    ----------
    conf_path : str
        Path to the data directory for a FES.
    
    Returns
    -------
    df_cat : pd.DataFrame
        A table of the colvar data, useful for knowing the simulation 
        information (window and run) for sampled beta-vec values.
    """
    # Initialize dataframe
    df_cat = pd.DataFrame(columns=["time", "opendot", "closeddot"])

    # Iterate over window data
    for w in range(1,171):
        for r in range(1,5):
            file = f"{ conf_path }/window{ w }/run{ r }/COLVAR_{ w }.dat"
            if os.path.exists(file):

                # Stride = 100 for 1 ns intervals
                df_new = add_colvar_data(w, r, file, stride=100)

                df_cat = pd.concat([df_cat, df_new])

    return df_cat

if __name__ == "__main__":
    main(sys.argv)