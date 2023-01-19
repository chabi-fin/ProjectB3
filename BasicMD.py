import sys
import os
import argparse
import TrajFunctions as tf
import MDAnalysis as mda
import numpy as np

try:
    parser = argparse.ArgumentParser(description="""
	BasicMD is a tool which provides a basic analysis for a biomolecular
    trajectory.
	Author:
        Lauren Marie Finn (chabi-fin)
	Output:
		*.png = figures from different analysis types.
    """)

    parser.add_argument("-f", "--top_traj_file",
						required = True,
						dest = "file_path",
						action = "store",
						nargs = "+",
						help = "Path to Structure/topology file (GROMACS TOP which "\
                        "contains topology and dihedral information) followed by "\
                        "Trajectory file(s) (GROMACS TRR, XTC) you will need to "\
                        "output the coordinates without solvent/ions. Required.")

    parser.add_argument("-o","--out",
						action = "store",
						dest = "out_path",
						default = "/home/lf1071fu/project_b3/figures/holo",
						help = "Path where the figures and output will be "\
                        "written. Default: cwd")

    parser.add_argument("-l", "--select_string",
						action = "store",
						dest = "selection_string",
						type = str,
						default = "all",
						help= "Selection string such as protein or resid, "\
                        "refer to MDAnalysis.select_atoms for more information.")

    parser.add_argument("-s", "--ref_structs",
                        action = "store",
                        dest = "ref_structs",
                        default = ["/home/lf1071fu/project_b3/structures/holo_state.pdb",
                                   "/home/lf1071fu/project_b3/structures/apo_state.pdb"],
                        help = "Path to the reference structure(s) (PDB).")

    parser.add_argument("-r", "--rmsd_group",
                        action = "store",
                        dest = "rmsd_group",
                        default = "backbone and (resid 198-231 or resid 743-776)",
                        help = "Selection string such as protein or resid "\
                        "used for calculating the RMSD, refer to "\
                        "MDAnalysis.select_atoms for more information.")

    parser.add_argument("-a", "--alignment_group",
                        action = "store",
                        dest = "alignment",
                        default = "backbone and (resid 8-251 or resid 552-795)",
                        help = "Selection string such as protein or resid used "\
                        "for alignment to a reference structure, refer to "\
                        "MDAnalysis.select_atoms for more information.")

    parser.add_argument("-n", "--ref_names",
                        action = "store",
                        dest = "ref_names",
                        #default = None,
                        default = ["holo state", "apo state"],
                        help = "Name the reference structure(s) for figures "\
                        "file naming, e.g. \"holo state\", \"apo state\"")

    args = parser.parse_args()

except argparse.ArgumentError:
	print("Command line arguments are ill-defined, please check the arguments")
	raise

for arg in vars(args):
    print(" {} {}".format(arg, getattr(args, arg) or ''))

file_path = args.file_path
out_path = args.out_path
selection_string = args.selection_string
ref_structs = args.ref_structs
rmsd_group = args.rmsd_group
align_group = args.alignment
ref_names = args.ref_names

topfile = file_path[0]
xtcfile = file_path[1:]

if ref_names == None:
    ref_names = []
    for i in range(len(ref_structs)):
        ref_names.append(f"state { i+1 }")
elif len(ref_names) != len(ref_structs):
    print("The number of reference structures and reference names must be the same.")
    sys.exit(1)

u = mda.Universe(topfile, xtcfile, topology_format="ITP")

u.transfer_to_memory()

# Determine RMSD to ref structures
for struct, ref in zip(ref_structs, ref_names):

    u_ref = mda.Universe(struct)

    rmsd = tf.get_rmsd(u, u_ref, align_group, rmsd_group)

    # Use data from every 100 picoseconds
    tf.plot_rmsd(rmsd[::10], out_path, group=r"$\beta$-flap", ref=ref)
