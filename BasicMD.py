import sys
import os
import argparse
import TrajAnalysis as ta

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
						dest = "filePath",
						action = "store",
						nargs = "+",
						help = "Path to Structure/topology file (GROMACS TOP which "/
                        "contains topology and dihedral information) followed by "/
                        "Trajectory file(s) (GROMACS TRR, XTC) you will need to "/
                        "output the coordinates without solvent/ions. Required.")

    parser.add_argument("-o","--out",
						action = "store",
						dest = "outFile",
						default = "outfile.out",
						help = "Path where the figures and output will be "/
                        "written. Default: cwd")

    parser.add_argument("-l", "--selectString",
						action = "store",
						dest = "selectionString",
						type = str,
						default = "all",
						help="Selection string for CodeEntropy such as protein "/
                        "or resid, refer to MDAnalysis.select_atoms for more "/
                        "information.")

	args = parser.parse_args()

except argparse.ArgumentError:
	print('Command line arguments are ill-defined, please check the arguments')
	raise

for arg in vars(args):
    print(' {} {}'.format(arg, getattr(args, arg) or ''))

filePath = args.filePath
outfile = args.outFile
selection_string = args.selectionString

topfile = filePath[0]
xtcfile = filePath[1:]

u = mda.Universe(topfile, xtcfile)

reduced_atom = MDAHelper.new_U_select_atom(u, selection_string)
