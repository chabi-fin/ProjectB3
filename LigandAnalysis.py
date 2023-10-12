import sys
import numpy as np
import os
import shutil
import argparse
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import MDAnalysis as mda
from MDAnalysis.analysis import rms, align
from MDAnalysis.analysis.distances import distance_array

def main(argv):

    try:
        parser = argparse.ArgumentParser()

        parser.add_argument("-p", "--path",
                            action = "store",
                            nargs='+',
                            dest = "path",
                            default = "/home/lf1071fu/project_b3/simulate/holo_state/open/data /home/lf1071fu/project_b3/simulate/holo_state/closed/data",
                            help = """Set path to the data directory.""")
        parser.add_argument("-r", "--recalc",
                            action = "store_true",
                            dest = "recalc",
                            default = False,
                            help = """Chose whether the trajectory arrays should  be recomputed.""")
        parser.add_argument("-f", "--figpath",
                            action = "store",
                            dest = "fig_path",
                            default = None,
                            help = """Set a path destination for the figure.""")              
        args = parser.parse_args()

    except argparse.ArgumentError:
        print("Command line arguments are ill-defined, please check the arguments")
        raise

    global path_head

    # Assign group selection from argparse 
    data_paths = args.path
    fig_path = args.fig_path
    recalc = args.recalc
    path_head = "/home/lf1071fu/project_b3"

    if not fig_path:
        fig_path = data_paths[0]

    datas = get_datas(data_paths, recalc)

    plot_coms(data_paths, datas, fig_path)

def get_core_res(recalc=False):
    """Finds the core residues which are immobile across the conformational states.

    Uses data from the combined simulation of the apo states open and closed simulations,
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
    core_res_path = f"{ path_head }/simulate/apo_state/open/data"
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

def get_datas(data_paths, recalc):
    """
    """
    datas = {}

    # Indicies of the inflexible residues
    core_res, core = get_core_res()

    for path in data_paths:

        analysis_path = f"{ path }/analysis"
        if not os.path.exists(analysis_path):
            os.makedirs(analysis_path)

        datas[path] = {}
        np_files = {"COM_dist" : f"{ analysis_path }/com_dist.npy", 
                    "time_ser" : f"{ analysis_path }/timeseries.npy"}

        if all(list(map(lambda x : os.path.exists(x), np_files.values()))) and not recalc:

            print(
                "LOADING NUMPY ARRAYS"
            )

            for key, file in np_files.items(): 
                datas[path][key] = np.load(file, allow_pickle=True)

        else:

            print(
                "EVALUATING WITH MDANALYSIS"
            )

            topol = f"{ path }/topol_Pro_Lig.top"
            if not os.path.exists(topol):
                with open(f"{ path }/topol.top", "r") as file:
                    lines = file.readlines()

                filtered_lines = [line for line in lines if all(not line.startswith(s) for s  in ["SOL", "NA", "CL"])]
                with open(topol, 'w') as file:
                    file.writelines(filtered_lines)

            # Load in universe objects for the simulation and the reference structures
            u = mda.Universe(topol, f"{ path }/fitted_traj.xtc",
                            topology_format='ITP')

            align.AlignTraj(u, u.select_atoms("protein"), select=core, in_memory=True).run()

            # Make a np array of the trajectory time series
            time_ser = []
            for ts in u.trajectory:
                time_ser.append(ts.time)
            time_ser = np.array(time_ser)

            datas[path]["time_ser"] = time_ser
            np.save(np_files["time_ser"], time_ser)
            
            # Make a np array of the COM dist wrt the starting position
            ipl = u.select_atoms("resname IPL")
            com_init = ipl.center_of_mass()
            com_dist = np.zeros(u.trajectory.n_frames)
            for ts in u.trajectory:
                com_dist[ts.frame] = distance_array(com_init, ipl.center_of_mass())

            datas[path]["COM_dist"] = com_dist
            np.save(np_files["COM_dist"], com_dist)

    return datas

def plot_coms(data_paths, datas, fig_path):
    """Makes a time series plot of the distance from the ligand's initial COM.

    Returns
    -------
    None.

    """
    fig, ax = plt.subplots(constrained_layout=True, figsize=(12,4))
    labels = ["open conf", "closed conf"]
    labels = ["str1", "str2", "str3"]
    stride = 50
    i = 0

    for p in data_paths:

        arrs = datas[p]
        time_ser = arrs["time_ser"]
        com_dist = arrs["COM_dist"]

        plt.plot(time_ser[::stride], com_dist[::stride], "-", lw=3, label=labels[i], alpha=0.8,
            path_effects=[pe.Stroke(linewidth=5, foreground='#595959'), pe.Normal()])

        i += 1

    num_ticks = 10
    xticks = time_ser[::(len(time_ser) // num_ticks)]
    ax.set_xticks(xticks)
    if time_ser[-1] > 1e6:
        # Use microseconds for time labels
        ax.set_xlabel(r'Time ($\mu$s)', fontsize=24, labelpad=5)
        ax.set_xticklabels(list(map(lambda x : str(x/1e6).split(".")[0], xticks)))
    else:
        # Use nanoseconds for time labels
        ax.set_xlabel(r'Time (ns)', fontsize=24, labelpad=5)
        ax.set_xticklabels(list(map(lambda x : str(x/1e3).split(".")[0], xticks)))

    ax.set_ylabel(r"Distance ($\AA$)", labelpad=5, fontsize=24)
    plt.legend(fontsize=20)

    plt.savefig(f"{ fig_path }/com_dist.png", dpi=300)
    plt.show()
    plt.close()

    return None

if __name__ == '__main__':
    main(sys.argv)