#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=500
#SBATCH --partition=agkeller
#SBATCH --qos=standard
#SBATCH --job-name=cattraj

module add GROMACS/2021.5-foss-2021b-CUDA-11.4.1-PLUMED-2.8.0
MDP="/home/lf1071fu/unbiased_sims/MDP"
home=$(pwd)

files=$(ls -v run*/md_run/holo_open.xtc)

commands=$(printf 'c\n%.0s' {1..19})'\nc'

echo $files[@]

echo $commands

# Concatenate trajectories
echo -e "$commands" | gmx trjcat -f ${files[@]} -o cat_trj.xtc -nobackup -settime

echo "24 24 24" | gmx trjconv -f cat_trj.xtc -s run1/md_run/holo_open.tpr -center yes -pbc cluster -o centered_traj.xtc -n index.ndx -nobackup

echo "4 24" | gmx trjconv -f centered_traj.xtc -s run1/md_run/holo_open.tpr -fit rot+trans -o fitted_traj.xtc -n index.ndx -nobackup

echo "24" | gmx trjconv -f fitted_traj.xtc -s run1/md_run/holo_open.tpr -skip 100 -o fitted_traj_100.xtc -n index.ndx -nobackup

rm cat_trj.xtc centered_traj.xtc
