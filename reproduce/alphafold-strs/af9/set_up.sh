#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=500
#SBATCH --partition=agkeller
#SBATCH --job-name=af_setup
#SBATCH --qos=standard

module add GROMACS/2021.5-foss-2021b-CUDA-11.4.1-PLUMED-2.8.0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=false
MDP="/home/lf1071fu/unbiased_sims/MDP"
home=$(pwd)

# Generate topology using .pdb of capped residue
echo "1 1 1" | gmx pdb2gmx -ff amber14sb -f af_9.pdb -o cpd_only.gro -ignh -water tip3p -nobackup -his
sed -i "s/HISE/ HIS/" topol.top

# Define dodecahedron box with ligand at center, > 1.2 nm to edge
gmx editconf -f cpd_only.gro -o cpd_box.gro -c -d 1.2 -bt dodecahedron -nobackup

# Solvate ligand with TIP3P
gmx solvate -cp cpd_box.gro -cs spc216 -o cpd_tip3p.gro -p topol.top -nobackup

# Add ions if needed
gmx grompp -f ${MDP}/em_steep.mdp -c cpd_tip3p.gro -p topol.top -o ions.tpr -maxwarn 1 -nobackup
echo "SOL" | gmx genion -s ions.tpr -o cpd_initial.gro -p topol.top -pname NA -pq 1 -np 36 -nname CL -nq -1 -nn 19 -nobackup
