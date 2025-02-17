#!/bin/sh
#SBATCH -o out
#SBATCH -p mpi
#SBATCH -J qe
#SBATCH -t 16:00:00
#SBATCH -n 10
#SBATCH --exclude=cnode31,cnode37,cnode24,cnode35,cnode30,cnode29

module load mpi/impi-5.0.3

srun -n 10 pw.x < scf.in >scf.out
