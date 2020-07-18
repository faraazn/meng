#!/bin/bash
#SBATCH -J retro-video
#SBATCH --mail-user=faraaz@mit.edu
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=1T
#SBATCH --time=24:00:00
#SBATCH --exclusive

## User python environment
HOME2=/nobackup/users/faraaz
PYTHON_VIRTUAL_ENVIRONMENT=py36
CONDA_ROOT=$HOME2/anaconda3

## Activate WMLCE virtual environment
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT
ulimit -s unlimited

## Creating SLURM nodes list
export NODELIST=nodelist.$
srun -l bash -c 'hostname' |  sort -k 2 -u | awk -vORS=, '{print $2":148"}' | sed 's/,$//' > $NODELIST

## Number of total processes
echo " "
echo " Nodelist:= " $SLURM_JOB_NODELIST
echo " Slurm ntasks:= " $SLURM_NTASKS
##echo " Number of nodes:= " $SLURM_JOB_NUM_NODES
##echo " GPUs per node:= " $SLURM_JOB_GPUS
##echo " Ntasks per node:= "  $SLURM_NTASKS_PER_NODE


####    Use MPI for communication with Horovod - this can be hard-coded during installation as well.
export HOROVOD_GPU_ALLREDUCE=MPI
export HOROVOD_GPU_ALLGATHER=MPI
export HOROVOD_GPU_BROADCAST=MPI
export NCCL_DEBUG=DEBUG

echo " Running on 1 node/GPU device"
echo ""
echo " Run started at:- "
date

## Horovod execution
mpirun -np $SLURM_NTASKS -H `cat $NODELIST` python /home/faraaz/workspace/meng/pytorch-a2c-ppo-acktr-gail/main.py

echo "Run completed at:- "
date
