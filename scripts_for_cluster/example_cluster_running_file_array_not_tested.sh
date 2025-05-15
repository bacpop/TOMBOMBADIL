#!/bin/bash # Indicates this is a bash script

#Submit this script with: sbatch thefilename
#For more details about each parameter, please check SLURM sbatch documentation https://slurm.schedmd.com/sbatch.html
#SBATCH --time=2:00:00   # walltime requested
#SBATCH --ntasks=1   # number of tasks
#SBATCH --cpus-per-task=1   # number of CPUs Per Task i.e if your code is multi-threaded
#SBATCH --nodes=1   # number of nodes/cores
#SBATCH --mem=8G   # memory per node. Will need to reduce run time and memory.
#SBATCH -J "PorB Single Locus"   # job name, rename
#SBATCH -o "%j.out"   # job output file %j is the name of the job on the cluster, a number
#SBATCH -e "%j.err"   # job error file#!/usr/bin/env bash
#SBATCH --array=0-  #number of genome files -1

# Extract the FASTA file path for the current job
GENOME_FILE=$(awk -F '\t' "NR==$((SLURM_ARRAY_TASK_ID+2)) {print \$NF}" data/modified_pangenome_summary.tsv)

echo "Processing file: $GENOME_FILE"

######################################################################################
##
## SECOND SECTION:
## Here we define the modules we need to be loaded in order to run our job.
## In this case, we need to load the R module, ... but first we make sure
## we start off with a sane, blank environment so we purge all modules
##
######################################################################################

module purge
module load r/4.2.2

######################################################################################
##
## THIRD SECTION:
## Here we define the executable/binary/script we want LSF to run, our real data
## crunching section. Remember, in our case: we want to run an R script that
## calculates factorial of 8 and our R script is called "r.script".
##
## As you can see, we specify the FULL path to the script and we redirect the output
## to a file.
##
######################################################################################

Rscript --vanilla sillyScript.R iris.txt out.txt # Run R script with arguments
#  sbatch example_cluster_running_file.sh # Run on the terminal on cluster separately