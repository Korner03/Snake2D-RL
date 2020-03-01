#!/bin/csh

#SBATCH --cpus-per-task=2
#SBATCH --output=<YOUR_OUTPUT_PATH_HERE>
#SBATCH --mem-per-cpu=500M
#SBATCH --account=aml
#SBATCH --constraint="sm"

source /cs/labs/dshahaf/omribloch/env/snake/snake/bin/activate.csh
module load tensorflow

python3 Snake.py -D 5000 -s 1000 -P "Linear(epsilon =0.3);Avoid(epsilon =0.3);Avoid(epsilon =0.3)" -bs "(90 ,90)" -plt 0.01 -pat 0.005 -r 0
