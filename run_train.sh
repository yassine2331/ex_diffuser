## Shebang
#!/bin/bash
## Resource Request
#SBATCH --job-name=ExperimentA
#SBATCH --output=ExperimentA.out
#SBATCH --time=1-12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=100G
#SBATCH --gpus=2
## Job Steps


srun echo " Start process "
srun python experiments/test.py
srun echo " End process "