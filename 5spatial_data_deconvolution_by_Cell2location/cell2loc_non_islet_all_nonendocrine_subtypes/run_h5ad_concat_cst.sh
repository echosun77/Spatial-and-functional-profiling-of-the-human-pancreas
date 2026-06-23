#!/bin/env bash
#SBATCH -A NAISS2025-22-1270  # Project account to charge compute time
#SBATCH -p alvis              # Partition (queue) to run the job in
#SBATCH -t 0-00:30:00         # Time limit: 10 hours
#SBATCH -N 1 --gpus-per-node=A40:1  # Request 1 node with 1 A40 GPU
#SBATCH --array=1          # Adjust based on total donor splits (2 per donor)
#SBATCH --mail-type=ALL       # Send email notifications for all job states
#SBATCH --mail-user=yike.xie@gu.se  # Email recipient
#SBATCH -o logs/20251013/%x_%A_%a.out
#SBATCH -e logs/20251013/%x_%A_%a.err
#SBATCH -J concat_h5ad_files_cst

#TODAY=$(date +%Y%m%d)
#mkdir -p logs/${TODAY}
#exec > logs/${TODAY}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out 2>&1

# Load required modules
ml purge
ml load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
ml load Python/3.11.3-GCCcore-12.3.0
ml load SciPy-bundle/2023.07-gfbf-2023a
ml load scikit-learn/1.3.1-gfbf-2023a
ml load matplotlib/3.7.2-gfbf-2023a
ml load h5py/3.9.0-foss-2023a
ml load IPython/8.14.0-GCCcore-12.3.0
ml load JupyterLab/4.0.5-GCCcore-12.3.0
ml load Pillow/10.0.0-GCCcore-12.3.0
ml load plotly.py/5.16.0-GCCcore-12.3.0
ml load JupyterNotebook/7.0.2-GCCcore-12.3.0

python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Activate the virtual environment
CUSTOM_VENV_PATH=~/mercedes_storage/fab/pyenv_fab_cell2location
source $CUSTOM_VENV_PATH/bin/activate

# Run
python concat_h5ads_cst.py \
    --input_dir ../../results/cell2loc_non_islet/cst_c2l/ \
    --output_dir ../../results/cell2loc_non_islet/cst_c2l/

