#!/bin/bash -l

#$ -pe omp 4
#$ -P ivc-ml
#$ -l h_rt=24:00:00
#$ -l gpus=1
#$ -l gpu_memory=16G
#$ -N gen_spatial
#$ -j y
#$ -m ea
#$ -o gen_spatial.out

cd /projectnb/ivc-ml/array/research/robotics/dreamworlds/scripts/3d_reasoning_qas/
conda activate genAI_design
python generate_3d_spatial_qas_procthor.py
