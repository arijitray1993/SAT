#!/bin/bash -l

#$ -pe omp 4
#$ -P ivc-ml
#$ -l h_rt=24:00:00
#$ -l gpus=1
#$ -l gpu_memory=16G
#$ -N gen_navigation
#$ -j y
#$ -m ea
#$ -o gen_navigation.out

cd /projectnb/ivc-ml/array/research/robotics/dreamworlds/scripts/3d_reasoning_qas/
conda activate genAI_design
python generate_navigation_questions_procthor.py
