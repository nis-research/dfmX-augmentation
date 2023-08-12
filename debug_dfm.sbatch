#!/bin/bash
#SBATCH -J Training
#SBATCH -c 2
#SBATCH -p main
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL 
#load modules
module load nvidia/cuda-11.7
module load nvidia/cuda-11.x_cudnn-8.6
module load anaconda3/2022.05

#source activate
source activate  debugenv # virtualenv # 

echo $SLURM_NODELIST

cd dfmX-augmentation

python -u Compute_DFM_CIFAR.py  --backbone_model resnet18 --model_path ./results/resnet18/version_1/checkpoints/last.ckpt      
# python -u train.py   --backbone_model resnet18 --lr 0.01  --save_dir results/   --p 0.3  --masks  ./DFMs/resnet18.pkl
# python -u Evaluation/verify_mask.py  --backbone_model resnet18  --model_path ./results/resnet18/version_1/checkpoints/last.ckpt  --m_path  ./DFMs/resnet18


conda deactivate
