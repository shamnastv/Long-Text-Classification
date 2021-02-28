#!/bin/sh
#SBATCH --job-name=nlp # Job name
#SBATCH --ntasks=4 # Run on a single CPU
#SBATCH --time=23:50:00 # Time limit hrs:min:sec
#SBATCH --output=test_job%j.out # Standard output and error log
#SBATCH --gres=gpu:1
#SBATCH --partition=cl1_48h-1G

#printf "min freq 1\n"

#python3 main.py --lr 1e-3 --num_layers 1 --hidden_dim 50

python3 main.py --lr 1e-3 --hidden_dim 100 --weight_decay 1e-6 --dropout .5
python3 main.py --lr 5e-4 --hidden_dim 100 --weight_decay 1e-6 --dropout .5
python3 main.py --lr 1e-4 --hidden_dim 100 --weight_decay 1e-6 --dropout .5
python3 main.py --lr 5e-5 --hidden_dim 100 --weight_decay 1e-6 --dropout .5

#python3 main.py --lr 1e-3 --hidden_dim 100 --weight_decay 1e-6 --dropout .5
#python3 main.py --lr 1e-3 --hidden_dim 50 --weight_decay 1e-3 --dropout .5
#python3 main.py --lr 5e-3 --hidden_dim 50 --weight_decay 1e-4 --dropout .5