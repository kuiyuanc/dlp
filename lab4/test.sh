#!/bin/sh
#SBATCH -A dlp_acct             # Account name (可透過 sshare 查看)
#SBATCH -p dlp_nodes            # Partition name (可透過 sinfo 查看 partition)
#SBATCH -N 1                    # Maximum number of nodes to be allocated
#SBATCH --gres=gpu:1            # specify GPU number
# 還有很多其他參數，可參考官方文件(https://slurm.schedmd.com/sbatch.html)
export LD_PRELOAD=/lib/x86_64-linux-gnu/libcuda.so.1
cd Lab4_template; srun python3 Tester.py --DR ../LAB4_Dataset --save_root ../data --ckpt_path ../checkpoints/epoch=70.ckpt --num_workers 1
