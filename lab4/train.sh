#!/bin/sh
#SBATCH -A dlp_acct             # Account name (可透過 sshare 查看)
#SBATCH -p dlp_nodes            # Partition name (可透過 sinfo 查看 partition)
#SBATCH -N 1                    # Maximum number of nodes to be allocated
#SBATCH --gres=gpu:1            # specify GPU number
# 還有很多其他參數，可參考官方文件(https://slurm.schedmd.com/sbatch.html)

COMMON_ARGS = --DR ../LAB4_Dataset --fast_train --store_visualization --num_workers 1 --per_save 10
NO_TFR_ARGS = --tfr 0 --tfr_d_step 0

export LD_PRELOAD=/lib/x86_64-linux-gnu/libcuda.so.1

cd Lab4_template
srun python3 Trainer.py --DR ../LAB4_Dataset --fast_train --store_visualization --num_workers 1 --per_save 10 --save_root ../checkpoints/tfr=0.0/kl=None --tfr 0 --tfr_d_step 0 --kl_anneal_type Without
srun python3 Trainer.py --DR ../LAB4_Dataset --fast_train --store_visualization --num_workers 1 --per_save 10 --save_root ../checkpoints/tfr=0.0/kl=Monotonic --tfr 0 --tfr_d_step 0 --kl_anneal_type Monotonic
srun python3 Trainer.py --DR ../LAB4_Dataset --fast_train --store_visualization --num_workers 1 --per_save 10 --save_root ../checkpoints/tfr=0.0/kl=Cyclical --tfr 0 --tfr_d_step 0
srun python3 Trainer.py --DR ../LAB4_Dataset --fast_train --store_visualization --num_workers 1 --per_save 10 --save_root ../checkpoints/tfr=1.0/kl=None --kl_anneal_type Without
