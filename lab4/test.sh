#!/bin/sh
#SBATCH -A dlp_acct             # Account name (可透過 sshare 查看)
#SBATCH -p dlp_nodes            # Partition name (可透過 sinfo 查看 partition)
#SBATCH -N 1                    # Maximum number of nodes to be allocated
#SBATCH --gres=gpu:1            # specify GPU number
# 還有很多其他參數，可參考官方文件(https://slurm.schedmd.com/sbatch.html)

export LD_PRELOAD=/lib/x86_64-linux-gnu/libcuda.so.1

cd Lab4_template
srun python3 Tester.py --DR ../LAB4_Dataset --num_workers 1 --save_root ../data/tfr=0.0/kl=None --ckpt_path ../checkpoints/tfr=0.0/kl=None/epoch=60.ckpt --tfr 0 --tfr_d_step 0 --kl_anneal_type Without --make_gif
srun python3 Tester.py --DR ../LAB4_Dataset --num_workers 1 --save_root ../data/tfr=0.0/kl=Monotonic --ckpt_path ../checkpoints/tfr=0.0/kl=Monotonic/epoch=60.ckpt --tfr 0 --tfr_d_step 0 --kl_anneal_type Monotonic --make_gif
srun python3 Tester.py --DR ../LAB4_Dataset --num_workers 1 --save_root ../data/tfr=0.0/kl=Cyclical --ckpt_path ../checkpoints/tfr=0.0/kl=Cyclical/epoch=60.ckpt --tfr 0 --tfr_d_step 0 --make_gif
srun python3 Tester.py --DR ../LAB4_Dataset --num_workers 1 --save_root ../data/tfr=1.0/kl=None --ckpt_path ../checkpoints/tfr=1.0/kl=None/epoch=60.ckpt --kl_anneal_type Without --make_gif
