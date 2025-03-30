# dlp
NYCU 2025-spring DLP Labs


## NYCU CSIT HPC Service

```bash=
bash
source /opt/conda/miniconda3/bin/activate
conda activate dlp_lab<number>
sbatch <script>.sh
```

```sh=
#!/bin/sh
SBATCH -A general              # Account name (可透過 sshare 查看)
SBATCH -p dlp_nodes            # Partition name (可透過 sinfo 查看 partition)
SBATCH -N 1                    # Maximum number of nodes to be allocated
SBATCH --gres=gpu:1            # specify GPU number
# 還有很多其他參數，可參考官方文件(https://slurm.schedmd.com/sbatch.html)

export LD_PRELOAD=/lib/x86_64-linux-gnu/libcuda.so.1 # for CUDA driver API

srun python3 <task>.py
```
