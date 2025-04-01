# NYCU Deep Learning Labs 2025 Spring


## NYCU CSIT HPC Service

- Reference
    - [使用說明 - 國立陽明交通大學資工系資訊中心](https://it.cs.nycu.edu.tw/workstation-guide)
    - [GPU 機房使用者教學 - 國立陽明交通大學資工系資訊中心](https://it.cs.nycu.edu.tw/gpu-room-tutor#)

- Login
    ```powershell
    ssh <username>@hpclogin[01-03].cs.nycu.edu.tw
    ```

- Manage files
    ```powershell
    scp -r <username>@<hostname>:<source-file-path> <destination-file-path>
    scp -r <source-file-path> <username>@<hostname>:<destination-file-path>
    ```

- Activate environment
    ```bash
    bash
    source /opt/conda/miniconda3/bin/activate
    conda activate dlp_lab<number>
    ```

- Submit job
    ```
    sbatch -A dlp_acct -p dlp_nodes --gres=gpu:1 --nodelist=cmpt100 <script>.sh
    ```

    - `<script>.sh`
        ```sh
        #!/bin/sh
        #SBATCH -A dlp_acct             # Account name (可透過 sshare 查看)
        #SBATCH -p dlp_nodes            # Partition name (可透過 sinfo 查看 partition)
        #SBATCH -N 1                    # Maximum number of nodes to be allocated
        #SBATCH --gres=gpu:1            # specify GPU number
        # 還有很多其他參數，可參考官方文件(https://slurm.schedmd.com/sbatch.html)

        export LD_PRELOAD=/lib/x86_64-linux-gnu/libcuda.so.1 # for CUDA driver API

        srun python3 <task>.py
        ```
