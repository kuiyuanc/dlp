#!/bin/sh
#SBATCH -A dlp_acct             # Account name (可透過 sshare 查看)
#SBATCH -p dlp_nodes            # Partition name (可透過 sinfo 查看 partition)
#SBATCH -N 1                    # Maximum number of nodes to be allocated
#SBATCH --gres=gpu:1            # specify GPU number
# 還有很多其他參數，可參考官方文件(https://slurm.schedmd.com/sbatch.html)

export LD_PRELOAD=/lib/x86_64-linux-gnu/libcuda.so.1 # for CUDA driver API

srun python3 inpainting.py --num_workers 1 --load-transformer-ckpt-path saved_models/drop/47_1.3327.pth --sweet-spot 8 --total-iter 8 --mask-func linear
srun python3 faster-pytorch-fid/fid_score_gpu.py --predicted-path test_results --device cuda:0 --num-workers 1 --gtcsv-path faster-pytorch-fid/test_gt.csv
srun python3 inpainting.py --num_workers 1 --load-transformer-ckpt-path saved_models/drop/47_1.3327.pth --sweet-spot 8 --total-iter 8 --mask-func cosine
srun python3 faster-pytorch-fid/fid_score_gpu.py --predicted-path test_results --device cuda:0 --num-workers 1 --gtcsv-path faster-pytorch-fid/test_gt.csv
srun python3 inpainting.py --num_workers 1 --load-transformer-ckpt-path saved_models/drop/47_1.3327.pth --sweet-spot 8 --total-iter 8 --mask-func square
srun python3 faster-pytorch-fid/fid_score_gpu.py --predicted-path test_results --device cuda:0 --num-workers 1 --gtcsv-path faster-pytorch-fid/test_gt.csv
srun python3 inpainting.py --num_workers 1 --load-transformer-ckpt-path saved_models/drop/47_1.3327.pth --sweet-spot 8 --total-iter 8 --mask-func log
srun python3 faster-pytorch-fid/fid_score_gpu.py --predicted-path test_results --device cuda:0 --num-workers 1 --gtcsv-path faster-pytorch-fid/test_gt.csv
srun python3 inpainting.py --num_workers 1 --load-transformer-ckpt-path saved_models/drop/47_1.3327.pth --sweet-spot 8 --total-iter 8 --mask-func sqrt
srun python3 faster-pytorch-fid/fid_score_gpu.py --predicted-path test_results --device cuda:0 --num-workers 1 --gtcsv-path faster-pytorch-fid/test_gt.csv
srun python3 inpainting.py --num_workers 1 --load-transformer-ckpt-path saved_models/drop/47_1.3327.pth --sweet-spot 10 --total-iter 10 --mask-func linear
srun python3 faster-pytorch-fid/fid_score_gpu.py --predicted-path test_results --device cuda:0 --num-workers 1 --gtcsv-path faster-pytorch-fid/test_gt.csv
srun python3 inpainting.py --num_workers 1 --load-transformer-ckpt-path saved_models/drop/47_1.3327.pth --sweet-spot 10 --total-iter 10 --mask-func cosine
srun python3 faster-pytorch-fid/fid_score_gpu.py --predicted-path test_results --device cuda:0 --num-workers 1 --gtcsv-path faster-pytorch-fid/test_gt.csv
srun python3 inpainting.py --num_workers 1 --load-transformer-ckpt-path saved_models/drop/47_1.3327.pth --sweet-spot 10 --total-iter 10 --mask-func square
srun python3 faster-pytorch-fid/fid_score_gpu.py --predicted-path test_results --device cuda:0 --num-workers 1 --gtcsv-path faster-pytorch-fid/test_gt.csv
srun python3 inpainting.py --num_workers 1 --load-transformer-ckpt-path saved_models/drop/47_1.3327.pth --sweet-spot 10 --total-iter 10 --mask-func log
srun python3 faster-pytorch-fid/fid_score_gpu.py --predicted-path test_results --device cuda:0 --num-workers 1 --gtcsv-path faster-pytorch-fid/test_gt.csv
srun python3 inpainting.py --num_workers 1 --load-transformer-ckpt-path saved_models/drop/47_1.3327.pth --sweet-spot 10 --total-iter 10 --mask-func sqrt
srun python3 faster-pytorch-fid/fid_score_gpu.py --predicted-path test_results --device cuda:0 --num-workers 1 --gtcsv-path faster-pytorch-fid/test_gt.csv
srun python3 inpainting.py --num_workers 1 --load-transformer-ckpt-path saved_models/drop/47_1.3327.pth --sweet-spot 12 --total-iter 12 --mask-func linear
srun python3 faster-pytorch-fid/fid_score_gpu.py --predicted-path test_results --device cuda:0 --num-workers 1 --gtcsv-path faster-pytorch-fid/test_gt.csv
srun python3 inpainting.py --num_workers 1 --load-transformer-ckpt-path saved_models/drop/47_1.3327.pth --sweet-spot 12 --total-iter 12 --mask-func cosine
srun python3 faster-pytorch-fid/fid_score_gpu.py --predicted-path test_results --device cuda:0 --num-workers 1 --gtcsv-path faster-pytorch-fid/test_gt.csv
srun python3 inpainting.py --num_workers 1 --load-transformer-ckpt-path saved_models/drop/47_1.3327.pth --sweet-spot 12 --total-iter 12 --mask-func square
srun python3 faster-pytorch-fid/fid_score_gpu.py --predicted-path test_results --device cuda:0 --num-workers 1 --gtcsv-path faster-pytorch-fid/test_gt.csv
srun python3 inpainting.py --num_workers 1 --load-transformer-ckpt-path saved_models/drop/47_1.3327.pth --sweet-spot 12 --total-iter 12 --mask-func log
srun python3 faster-pytorch-fid/fid_score_gpu.py --predicted-path test_results --device cuda:0 --num-workers 1 --gtcsv-path faster-pytorch-fid/test_gt.csv
srun python3 inpainting.py --num_workers 1 --load-transformer-ckpt-path saved_models/drop/47_1.3327.pth --sweet-spot 12 --total-iter 12 --mask-func sqrt
srun python3 faster-pytorch-fid/fid_score_gpu.py --predicted-path test_results --device cuda:0 --num-workers 1 --gtcsv-path faster-pytorch-fid/test_gt.csv
srun python3 inpainting.py --num_workers 1 --load-transformer-ckpt-path saved_models/drop/47_1.3327.pth --sweet-spot 8 --total-iter 12 --mask-func linear
srun python3 faster-pytorch-fid/fid_score_gpu.py --predicted-path test_results --device cuda:0 --num-workers 1 --gtcsv-path faster-pytorch-fid/test_gt.csv
srun python3 inpainting.py --num_workers 1 --load-transformer-ckpt-path saved_models/drop/47_1.3327.pth --sweet-spot 8 --total-iter 12 --mask-func cosine
srun python3 faster-pytorch-fid/fid_score_gpu.py --predicted-path test_results --device cuda:0 --num-workers 1 --gtcsv-path faster-pytorch-fid/test_gt.csv
srun python3 inpainting.py --num_workers 1 --load-transformer-ckpt-path saved_models/drop/47_1.3327.pth --sweet-spot 8 --total-iter 12 --mask-func square
srun python3 faster-pytorch-fid/fid_score_gpu.py --predicted-path test_results --device cuda:0 --num-workers 1 --gtcsv-path faster-pytorch-fid/test_gt.csv
srun python3 inpainting.py --num_workers 1 --load-transformer-ckpt-path saved_models/drop/47_1.3327.pth --sweet-spot 8 --total-iter 12 --mask-func log
srun python3 faster-pytorch-fid/fid_score_gpu.py --predicted-path test_results --device cuda:0 --num-workers 1 --gtcsv-path faster-pytorch-fid/test_gt.csv
srun python3 inpainting.py --num_workers 1 --load-transformer-ckpt-path saved_models/drop/47_1.3327.pth --sweet-spot 8 --total-iter 12 --mask-func sqrt
srun python3 faster-pytorch-fid/fid_score_gpu.py --predicted-path test_results --device cuda:0 --num-workers 1 --gtcsv-path faster-pytorch-fid/test_gt.csv
srun python3 inpainting.py --num_workers 1 --load-transformer-ckpt-path saved_models/drop/47_1.3327.pth --sweet-spot 10 --total-iter 12 --mask-func linear
srun python3 faster-pytorch-fid/fid_score_gpu.py --predicted-path test_results --device cuda:0 --num-workers 1 --gtcsv-path faster-pytorch-fid/test_gt.csv
srun python3 inpainting.py --num_workers 1 --load-transformer-ckpt-path saved_models/drop/47_1.3327.pth --sweet-spot 10 --total-iter 12 --mask-func cosine
srun python3 faster-pytorch-fid/fid_score_gpu.py --predicted-path test_results --device cuda:0 --num-workers 1 --gtcsv-path faster-pytorch-fid/test_gt.csv
srun python3 inpainting.py --num_workers 1 --load-transformer-ckpt-path saved_models/drop/47_1.3327.pth --sweet-spot 10 --total-iter 12 --mask-func square
srun python3 faster-pytorch-fid/fid_score_gpu.py --predicted-path test_results --device cuda:0 --num-workers 1 --gtcsv-path faster-pytorch-fid/test_gt.csv
srun python3 inpainting.py --num_workers 1 --load-transformer-ckpt-path saved_models/drop/47_1.3327.pth --sweet-spot 10 --total-iter 12 --mask-func log
srun python3 faster-pytorch-fid/fid_score_gpu.py --predicted-path test_results --device cuda:0 --num-workers 1 --gtcsv-path faster-pytorch-fid/test_gt.csv
srun python3 inpainting.py --num_workers 1 --load-transformer-ckpt-path saved_models/drop/47_1.3327.pth --sweet-spot 10 --total-iter 12 --mask-func sqrt
srun python3 faster-pytorch-fid/fid_score_gpu.py --predicted-path test_results --device cuda:0 --num-workers 1 --gtcsv-path faster-pytorch-fid/test_gt.csv
