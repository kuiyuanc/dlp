export LD_PRELOAD=/lib/x86_64-linux-gnu/libcuda.so.1
cd Lab4_template; srun python3 Trainer.py --DR ../LAB4_Dataset --save_root ../checkpoints --fast_train --store_visualization --num_workers 1 --per_save 10
