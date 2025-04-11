export LD_PRELOAD=/lib/x86_64-linux-gnu/libcuda.so.1
cd Lab4_template; srun python3 Tester.py --DR ../LAB4_Dataset --save_root ../data --ckpt_path ../checkpoints/epoch=70.ckpt --num_workers 1
