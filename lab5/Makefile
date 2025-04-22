PY = pipenv run py
TRAIN = $(PY) src/train.py
TEST = $(PY) src/test_model.py

SBATCH = sbatch -A dlp_acct -p dlp_nodes --gres=gpu:1

ARGS =

CARTPOLE_CONFIG = --batch-size 64 --memory-size 10000 --lr 0.001  --gamma 0.99 --epsilon-decay 0.995  --epsilon-min 0.01 --target-update-frequency 10   --replay-start-size 1000  --max-episode-steps 1000 --num-episodes 500
ATARI_CONFIG    = --batch-size 32 --memory-size 50000 --lr 0.0001 --gamma 0.99 --epsilon-decay 0.9999 --epsilon-min 0.01 --target-update-frequency 1000 --replay-start-size 25000 --train-per-step 4       --num-episodes 1500
ENHANCE_CONFIG  = --double --alpha 0.6 --beta 0.4 --epsilon 0.01 --return-steps 3
EXTRA_CONFIG    = --skip-frames 4

train1:
ifeq ($(OS),Windows_NT)
	$(TRAIN) --task 1 $(CARTPOLE_CONFIG) $(ARGS)
else
	$(SBATCH) train.sh --task 1 $(CARTPOLE_CONFIG) $(ARGS)
endif

train2:
ifeq ($(OS),Windows_NT)
	$(TRAIN) --task 2 $(ATARI_CONFIG) $(ARGS)
else
	$(SBATCH) train.sh --task 2 $(ATARI_CONFIG) $(ARGS)
endif

train3:
ifeq ($(OS),Windows_NT)
	$(TRAIN) --task 3 $(ATARI_CONFIG) $(EXTRA_CONFIG) $(ARGS)
else
	$(SBATCH) train.sh --task 3 $(ATARI_CONFIG) $(EXTRA_CONFIG) $(ARGS)
endif

test1:
ifeq ($(OS),Windows_NT)
	$(TEST) --task 1 $(ARGS)
else
	$(SBATCH) test.sh --task 1 $(ARGS)
endif

test2:
ifeq ($(OS),Windows_NT)
	$(TEST) --task 2 $(ARGS)
else
	$(SBATCH) test.sh --task 2 $(ARGS)
endif

test3:
ifeq ($(OS),Windows_NT)
	$(TEST) --task 3 $(ARGS)
else
	$(SBATCH) test.sh --task 3 $(ARGS)
endif

train: train1 train2 train3

test:
ifeq ($(OS),Windows_NT)
	$(TEST) --task 0 $(ARGS)
else
	$(SBATCH) test.sh --task 0 $(ARGS)
endif

conn:
	ssh chenky2003@hpclogin02.cs.nycu.edu.tw

up:
	source /opt/conda/miniconda3/bin/activate
	conda activate dlp_lab5
