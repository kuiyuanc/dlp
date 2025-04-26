PY = uv run
TRAIN = $(PY) src/train.py
TEST = $(PY) src/test_model.py

SBATCH = sbatch -A dlp_acct -p dlp_nodes -N 1 --gres=gpu:1

args =

API_KEY         = --wandb-api-key 65bab6fbcc749c9a7ca8f47f40d6ab7d2f5705cb
CARTPOLE_CONFIG = --batch-size 128 --memory-size 4096 --learning-rate 0.001 --epsilon-decay 0.995 --epsilon-min 0.01 --target-update-frequency 16 --replay-start-size 512 --max-episode-steps 1000 --tau 1 --backup-frequency 50 --early-stop 480 --num-episodes 500
ENHANCE_CONFIG  = --double
SWEEP_CONFIG    = --sweep

train1:
ifeq ($(OS),Windows_NT)
	$(TRAIN) --task 1 $(CARTPOLE_CONFIG) $(args) $(API_KEY)
else
	$(SBATCH) train.sh --task 1 $(CARTPOLE_CONFIG) $(args) $(API_KEY)
endif

train2:
ifeq ($(OS),Windows_NT)
	$(TRAIN) --task 2 $(args) --tau 1 $(API_KEY)
else
	$(SBATCH) train.sh --task 2 $(args) --tau 1 $(API_KEY)
endif

train3:
ifeq ($(OS),Windows_NT)
	$(TRAIN) --task 3 $(ENHANCE_CONFIG) $(args) $(API_KEY)
else
	$(SBATCH) train.sh --task 3 $(ENHANCE_CONFIG) $(args) $(API_KEY)
endif

sweep1:
ifeq ($(OS),Windows_NT)
	$(TRAIN) --task 1 $(CARTPOLE_CONFIG) $(SWEEP_CONFIG) $(args) $(API_KEY)
else
	$(SBATCH) train.sh --task 1 $(CARTPOLE_CONFIG) $(SWEEP_CONFIG) $(args) $(API_KEY)
endif

sweep2:
ifeq ($(OS),Windows_NT)
	$(TRAIN) --task 2 $(SWEEP_CONFIG) $(args) $(API_KEY)
else
	$(SBATCH) train.sh --task 2 $(SWEEP_CONFIG) $(args) $(API_KEY)
endif

sweep3:
ifeq ($(OS),Windows_NT)
	$(TRAIN) --task 3 $(SWEEP_CONFIG) $(args) $(API_KEY)
else
	$(SBATCH) train.sh --task 3 $(SWEEP_CONFIG) $(args) $(API_KEY)
endif

test1:
ifeq ($(OS),Windows_NT)
	$(TEST) --task 1 $(args)
else
	$(SBATCH) test.sh --task 1 $(args)
endif

test2:
ifeq ($(OS),Windows_NT)
	$(TEST) --task 2 $(args)
else
	$(SBATCH) test.sh --task 2 $(args)
endif

test3:
ifeq ($(OS),Windows_NT)
	$(TEST) --task 3 $(args)
else
	$(SBATCH) test.sh --task 3 $(args)
endif

train: train1 train2 train3

test:
ifeq ($(OS),Windows_NT)
	$(TEST) --task 0 $(args)
else
	$(SBATCH) test.sh --task 0 $(args)
endif

conn:
	ssh chenky2003@hpclogin02.cs.nycu.edu.tw

up:
	source /opt/conda/miniconda3/bin/activate
	conda activate dlp_lab5
