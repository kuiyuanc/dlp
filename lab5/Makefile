ARGS =

CARTPOLE_CONFIG = --batch-size 64 --memory-size 10000 --lr 0.001  --gamma 0.99 --epsilon-decay 0.995    --epsilon-min 0.01 --target-update-frequency 10   --replay-start-size 1000  --max-episode-steps 1000 --num-episodes 500
ATARI_CONFIG    = --batch-size 32 --memory-size 50000 --lr 0.0001 --gamma 0.99 --epsilon-decay 0.9999 --epsilon-min 0.01 --target-update-frequency 1000 --replay-start-size 25000 --train-per-step 4       --num-episodes 1500
ENHANCE_CONFIG  = --double --alpha 0.6 --beta 0.4 --epsilon 0.01 --return-steps 3
EXTRA_CONFIG    = --skip-frames 4

train1:
	pipenv run py src/train.py --task 1 $(CARTPOLE_CONFIG) $(ARGS)
	pipenv run py src/train.py --task 1 $(CARTPOLE_CONFIG) $(ARGS) --double
	pipenv run py src/train.py --task 1 $(CARTPOLE_CONFIG) $(ARGS) --reward-scaling -1.0

train2:
	pipenv run py src/train.py --task 2 $(ATARI_CONFIG) $(ARGS)

train3:
	pipenv run py src/train.py --task 3 $(ATARI_CONFIG) $(ENHANCE_CONFIG) $(EXTRA_CONFIG) $(ARGS)
	pipenv run py src/train.py --task 3 $(ATARI_CONFIG) --double --alpha 0.0                           --return-steps 1 $(EXTRA_CONFIG) $(ARGS)
	pipenv run py src/train.py --task 3 $(ATARI_CONFIG)          --alpha 0.6 --beta 0.4 --epsilon 0.01 --return-steps 1 $(EXTRA_CONFIG) $(ARGS)
	pipenv run py src/train.py --task 3 $(ATARI_CONFIG)          --alpha 0.0                           --return-steps 3 $(EXTRA_CONFIG) $(ARGS)

test1:
	pipenv run py src/test_model.py $(ARGS) --task 1

test2:
	pipenv run py src/test_model.py $(ARGS) --task 2

test3:
	pipenv run py src/test_model.py $(ARGS) --task 3

train: train1 train2 train3

test:
	pipenv run py src/test_model.py --task 0
