CARTPOLE_CONFIG = --batch-size 64 --memory-size 10000 --lr 0.001   --gamma 0.99 --epsilon-decay 0.995    --epsilon-min 0.01 --target-update-frequency 10   --replay-start-size 1000  --max-episode-steps 1000 --num-episodes 500
ATARI_CONFIG    = --batch-size 64 --memory-size 50000 --lr 0.00025 --gamma 0.99 --epsilon-decay 0.999995 --epsilon-min 0.01 --target-update-frequency 1000 --replay-start-size 50000 --train-per-step 4 --alpha 0.6 --beta 0.4 --epsilon 0.01 --anneal-steps 800000 --return-steps 3 --skip-frames 4 --num-episodes 500

train1:
	pipenv run py src/train.py --task 1 $(CARTPOLE_CONFIG)

train2:
	pipenv run py src/train.py --task 2 $(ATARI_CONFIG)

train3:
	pipenv run py src/train.py --task 3 $(ATARI_CONFIG)

test1:
	pipenv run py src/test_model.py --task 1

test2:
	pipenv run py src/test_model.py --task 2

test3:
	pipenv run py src/test_model.py --task 3

train: train1 train2 train3

test:
	pipenv run py src/test_model.py --task 0
