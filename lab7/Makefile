.PHONY: train train1 train2 train3

train1:
	uv run src/a2c_pendulum.py

#	uv run src/a2c_pendulum.py --entropy-weight 0.0
#	uv run src/a2c_pendulum.py --entropy-weight 0.0025
#	uv run src/a2c_pendulum.py --entropy-weight 0.04

train2:
	uv run src/ppo_pendulum.py

#	uv run src/ppo_pendulum.py --epsilon 0.0
#	uv run src/ppo_pendulum.py --epsilon 0.05
#	uv run src/ppo_pendulum.py --epsilon 0.8

#	uv run src/ppo_pendulum.py --entropy-weight 0.0
#	uv run src/ppo_pendulum.py --entropy-weight 0.0025
#	uv run src/ppo_pendulum.py --entropy-weight 0.04

train3:
	uv run src/ppo_walker.py

#	uv run src/ppo_walker.py --epsilon 0.1
#	uv run src/ppo_walker.py --epsilon 0.4

#	uv run src/ppo_walker.py --entropy-weight 0.0
#	uv run src/ppo_walker.py --entropy-weight 0.005
#	uv run src/ppo_walker.py --entropy-weight 0.02

train: train1 train2 train3

test1:
#	uv run src/a2c_pendulum.py --test __name__ --wandb-id __id__ --verbose

test2:
#	uv run src/ppo_pendulum.py --test __name__ --wandb-id __id__ --verbose

test3:
#	uv run src/ppo_walker.py --test __name__ --wandb-id __id__ --verbose

test: test1 test2 test3
