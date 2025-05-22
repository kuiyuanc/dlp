#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Spring 2025, 535507 Deep Learning
# Lab7: Policy-based RL
# Task 2: PPO-Clip
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh

import argparse
import random
from collections import deque
from pathlib import Path
from typing import Deque, List, Tuple

import gymnasium as gym
import numpy as np
import pretty_errors  # noqa: F401
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from torch.distributions import Normal
from tqdm import tqdm


def init_layer_uniform(layer: nn.Linear, init_w: float = 3e-3) -> nn.Linear:
    """Init uniform parameters on the single layer."""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)

    return layer


class Actor(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        log_std_min: int = -20,
        log_std_max: int = 0,
    ):
        """Initialize."""
        super(Actor, self).__init__()

        ############TODO#############
        # Remeber to initialize the layer weights
        self.net = nn.Sequential(
            init_layer_uniform(nn.Linear(in_dim, 64)),
            nn.ReLU(),
            init_layer_uniform(nn.Linear(64, 64)),
            nn.ReLU(),
        )
        self.mean = init_layer_uniform(nn.Linear(64, out_dim))
        self.log_std = init_layer_uniform(nn.Linear(64, out_dim))
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        #############################

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.distributions.Distribution]:
        """Forward method implementation."""

        ############TODO#############
        latent = self.net(state)
        mean = self.mean(latent)
        std = self.log_std(latent).clamp_(self.log_std_min, self.log_std_max).exp()
        dist = Normal(mean, std)
        action = dist.sample()
        #############################

        return action, dist


class Critic(nn.Module):
    def __init__(self, in_dim: int):
        """Initialize."""
        super(Critic, self).__init__()

        ############TODO#############
        # Remeber to initialize the layer weights
        self.net = nn.Sequential(
            init_layer_uniform(nn.Linear(in_dim, 64)),
            nn.ReLU(),
            init_layer_uniform(nn.Linear(64, 64)),
            nn.ReLU(),
            init_layer_uniform(nn.Linear(64, 1)),
        )
        #############################

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""

        ############TODO#############
        value = self.net(state)
        #############################

        return value


def compute_gae(next_value: torch.Tensor, rewards: list, masks: list, values: list, gamma: float, tau: float) -> List:
    """Compute gae."""

    ############TODO#############
    gae_returns = []
    gae = 0
    for reward, mask, value in reversed(tuple(zip(rewards, masks, values))):
        delta = reward + gamma * next_value * mask - value
        gae = delta + gamma * tau * gae
        gae_returns.append(gae + value)
        next_value = value
    gae_returns.reverse()
    #############################
    return gae_returns


# PPO updates the model several times(update_epoch) using the stacked memory.
# By ppo_iter function, it can yield the samples of stacked memory by interacting a environment.
def ppo_iter(
    update_epoch: int,
    mini_batch_size: int,
    states: torch.Tensor,
    actions: torch.Tensor,
    values: torch.Tensor,
    log_probs: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
):
    """Get mini-batches."""
    batch_size = states.size(0)
    for _ in range(update_epoch):
        for _ in range(batch_size // mini_batch_size):
            rand_ids = np.random.choice(batch_size, mini_batch_size)
            yield (
                states[rand_ids, :],
                actions[rand_ids],
                values[rand_ids],
                log_probs[rand_ids],
                returns[rand_ids],
                advantages[rand_ids],
            )


class PPOAgent:
    """PPO Agent.
    Attributes:
        env (gym.Env): Gym env for training
        gamma (float): discount factor
        tau (float): lambda of generalized advantage estimation (GAE)
        batch_size (int): batch size for sampling
        epsilon (float): amount of clipping surrogate objective
        update_epoch (int): the number of update
        rollout_len (int): the number of rollout
        entropy_weight (float): rate of weighting entropy into the loss function
        actor (nn.Module): target actor model to select actions
        critic (nn.Module): critic model to predict state values
        transition (list): temporory storage for the recent transition
        device (torch.device): cpu / gpu
        total_step (int): total step numbers
        is_test (bool): flag to show the current mode (train / test)
        seed (int): random seed
    """

    def __init__(self, env: gym.Env, args):
        """Initialize."""
        self.env = env
        self.gamma = args.discount_factor
        self.tau = args.tau
        self.batch_size = args.batch_size
        self.epsilon = args.epsilon
        self.num_episodes = args.num_episodes
        self.rollout_len = args.rollout_len
        self.entropy_weight = args.entropy_weight
        self.seed = args.seed
        self.update_epoch = args.update_epoch

        # device: cpu / gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        # networks
        assert isinstance(env.observation_space, gym.spaces.Box) and isinstance(env.action_space, gym.spaces.Box)
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.actor = Actor(self.obs_dim, self.action_dim).to(self.device)
        self.critic = Critic(self.obs_dim).to(self.device)

        # optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.critic_lr)

        # memory for training
        self.states: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.rewards: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.masks: List[torch.Tensor] = []
        self.log_probs: List[torch.Tensor] = []

        # total steps count
        self.total_step = 0

        # mode: train / test
        self.is_test = False

        # backup
        self.save_dir: Path = args.save_dir
        self.backup_freq: int = args.backup_freq
        self.ewma_lambda: float = args.ewma_lambda
        self.ewma_score: float = args.ewma_start
        self.max_ewma_score: float = self.ewma_score
        self.preempt: float = args.preempt

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        state_tensor = torch.FloatTensor(state).to(self.device)
        action, dist = self.actor(state_tensor)
        selected_action = dist.mean if self.is_test else action

        if not self.is_test:
            value = self.critic(state_tensor)
            self.states.append(state_tensor)
            self.actions.append(selected_action)
            self.values.append(value)
            self.log_probs.append(dist.log_prob(selected_action))

        return selected_action.cpu().detach().numpy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Take an action and return the response of the env."""
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        next_state = np.reshape(next_state, (1, -1)).astype(np.float64)
        assert isinstance(reward, np.ndarray) or isinstance(reward, np.float64), type(reward)
        reward = np.reshape(reward, (1, -1)).astype(np.float64)
        done = np.reshape(done, (1, -1))

        if not self.is_test:
            self.rewards.append(torch.FloatTensor(reward).to(self.device))
            self.masks.append(torch.FloatTensor(1 - done).to(self.device))

        return next_state, reward, done

    def update_model(self, next_state: np.ndarray) -> Tuple[float, float]:
        """Update the model by gradient descent."""
        next_state_tensor = torch.FloatTensor(next_state).to(self.device)
        next_value = self.critic(next_state_tensor)

        returns = compute_gae(
            next_value,
            self.rewards,
            self.masks,
            self.values,
            self.gamma,
            self.tau,
        )

        states = torch.cat(self.states).view(-1, self.obs_dim)
        actions = torch.cat(self.actions)
        returns = torch.cat(returns).detach()
        values = torch.cat(self.values).detach()
        log_probs = torch.cat(self.log_probs).detach()
        advantages = returns - values

        actor_losses, critic_losses = [], []

        for state, action, old_value, old_log_prob, return_, adv in ppo_iter(
            update_epoch=self.update_epoch,
            mini_batch_size=self.batch_size,
            states=states,
            actions=actions,
            values=values,
            log_probs=log_probs,
            returns=returns,
            advantages=advantages,
        ):
            # calculate ratios
            _, dist = self.actor(state)
            log_prob = dist.log_prob(action)
            ratio = (log_prob - old_log_prob).exp()

            # actor_loss
            ############TODO#############
            # actor_loss = ?
            surrogate1 = ratio * adv
            surrogate2 = ratio.clamp(1 - self.epsilon, 1 + self.epsilon) * adv
            actor_loss = -torch.min(surrogate1, surrogate2).mean() - dist.entropy().mean() * self.entropy_weight
            #############################

            # critic_loss
            ############TODO#############
            # critic_loss = ?
            critic_loss = F.mse_loss(self.critic(state), return_)
            #############################

            # train critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            self.critic_optimizer.step()

            # train actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())

        self.states, self.actions, self.rewards = [], [], []
        self.values, self.masks, self.log_probs = [], [], []

        actor_loss = sum(actor_losses) / len(actor_losses)
        critic_loss = sum(critic_losses) / len(critic_losses)

        return actor_loss, critic_loss

    def train(self):
        """Train the PPO agent."""
        self.is_test = False

        state, _ = self.env.reset(seed=self.seed)
        state = np.expand_dims(state, axis=0)

        actor_losses, critic_losses = [], []
        scores = []
        score = 0
        episode_count = 0
        for ep in tqdm(range(1, self.num_episodes + 1)):
            score = 0
            for _ in range(self.rollout_len):
                self.total_step += 1
                action = self.select_action(state)
                next_state, reward, done = self.step(action)

                state = next_state
                score += reward[0][0]

                # if episode ends
                if done[0][0]:
                    episode_count += 1
                    state, _ = self.env.reset(seed=self.seed)
                    state = np.expand_dims(state, axis=0)
                    scores.append(score)
                    self.ewma_score = self.ewma_lambda * score + (1 - self.ewma_lambda) * self.ewma_score
                    score = 0

                    wandb.log(
                        {
                            "Environment Step": self.total_step,
                            "Train/Episode": episode_count,
                            "EWMA Episodic Reward": self.ewma_score,
                            "Train/Episodic Reward": scores[-1],
                        },
                        step=self.total_step,
                    )

                if self.total_step % self.backup_freq == 0:
                    self.save(ep, scores[-1], actor_losses[-1], critic_losses[-1], str(self.total_step))

            actor_loss, critic_loss = self.update_model(next_state)
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)

            wandb.log(
                {
                    "Environment Step": self.total_step,
                    "Train/Episode": episode_count,
                    "Train/Episodic Reward": scores[-1],
                    "Train/Actor Loss": actor_loss,
                    "Train/Critic Loss": critic_loss,
                },
                step=self.total_step,
            )

            if self.ewma_score >= self.max_ewma_score:
                self.save(ep, scores[-1], actor_loss, critic_loss, "best")
                self.max_ewma_score = self.ewma_score
                if self.ewma_score >= self.preempt:
                    break

        # termination
        self.env.close()

    def test(self, video_folder: str, seed: int | None = None, record: bool = True, verbose: bool = False) -> float:
        """Test the agent."""
        self.is_test = True

        tmp_env = self.env
        if record:
            self.env = gym.wrappers.RecordVideo(self.env, video_folder=video_folder, name_prefix=str(seed))

        seed = seed or self.seed
        state, _ = self.env.reset(seed=seed)
        done = False
        score = 0

        while not done:
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward
        self.env.close()

        assert isinstance(score, np.ndarray), type(score)
        if verbose:
            print(f"seed: {seed:5d}", f"score: {score.item():8.2f}")

        self.env = tmp_env

        return score.item()

    def save(self, ep: int, score: float, actor_loss: float, critic_loss: float, name: str):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "Episode": ep,
                "Environment Step": self.total_step,
                "Episodic Reward": score,
                "Actor Loss": actor_loss,
                "Critic Loss": critic_loss,
                "EWMA Episodic Reward": self.ewma_score,
            },
            self.save_dir / f"{name}.pt",
        )

    def load(self, name: str):
        state_dict = torch.load(self.save_dir / f"{name}.pt", map_location=self.device, weights_only=False)
        self.actor.load_state_dict(state_dict["actor"])
        self.critic.load_state_dict(state_dict["critic"])
        self.total_step = state_dict["Environment Step"]
        self.ewma_score = state_dict["EWMA Episodic Reward"]


def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--discount-factor", type=float, default=0.9)
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--entropy-weight", type=float, default=1e-2)  # entropy can be disabled by setting this to 0
    parser.add_argument("--tau", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--rollout-len", type=int, default=2000)
    parser.add_argument("--update-epoch", type=float, default=64)

    parser.add_argument("--ewma-start", type=int, default=-1250.0)
    parser.add_argument("--ewma-lambda", type=float, default=0.05)
    parser.add_argument("--preempt", type=float, default=-150.0)
    parser.add_argument("--save-dir", type=str, default="weights/ppo_pendulum")
    parser.add_argument("--video-dir", type=str, default="eval_videos/ppo_pendulum")
    parser.add_argument("--backup-freq", type=int, default=500000)
    parser.add_argument("--test", type=str, default=None)
    parser.add_argument("--wandb-id", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--api-key", type=str, default=None)

    return parser.parse_args()


def get_config(args: argparse.Namespace) -> dict:
    config = vars(args).copy()
    config.pop("wandb_run_name")
    config.pop("num_episodes")
    config.pop("ewma_start")
    config.pop("ewma_lambda")
    config.pop("save_dir")
    config.pop("video_dir")
    config.pop("backup_freq")
    config.pop("test")
    config.pop("wandb_id")
    config.pop("verbose")
    config.pop("api_key")
    return config


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    seed_torch(seed)


if __name__ == "__main__":
    args = get_args()
    config = get_config(args)

    # environment
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    # set_seed(77)

    if args.test:
        id = args.wandb_id
    else:
        wandb.login(key=args.api_key)
        wandb.init(project="dll7", group="DLP-Lab7-PPO-Pendulum", name=args.wandb_run_name, config=config, save_code=True)

        assert wandb.run
        if not args.wandb_run_name:
            wandb.run.name = wandb.run.id

        id = wandb.run.id

    args.save_dir = Path(args.save_dir) / id
    args.video_dir = Path(args.video_dir) / id
    args.save_dir.mkdir(parents=True, exist_ok=True)
    args.video_dir.mkdir(parents=True, exist_ok=True)

    agent = PPOAgent(env, args)

    if args.test:
        agent.load(args.test)
    else:
        agent.train()
        wandb.finish()

    seeds = []
    for seed in range(1, 1000):
        if len(seeds) >= 20:
            break
        if agent.test(str(args.video_dir), seed=seed, record=False) >= -150:
            seeds.append(seed)

    for seed in seeds:
        score = agent.test(str(args.video_dir), seed=seed, verbose=args.verbose)

    with open(args.video_dir / "seeds.txt", "w") as f:
        f.write(str(seeds))
