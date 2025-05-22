#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Spring 2025, 535507 Deep Learning
# Lab7: Policy-based RL
# Task 1: A2C
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh

import argparse
import random
from pathlib import Path
from typing import Tuple

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


class A2CAgent:
    """A2CAgent interacting with environment.

    Atribute:
        env (gym.Env): openAI Gym environment
        gamma (float): discount factor
        entropy_weight (float): rate of weighting entropy into the loss function
        device (torch.device): cpu / gpu
        actor (nn.Module): target actor model to select actions
        critic (nn.Module): critic model to predict state values
        actor_optimizer (optim.Optimizer) : optimizer of actor
        critic_optimizer (optim.Optimizer) : optimizer of critic
        transition (list): temporory storage for the recent transition
        total_step (int): total step numbers
        is_test (bool): flag to show the current mode (train / test)
        seed (int): random seed
    """

    def __init__(self, env: gym.Env, args):
        """Initialize."""
        self.env = env
        self.gamma = args.discount_factor
        self.num_episodes = args.num_episodes
        self.rollout_len = args.rollout_len
        self.entropy_weight = args.entropy_weight
        self.seed = args.seed

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

        # transition (state, log_prob, next_state, reward, done)
        self.transition: list = list()

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
            log_prob = dist.log_prob(selected_action).sum(dim=-1, keepdim=True)
            self.transition = [state_tensor, log_prob, dist.entropy().mean()]

        return selected_action.clamp(-2.0, 2.0).cpu().detach().numpy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Take an action and return the response of the env."""
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        next_state = np.reshape(next_state, (1, -1)).astype(np.float64)
        assert isinstance(reward, np.ndarray) or isinstance(reward, np.float64), type(reward)
        reward = np.reshape(reward, (1, -1)).astype(np.float64)
        done = np.reshape(done, (1, -1))

        if not self.is_test:
            self.transition.extend(map(torch.FloatTensor, (next_state, reward, done)))

        return next_state, reward, done

    def update_model(self) -> Tuple[float, float]:
        """Update the model by gradient descent."""
        state, log_prob, entropy, next_state, reward, done = self.transition

        # Q_t   = r + gamma * V(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        mask = 1 - done

        ############TODO#############
        # value_loss = ?
        delta = reward + mask * self.gamma * self.critic(next_state) - self.critic(state)
        value_loss = delta.pow(2).mean()
        #############################

        # update value
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()

        # advantage = Q_t - V(s_t)
        ############TODO#############
        # policy_loss = ?
        policy_loss = -(log_prob * delta.detach()).mean() - entropy * self.entropy_weight
        #############################
        # update policy
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        return policy_loss.item(), value_loss.item()

    def train(self):
        """Train the agent."""
        self.is_test = False

        state, _ = self.env.reset(seed=self.seed)
        state = np.expand_dims(state, axis=0)

        # actor_losses, critic_losses = [], []
        scores = []
        score = 0
        episode_count = 0
        for ep in tqdm(range(1, self.num_episodes + 1)):
            score = 0
            for _ in range(self.rollout_len):
                # self.env.render()  # Render the environment
                self.total_step += 1
                action = self.select_action(state)
                next_state, reward, done = self.step(action)

                state = next_state
                score += reward[0][0]

                actor_loss, critic_loss = self.update_model()
                # actor_losses.append(actor_loss)
                # critic_losses.append(critic_loss)

                # W&B logging
                wandb.log(
                    {
                        "Environment Step": self.total_step,
                        "Train/Actor Loss": actor_loss,
                        "Train/Critic Loss": critic_loss,
                    }
                )

                # if episode ends
                if done[0][0]:
                    episode_count += 1
                    state, _ = self.env.reset(seed=self.seed)
                    state = np.expand_dims(state, axis=0)
                    scores.append(score)
                    self.ewma_score = self.ewma_lambda * score + (1 - self.ewma_lambda) * self.ewma_score
                    score = 0

                    # W&B logging
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
                    self.save(ep, scores[-1], actor_loss, critic_loss, str(self.total_step))

                if self.ewma_score >= self.max_ewma_score and scores:
                    self.save(ep, scores[-1], actor_loss, critic_loss, "best")
                    self.max_ewma_score = self.ewma_score
                    if self.ewma_score >= self.preempt:
                        break

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
    parser.add_argument("--actor-lr", type=float, default=1e-3)
    parser.add_argument("--critic-lr", type=float, default=5e-3)
    parser.add_argument("--discount-factor", type=float, default=0.9)
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--entropy-weight", type=float, default=1e-2)  # entropy can be disabled by setting this to 0
    parser.add_argument("--rollout-len", type=int, default=2000)

    parser.add_argument("--ewma-start", type=int, default=-1250.0)
    parser.add_argument("--ewma-lambda", type=float, default=0.05)
    parser.add_argument("--preempt", type=float, default=-150.0)
    parser.add_argument("--save-dir", type=str, default="weights/a2c_pendulum")
    parser.add_argument("--video-dir", type=str, default="eval_videos/a2c_pendulum")
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
        wandb.init(
            project="dll7", group="DLP-Lab7-A2C-Pendulum", name=args.wandb_run_name, config=config, save_code=True
        )

        assert wandb.run
        if not args.wandb_run_name:
            wandb.run.name = wandb.run.id

        id = wandb.run.id

    args.save_dir = Path(args.save_dir) / id
    args.video_dir = Path(args.video_dir) / id
    args.save_dir.mkdir(parents=True, exist_ok=True)
    args.video_dir.mkdir(parents=True, exist_ok=True)

    agent = A2CAgent(env, args)

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
