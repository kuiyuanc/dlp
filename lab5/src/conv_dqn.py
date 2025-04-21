# Spring 2025, 535507 Deep Learning
# Lab5: Value-based RL
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh

import argparse
import os
import pickle
from pathlib import Path

import ale_py
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from dqn import AtariPreprocessor, DQNAgent, init_weights

gym.register_envs(ale_py)


class ConvDQN(nn.Module):
    def __init__(self, num_actions: int, input_channels: int = 4):
        super(ConvDQN, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x / 255.0)


class ConvDQNAgent(DQNAgent):
    def __init__(self, args: argparse.Namespace, env_name: str = "ALE/Pong-v5"):
        super(ConvDQNAgent, self).__init__(args, env_name)

        self.preprocessor = AtariPreprocessor()

        self.q_net = ConvDQN(self.num_actions).to(self.device)
        self.q_net.apply(init_weights)
        self.target_net = ConvDQN(self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)

        if args.model_path:
            state_dict = torch.load(args.model_path, map_location=self.device)
            self.q_net.load_state_dict(state_dict["q_net"])
            self.target_net.load_state_dict(state_dict["target_net"])

        self.best_reward = -21  # Initilized to 0 for CartPole and to -21 for Pong

    def run(self, episodes: int = 1000) -> None:
        for ep in range(self.ep, episodes + 1):
            obs, _ = self.env.reset()

            state = self.preprocessor.reset(obs)
            done = False
            total_reward = 0
            step_count = 0

            while not done and step_count < self.max_episode_steps:
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                next_state = self.preprocessor.step(next_obs)
                self.memory.append((state, action, reward * self.reward_scaling, next_state, done))

                for _ in range(self.train_per_step):
                    self.train()

                state = next_state
                total_reward += float(reward)
                self.env_count += 1
                step_count += 1

            wandb.log(
                {
                    "Episode": ep,
                    "Train Episode Length": step_count,
                    "Total Reward": total_reward,
                    "Env Step Count": self.env_count,
                    "Update Count": self.train_count,
                    "Epsilon": self.epsilon,
                }
            )

            if ep % 10 == 0:
                model_path = os.path.join(self.save_dir, f"model_ep{ep}.pt")
                torch.save({"q_net": self.q_net.state_dict(), "target_net": self.target_net.state_dict()}, model_path)
                with open(Path(self.save_dir, f"model_ep{ep}.pkl"), "wb") as f:
                    pickle.dump((self.epsilon, self.env_count, self.train_count, self.best_reward, ep + 1), f)

                eval_reward, episode_len = self.evaluate()
                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    model_path = os.path.join(self.save_dir, "best_model.pt")
                    torch.save({"q_net": self.q_net.state_dict()}, model_path)

                wandb.log(
                    {
                        "Episode": ep,
                        "Env Step Count": self.env_count,
                        "Update Count": self.train_count,
                        "Eval Reward": eval_reward,
                        "Eval Episode Length": episode_len,
                    }
                )

    def evaluate(self):
        obs, _ = self.test_env.reset()
        state = self.preprocessor.reset(obs)
        done = False
        total_reward = 0
        step_count = 0

        while not done:
            state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                action = self.q_net(state_tensor).argmax().item()
            next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
            done = terminated or truncated
            total_reward += float(reward)
            state = self.preprocessor.step(next_obs)
            step_count += 1

        return total_reward, step_count
