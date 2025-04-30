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
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.learning_rate)

        if args.model_path:
            state_dict = torch.load(args.model_path, map_location=self.device)
            self.q_net.load_state_dict(state_dict["q_net"])
            self.target_net.load_state_dict(state_dict["target_net"])

        if self.device.type.startswith("xpu"):
            import intel_extension_for_pytorch as ipex

            self.q_net, self.optimizer = ipex.optimize(self.q_net, torch.float32, self.optimizer, inplace=True)

        self.best_reward = -21  # Initilized to 0 for CartPole and to -21 for Pong

    def run(self, episodes: int = 1000) -> None:
        ewma_return = self.best_reward

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
                state = torch.from_numpy(state).detach_().float()
                action = torch.tensor(action, dtype=torch.int64)
                reward_tensor = torch.tensor(reward, dtype=torch.float32)
                next_state_tensor = torch.from_numpy(next_state).detach_().float()
                done_tensor = torch.tensor(done, dtype=torch.bool)
                self.memory.append((state, action, reward_tensor, next_state_tensor, done_tensor))

                for _ in range(self.train_per_step):
                    self.train()

                state = next_state
                total_reward += float(reward)
                self.env_count += 1
                step_count += 1

            wandb.log(
                {
                    "Episodes": ep,
                    "Train/Episodic Length": step_count,
                    "Train/Return": total_reward,
                    "Environment Steps": self.env_count,
                    "Train/Steps": self.train_count,
                    "Train/Epsilon": self.epsilon,
                }
            )

            if ep % self.backup_frequency == 0:
                model_path = os.path.join(self.save_dir, f"model_ep{ep}.pt")
                torch.save({"q_net": self.q_net.state_dict(), "target_net": self.target_net.state_dict()}, model_path)
                with open(Path(self.save_dir, f"model_ep{ep}.pkl"), "wb") as f:
                    pickle.dump((self.epsilon, self.env_count, self.train_count, self.best_reward, ep + 1), f)

            if ep % self.eval_frequency == 0:
                eval_steps = int(1 + np.log2(self.eval_frequency))
                eval_reward, episode_len = map(np.mean, zip(*(self.evaluate() for _ in range(eval_steps))))
                if eval_reward >= self.best_reward:
                    self.best_reward = eval_reward
                    model_path = os.path.join(self.save_dir, "best_model.pt")
                    torch.save({"q_net": self.q_net.state_dict()}, model_path)

                ewma_return = 0.05 * eval_reward + (1 - 0.05) * ewma_return
                wandb.log(
                    {
                        "Episodes": ep,
                        "Environment Steps": self.env_count,
                        "Train/Steps": self.train_count,
                        "Eval/Return": eval_reward,
                        "Eval/Episodic Length": episode_len,
                        "EWMA Return": ewma_return,
                    }
                )

                if ewma_return > self.early_stop:
                    break

    def evaluate(self):
        obs, _ = self.test_env.reset()
        state = self.preprocessor.reset(obs)
        done = False
        total_reward = 0
        step_count = 0

        while not done:
            state_tensor = torch.from_numpy(np.asarray(state)).float().unsqueeze_(0).to(self.device)
            with torch.no_grad():
                action = self.q_net(state_tensor).argmax().item()
            next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
            done = terminated or truncated
            total_reward += float(reward)
            state = self.preprocessor.step(next_obs)
            step_count += 1

        return total_reward, step_count
