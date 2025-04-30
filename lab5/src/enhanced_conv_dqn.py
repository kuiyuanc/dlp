# Spring 2025, 535507 Deep Learning
# Lab5: Value-based RL
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh

import argparse
import os
import pickle
from collections import deque
from pathlib import Path

import ale_py
import gymnasium as gym
import numpy as np
import torch
import wandb
from conv_dqn import ConvDQNAgent
from dqn import PrioritizedReplayBuffer, Transition

gym.register_envs(ale_py)


class EnhancedConvDQNAgent(ConvDQNAgent):
    def __init__(self, args: argparse.Namespace, env_name: str = "ALE/Pong-v5"):
        super(EnhancedConvDQNAgent, self).__init__(args, env_name)

        self.memory = PrioritizedReplayBuffer(args.memory_size, args.alpha, args.beta, args.epsilon)
        self.return_steps = args.return_steps
        self.discount = self.gamma**self.return_steps

        self.beta_step = (1 - args.beta) / args.beta_anneal_steps if args.beta_anneal else 0.0
        if args.args_path:
            with open(f"{str(args.args_path).removesuffix('.pkl')}_beta.pkl", "rb") as f:
                self.memory.beta = pickle.load(f)

    def run(self, episodes: int = 1000) -> None:
        ewma_return = self.best_reward

        for ep in range(self.ep, episodes + 1):
            obs, _ = self.env.reset()

            state = self.preprocessor.reset(obs)
            state = torch.from_numpy(state).float().unsqueeze_(dim=0)
            done = False
            total_reward = 0
            step_count = 0

            while not done and step_count < self.max_episode_steps:
                action = self.select_action(state)

                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                next_state = self.preprocessor.step(next_obs)
                next_state = torch.from_numpy(next_state).float().unsqueeze_(dim=0)
                action = torch.tensor(action, dtype=torch.int64)
                reward = torch.tensor(reward, dtype=torch.float32)
                done = torch.tensor(done, dtype=torch.float32)

                self.memory.add(Transition(state, action, reward, next_state, done))

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
                with open(Path(self.save_dir, f"model_ep{ep}_beta.pkl"), "wb") as f:
                    pickle.dump(self.memory.beta, f)

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

    def train(self):
        if len(self.memory) < self.replay_start_size:
            return

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.train_count += 1

        samples, indices, weights = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, masks = zip(*samples)
        weights = torch.from_numpy(weights).to(self.device)

        states = torch.stack(states).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        actions = torch.stack(actions).to(self.device).unsqueeze_(1)
        rewards = torch.stack(rewards).to(self.device)
        masks = torch.stack(masks).to(self.device)
        q_values = self.q_net(states).gather(1, actions).flatten()

        target_q = self.get_target_q(rewards, next_states, masks, self.discount)
        errors = target_q - q_values
        loss = (errors**2 * weights).mean()
        self.memory.update_priorities(indices, errors.detach().cpu().numpy())
        self.memory.beta_anneal(self.beta_step)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.train_count % self.target_update_frequency == 0:
            # self.target_net.load_state_dict(self.q_net.state_dict())
            for target_param, param in zip(self.target_net.parameters(), self.q_net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        wandb.log(
            {
                "Environment Steps": self.env_count,
                "Train/Steps": self.train_count,
                "Train/Loss": loss.item(),
                "Train/Batch Q Mean": q_values.mean().item(),
                "Train/Batch Q Standard Deviation": q_values.std().item(),
                "Train/Importance Sampling beta": self.memory.beta,
            }
        )

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
