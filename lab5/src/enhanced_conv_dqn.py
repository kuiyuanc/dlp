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
import wandb
from conv_dqn import ConvDQNAgent
from dqn import PrioritizedReplayBuffer

gym.register_envs(ale_py)


class RainbowConvDQNAgent(ConvDQNAgent):
    def __init__(self, args: argparse.Namespace, env_name: str = "ALE/Pong-v5"):
        super(RainbowConvDQNAgent, self).__init__(args, env_name)

        self.memory = PrioritizedReplayBuffer(args.memory_size, args.alpha, args.beta, args.epsilon)
        self.beta = args.beta
        self.beta_step = (1 - self.beta) / args.anneal_steps
        self.return_steps = args.return_steps
        self.discount = self.gamma ** np.arange(self.return_steps, dtype=np.float32)

        if args.args_path:
            with open(f"{str(args.args_path).removesuffix('.pkl')}_beta.pkl", "rb") as f:
                self.memory.beta = pickle.load(f)

    def run(self, episodes: int = 1000) -> None:
        for ep in range(self.ep, episodes + 1):
            obs, _ = self.env.reset()

            state = self.preprocessor.reset(obs)
            done = False
            total_reward = 0
            step_count = 0

            states, actions, rewards = [], [], []

            while not done and step_count < self.max_episode_steps:
                action = self.select_action(state)
                segment_reward = 0

                for _ in range(self.skip_frames):
                    next_obs, reward, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated

                    segment_reward += reward * self.reward_scaling
                    next_state = self.preprocessor.step(next_obs)

                    if done:
                        break

                states.append(state)
                actions.append(action)
                rewards.append(segment_reward)

                state = next_state
                total_reward += segment_reward
                self.env_count += 1
                step_count += 1

            next_states = states[self.return_steps :] + [next_state] * self.return_steps
            dones = [False] * (len(states) - self.return_steps) + [done] * self.return_steps

            n = len(states)
            gammas = np.ones((n,), dtype=np.float32) * self.return_steps
            gammas[-self.return_steps :] -= np.arange(self.return_steps, dtype=np.float32)
            gammas = self.gamma**gammas

            multi_step_rewards = np.zeros((n, self.return_steps), dtype=np.float32)
            for i in range(n):
                end = min(self.return_steps, n - i)
                multi_step_rewards[i, :end] = rewards[i : i + end]
            rewards = (multi_step_rewards @ self.discount) * self.reward_scaling

            states_tensor = torch.from_numpy(np.asarray(states, dtype=np.float32)).to(self.device)
            next_states_tensor = torch.from_numpy(np.asarray(next_states, dtype=np.float32)).to(self.device)
            actions_tensor = torch.tensor(actions, dtype=torch.int64).to(self.device)
            rewards_tensor = torch.from_numpy(rewards).to(self.device)
            dones_tensor = torch.tensor(dones, dtype=torch.float32).to(self.device)
            gammas_tensor = torch.from_numpy(gammas).to(self.device)

            with torch.no_grad():
                q_values = self.q_net(states_tensor).gather(1, actions_tensor.unsqueeze_(1)).squeeze_(1)
                next_actions = self.q_net(next_states_tensor).argmax(1, keepdim=True)
                next_target_q = self.target_net(next_states_tensor).gather(1, next_actions).squeeze_(1)
                target_q = rewards_tensor + gammas_tensor * next_target_q * (1 - dones_tensor)
                errors = target_q - q_values

            for record, error in zip(zip(states, actions, rewards, next_states, dones, gammas), errors):
                self.memory.append(record, error)

            for _ in range(self.train_per_step * n):
                self.train()

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
                eval_reward, episode_len = self.evaluate()
                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    model_path = os.path.join(self.save_dir, "best_model.pt")
                    torch.save({"q_net": self.q_net.state_dict()}, model_path)

                name = f"model_ep{ep}_step{self.env_count}_reward{eval_reward}"
                model_path = os.path.join(self.save_dir, f"{name}.pt")
                torch.save({"q_net": self.q_net.state_dict(), "target_net": self.target_net.state_dict()}, model_path)
                with open(Path(self.save_dir, f"{name}.pkl"), "wb") as f:
                    pickle.dump((self.epsilon, self.env_count, self.train_count, self.best_reward, ep + 1), f)
                with open(Path(self.save_dir, f"{name}_beta.pkl"), "wb") as f:
                    pickle.dump(self.memory.beta, f)

                wandb.log(
                    {
                        "Episode": ep,
                        "Env Step Count": self.env_count,
                        "Update Count": self.train_count,
                        "Eval Reward": eval_reward,
                        "Eval Episode Length": episode_len,
                    }
                )

    def train(self):
        if len(self.memory) < self.replay_start_size:
            return

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.train_count += 1

        samples, indices, weights = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones, gammas = zip(*samples)
        weights = torch.from_numpy(weights).to(self.device)

        states = torch.from_numpy(np.asarray(states, dtype=np.float32)).to(self.device)
        next_states = torch.from_numpy(np.asarray(next_states, dtype=np.float32)).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        gammas = torch.tensor(gammas, dtype=torch.float32).to(self.device)
        q_values = self.q_net(states).gather(1, actions.unsqueeze_(1)).squeeze_(1)

        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(1, keepdim=True)
            next_target_q = self.target_net(next_states).gather(1, next_actions).squeeze_(1)
            target_q = rewards + gammas * next_target_q * (1 - dones)

        errors = target_q - q_values
        loss = (errors**2 * weights).mean()
        self.memory.update_priorities(indices, errors.detach().cpu().numpy())
        self.memory.beta_anneal(self.beta_step)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        wandb.log(
            {
                "Env Step Count": self.env_count,
                "Update Count": self.train_count,
                "Loss": loss.item(),
                "Q mean": q_values.mean().item(),
                "Q std": q_values.std().item(),
                "Importance Sampling beta": self.memory.beta,
            }
        )
