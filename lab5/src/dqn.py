# Spring 2025, 535507 Deep Learning
# Lab5: Value-based RL
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh

import argparse
import os
import pickle
import random

# import time
from collections import deque
from pathlib import Path

# import ale_py
import cv2
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from replay import ReplayBuffer

# gym.register_envs(ale_py)


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class DQN(nn.Module):
    """
    Design the architecture of your deep Q network
    - Input size is the same as the state dimension; the output size is the same as the number of actions
    - Feel free to change the architecture (e.g. number of hidden layers and the width of each hidden layer) as you like
    - Feel free to add any member variables/functions whenever needed
    """

    def __init__(self, num_actions: int, input_dim: int = 4):
        super(DQN, self).__init__()
        # An example:
        # self.network = nn.Sequential(
        #    nn.Linear(input_dim, 64),
        #    nn.ReLU(),
        #    nn.Linear(64, 64),
        #    nn.ReLU(),
        #    nn.Linear(64, num_actions)
        # )
        ########## YOUR CODE HERE (5~10 lines) ##########
        layers = (nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, num_actions))
        self.network = nn.Sequential(*layers)

        ########## END OF YOUR CODE ##########

    def forward(self, x):
        return self.network(x)


class AtariPreprocessor:
    """
    Preprocesing the state input of DQN for Atari
    """

    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)


class PrioritizedReplayBuffer:
    """
    Prioritizing the samples in the replay memory by the Bellman error
    See the paper (Schaul et al., 2016) at https://arxiv.org/abs/1511.05952
    """

    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4, epsilon: float = 0.01):
        self.capacity: int = capacity
        self.alpha: float = alpha
        self.beta: float = beta
        self.buffer: list = [None] * capacity
        self.priorities: np.ndarray = np.zeros((capacity,), dtype=np.float32)
        self.pos: int = 0
        self.epsilon: float = epsilon
        self.len: int = 0

    def __len__(self) -> int:
        return self.len

    def add(self, transition: tuple, error: float) -> None:
        ########## YOUR CODE HERE (for Task 3) ##########
        self.buffer[self.pos] = transition
        self.priorities[self.pos] = (abs(error) + self.epsilon) ** self.alpha
        self.pos = (self.pos + 1) % self.capacity
        self.len = min(self.capacity, self.len + 1)
        ########## END OF YOUR CODE (for Task 3) ##########
        return

    def sample(self, batch_size: int) -> tuple:
        ########## YOUR CODE HERE (for Task 3) ##########
        p = self.priorities / self.priorities.sum()
        indices = np.random.choice(self.len, batch_size, p=p[: self.len])
        weights = (self.len * p[indices]) ** (-self.beta)
        weights /= weights.max()
        samples = (self.buffer[i] for i in indices)
        ########## END OF YOUR CODE (for Task 3) ##########
        return samples, indices, weights

    def update_priorities(self, indices: np.ndarray, errors: np.ndarray) -> None:
        ########## YOUR CODE HERE (for Task 3) ##########
        self.priorities[indices] = (abs(errors) + self.epsilon) ** self.alpha
        ########## END OF YOUR CODE (for Task 3) ##########
        return


class DQNAgent:
    def __init__(self, args: argparse.Namespace, env_name: str = "CartPole-v1"):
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.env.reset(seed=args.seed)
        self.test_env = gym.make(env_name, render_mode="rgb_array")
        self.test_env.reset(seed=args.seed)
        assert isinstance(self.env.action_space, gym.spaces.Discrete)
        self.num_actions = int(self.env.action_space.n)
        # self.preprocessor = AtariPreprocessor()

        # device = "cuda" if torch.cuda.is_available() else "cpu"
        if args.device.startswith("cuda"):
            args.device = args.device if torch.cuda.is_available() else "xpu"
        if args.device.startswith("xpu"):
            args.device = args.device if torch.xpu.is_available() else "cpu"
        self.device = torch.device(args.device)
        print("Using device:", self.device)

        self.q_net = DQN(self.num_actions).to(self.device)
        self.q_net.apply(init_weights)
        self.target_net = DQN(self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.learning_rate)

        if args.model_path:
            state_dict = torch.load(args.model_path, map_location=self.device)
            self.q_net.load_state_dict(state_dict["q_net"])
            self.target_net.load_state_dict(state_dict["target_net"])

        if self.device.type.startswith("xpu"):
            import intel_extension_for_pytorch as ipex

            self.q_net, self.optimizer = ipex.optimize(self.q_net, torch.float32, self.optimizer, inplace=True)

        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min

        self.env_count = 0
        self.train_count = 0
        self.best_reward = 0  # Initilized to 0 for CartPole and to -21 for Pong
        self.max_episode_steps = args.max_episode_steps
        self.replay_start_size = args.replay_start_size
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step = args.train_per_step
        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.memory = ReplayBuffer(args.memory_size)

        self.get_target_q = self._target_q_double if args.double else self._target_q_vanilla
        self.skip_frames = args.skip_frames
        self.tau = args.tau

        self.early_stop = args.early_stop
        self.eval_frequency = args.eval_frequency
        self.backup_frequency = args.backup_frequency
        self.ep = 1

        if args.args_path:
            with open(args.args_path, "rb") as f:
                self.epsilon, self.env_count, self.train_count, self.best_reward, self.ep = pickle.load(f)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        state_tensor = torch.from_numpy(np.asarray(state)).float().unsqueeze_(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()

    def run(self, episodes: int = 1000) -> None:
        ewma_return = self.best_reward

        for ep in range(self.ep, episodes + 1):
            obs, _ = self.env.reset()

            # state = self.preprocessor.reset(obs)
            state = obs
            done = False
            total_reward = 0
            step_count = 0

            while not done and step_count < self.max_episode_steps:
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                # next_state = self.preprocessor.step(next_obs)
                next_state = next_obs
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

                # if self.env_count % 1000 == 0:
                #     print(
                #         f"[Collect] Ep: {ep} Step: {step_count} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}"
                #     )
                #     wandb.log(
                #         {
                #             "Episode": ep,
                #             "Step Count": step_count,
                #             "Env Step Count": self.env_count,
                #             "Update Count": self.train_count,
                #             "Epsilon": self.epsilon,
                #         }
                #     )
                #     ########## YOUR CODE HERE  ##########
                #     # Add additional wandb logs for debugging if needed

                #     ########## END OF YOUR CODE ##########
            # print(
            #     f"[Eval] Ep: {ep} Total Reward: {total_reward} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}"
            # )
            ########## YOUR CODE HERE  ##########
            # Add additional wandb logs for debugging if needed
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

            ########## END OF YOUR CODE ##########
            # if ep % 100 == 0:
            if ep % self.backup_frequency == 0:
                model_path = os.path.join(self.save_dir, f"model_ep{ep}.pt")
                torch.save({"q_net": self.q_net.state_dict(), "target_net": self.target_net.state_dict()}, model_path)
                with open(Path(self.save_dir, f"model_ep{ep}.pkl"), "wb") as f:
                    pickle.dump((self.epsilon, self.env_count, self.train_count, self.best_reward, ep + 1), f)
                # print(f"Saved model checkpoint to {model_path}")

            # if ep % 20 == 0:
            if ep % self.eval_frequency == 0:
                eval_reward, episode_len = self.evaluate()
                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    model_path = os.path.join(self.save_dir, "best_model.pt")
                    torch.save({"q_net": self.q_net.state_dict()}, model_path)
                    # print(f"Saved new best model to {model_path} with reward {eval_reward}")
                # print(f"[TrueEval] Ep: {ep} Return: {eval_reward:.2f} SC: {self.env_count} UC: {self.train_count}")
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
        # state = self.preprocessor.reset(obs)
        state = obs
        done = False
        total_reward = 0
        step_count = 0

        while not done:
            state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze_(0).to(self.device)
            with torch.no_grad():
                action = self.q_net(state_tensor).argmax().item()
            next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
            done = terminated or truncated
            total_reward += float(reward)
            # state = self.preprocessor.step(next_obs)
            state = next_obs
            step_count += 1

        return total_reward, step_count

    def train(self):
        if len(self.memory) < self.replay_start_size:
            return

        # Decay function for epsilin-greedy exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.train_count += 1

        ########## YOUR CODE HERE (<5 lines) ##########
        # Sample a mini-batch of (s,a,r,s',done) from the replay buffer
        states, actions, rewards, next_states, masks = zip(*self.memory.sample(self.batch_size))

        ########## END OF YOUR CODE ##########

        # Convert the states, actions, rewards, next_states, and dones into torch tensors
        # NOTE: Enable this part after you finish the mini-batch sampling
        # states = torch.from_numpy(np.asarray(states, dtype=np.float32)).to(self.device)
        # next_states = torch.from_numpy(np.asarray(next_states, dtype=np.float32)).to(self.device)
        # actions = torch.tensor(actions, dtype=torch.int64, device=self.device)
        # rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        # masks = torch.tensor(masks, dtype=torch.bool, device=self.device)
        states = torch.stack(states).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        actions = torch.stack(actions).to(self.device).unsqueeze_(0)
        rewards = torch.stack(rewards).to(self.device)
        masks = torch.stack(masks).to(self.device)
        q_values = self.q_net(states).gather(1, actions).flatten()

        ########## YOUR CODE HERE (~10 lines) ##########
        # Implement the loss function of DQN and the gradient updates
        target_q = self.get_target_q(rewards, next_states, masks)
        loss = nn.MSELoss()(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        ########## END OF YOUR CODE ##########

        if self.train_count % self.target_update_frequency == 0:
            # self.target_net.load_state_dict(self.q_net.state_dict())
            for target_param, param in zip(self.target_net.parameters(), self.q_net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        # NOTE: Enable this part if "loss" is defined
        # if self.train_count % 1000 == 0:
        #    print(f"[Train #{self.train_count}] Loss: {loss.item():.4f} Q mean: {q_values.mean().item():.3f} std: {q_values.std().item():.3f}")

        wandb.log(
            {
                "Environment Steps": self.env_count,
                "Train/Steps": self.train_count,
                "Train/Loss": loss.item(),
                "Train/Batch Q Mean": q_values.mean().item(),
                "Train/Batch Q Standard Deviation": q_values.std().item(),
            }
        )

    @torch.no_grad()
    def _target_q_vanilla(self, rewards, next_states: torch.Tensor, masks, gammas=None) -> torch.Tensor:
        gammas = self.gamma if gammas is None else gammas
        return rewards + torch.masked_fill(gammas * self.target_net(next_states).amax(dim=1), masks, 0)

    @torch.no_grad()
    def _target_q_double(self, rewards, next_states: torch.Tensor, masks, gammas=None) -> torch.Tensor:
        gammas = self.gamma if gammas is None else gammas
        next_actions = self.q_net(next_states).argmax(1, keepdim=True)
        next_target_q = self.target_net(next_states).gather(1, next_actions).flatten()
        return rewards + torch.masked_fill(gammas * next_target_q, masks, 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, default="./results")
    parser.add_argument("--wandb-run-name", type=str, default="cartpole-run")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--memory-size", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.999999)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--target-update-frequency", type=int, default=1000)
    parser.add_argument("--replay-start-size", type=int, default=50000)
    parser.add_argument("--max-episode-steps", type=int, default=10000)
    parser.add_argument("--train-per-step", type=int, default=1)
    args = parser.parse_args()

    wandb.init(project="DLP-Lab5-DQN-CartPole", name=args.wandb_run_name, save_code=True)
    agent = DQNAgent(args=args)
    agent.run()
