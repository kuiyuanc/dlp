import argparse
import random
from collections import deque, namedtuple
from pathlib import Path
from typing import Generator, Union

import ale_py
import cv2
import gymnasium as gym
import numpy as np
import pretty_errors  # noqa: F401
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
from torch import nn

gym.register_envs(ale_py)

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))


class AtariPreprocessor:
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


class PER:
    def __init__(
        self,
        capacity: int = 25000,
        alpha: float = 0.6,
        beta: float = 0.4,
        epsilon: float = 0.01,
        max_error: float = 10.0,
        gamma: float = 0.99,
        n_step: int = 3,
    ):
        self._init_per(capacity, alpha, beta, epsilon, max_error)
        self._init_n_step(gamma, n_step)

    def __len__(self) -> int:
        return self.size

    def add(self, transition: Transition, error: float = 1.0) -> None:
        self.n_step_buffer.append(transition)
        if len(self.n_step_buffer) < self.n_step:
            return

        n_step_transition = self._get_n_step_transition()
        self.buffer[self.pos] = n_step_transition
        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

        if n_step_transition.done:
            self.n_step_buffer.clear()

    def sample(self, batch_size: int) -> tuple[Generator, np.ndarray, np.ndarray]:
        p = self.priorities / self.priorities.sum()
        indices = np.random.choice(self.size, batch_size, p=p[: self.size])
        weights = (self.size * p[indices]) ** (-self.beta)
        weights /= weights.max()
        samples = (self.buffer[i] for i in indices)
        return samples, indices, weights

    def update_priorities(self, indices: np.ndarray, errors: np.ndarray) -> None:
        self.priorities[indices] = np.power(abs(errors) + self.epsilon, self.alpha)

    def beta_anneal(self, step: float) -> None:
        self.beta = min(1, self.beta + step)

    def _init_per(self, capacity: int, alpha: float, beta: float, epsilon: float, max_error: float) -> None:
        self.capacity: int = capacity
        self.alpha: float = alpha
        self.beta: float = beta
        self.epsilon: float = epsilon
        self.max_priority: float = max_error**alpha
        self.buffer: list[Union[Transition, None]] = [None] * capacity
        self.priorities: np.ndarray = np.zeros((capacity,), dtype=np.float32)
        self.pos: int = 0
        self.size: int = 0

    def _init_n_step(self, gamma: float, n_step: int) -> None:
        self.n_step: int = n_step
        self.gammas: np.ndarray = gamma ** np.arange(n_step, dtype=np.float32)
        self.n_step_buffer: deque[Transition] = deque(maxlen=n_step)

    def _get_n_step_transition(self) -> Transition:
        state: torch.Tensor = self.n_step_buffer[0].state
        action: torch.Tensor = self.n_step_buffer[0].action
        next_state: torch.Tensor = self.n_step_buffer[-1].next_state
        reward: np.float64 = np.dot(self.gammas, tuple(t.reward for t in self.n_step_buffer))
        done: torch.Tensor = self.n_step_buffer[-1].done
        return Transition(state, action, torch.tensor(reward, dtype=torch.float32), next_state, done)


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.FloatTensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer("bias_epsilon", torch.FloatTensor(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))

    def reset_noise(self):
        self.weight_epsilon.normal_()  # type: ignore
        self.bias_epsilon.normal_()  # type: ignore

    def forward(self, input):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon  # type: ignore
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon  # type: ignore
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(input, weight, bias)


class NoisyDuelingDistributionalNet(nn.Module):
    def __init__(self, n_actions, n_atoms, v_min, v_max, in_channels: int = 4):
        super().__init__()
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (n_atoms - 1)
        self.n_actions = n_actions
        self.register_buffer("support", torch.linspace(v_min, v_max, n_atoms))

        self.network = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        conv_output_size = 3136

        self.value_head = nn.Sequential(NoisyLinear(conv_output_size, 512), nn.ReLU(), NoisyLinear(512, n_atoms))

        self.advantage_head = nn.Sequential(
            NoisyLinear(conv_output_size, 512), nn.ReLU(), NoisyLinear(512, n_atoms * self.n_actions)
        )

    def forward(self, x):
        h = self.network(x / 255.0)
        value = self.value_head(h).view(-1, 1, self.n_atoms)
        advantage = self.advantage_head(h).view(-1, self.n_actions, self.n_atoms)
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        q_dist = F.softmax(q_atoms, dim=2)
        return q_dist

    def reset_noise(self):
        for layer in self.value_head:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()
        for layer in self.advantage_head:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()


class Agent:
    def __init__(self, args: argparse.Namespace, env_name: str = "ALE/Pong-v5"):
        self._make_env(env_name, args.seed)
        self.preprocessor = AtariPreprocessor()

        assert isinstance(self.env.action_space, gym.spaces.Discrete)
        self.num_actions = int(self.env.action_space.n)

        self._init_hyperparam(args)
        self._init_count()
        self._config_train(args)

        self._init_net()
        self.memory = PER(args.memory_size, args.alpha, args.beta, args.epsilon, args.n_step)

    @torch.no_grad()
    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        q_dist = self.q_net(state.to(self.device))
        q_values = torch.sum(q_dist * self.q_net.support, dim=2)
        return q_values.argmax(dim=1)

    def run(self, episodes: int = 1000) -> None:
        ewma_reward = self.best_reward

        for ep in range(1, 1 + episodes):
            obs, _ = self.env.reset()
            state = self.preprocessor.reset(obs)
            state = torch.from_numpy(state).float().unsqueeze_(dim=0)
            done = False
            total_reward = 0
            step_count = 0

            while not done and step_count < self.max_episode_steps:
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action.item())
                done = terminated or truncated

                next_state = self.preprocessor.step(next_obs)
                next_state = torch.from_numpy(next_state).float().unsqueeze_(dim=0)
                reward = torch.tensor(reward, dtype=torch.float32)
                done = torch.tensor(done, dtype=torch.float32)

                # (2D, 2D, 0D, 2D, 0D)
                self.memory.add(Transition(state, action, reward, next_state, done))

                if self.replay_start_size <= self.env_count:
                    for _ in range(self.train_per_step):
                        self.train()

                state = next_state
                total_reward += reward.item()
                self.env_count += 1
                step_count += 1

            wandb.log(
                {
                    "Episodes": ep,
                    "Train/Episodic Length": step_count,
                    "Train/Episodic Reward": total_reward,
                    "Environment Steps": self.env_count,
                    "Train/Steps": self.train_count,
                }
            )

            if ep % self.backup_frequency == 0:
                model_path = Path(self.save_dir, f"model_ep{ep}.pt")
                torch.save({"q_net": self.q_net.state_dict()}, model_path)

            if ep % self.eval_frequency == 0:
                num_evals = int(1 + np.log2(self.eval_frequency))
                eval_reward, episode_len = self.evaluate(num_evals)

                if eval_reward >= self.best_reward:
                    self.best_reward = float(eval_reward)
                    model_path = Path(self.save_dir, "best_model.pt")
                    torch.save({"q_net": self.q_net.state_dict()}, model_path)

                ewma_reward = 0.05 * eval_reward + (1 - 0.05) * ewma_reward
                wandb.log(
                    {
                        "Episodes": ep,
                        "Environment Steps": self.env_count,
                        "Train/Steps": self.train_count,
                        "Eval/Episodic Reward": eval_reward,
                        "Eval/Episodic Length": episode_len,
                        "EWMA Reward": ewma_reward,
                    }
                )

                if ewma_reward > self.early_stop:
                    break

    def train(self) -> None:
        self.q_net.train()
        self.train_count += 1

        samples, indices, weights = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)

        weights = torch.from_numpy(weights).to(self.device)  # 1D
        states = torch.cat(states).to(self.device)  # 2D
        next_states = torch.cat(next_states).to(self.device)  # 2D
        actions = torch.cat(actions).to(self.device)  # 1D
        rewards = torch.stack(rewards).to(self.device).unsqueeze_(-1)  # 2D
        dones = torch.stack(dones).to(self.device).unsqueeze_(-1)  # 2D

        self.q_net.reset_noise()
        self.target_net.reset_noise()

        with torch.no_grad():
            next_dist = self.target_net(next_states)
            next_actions = self.select_action(next_states)
            next_pmfs = next_dist[torch.arange(self.batch_size), next_actions]

            assert isinstance(self.target_net.support, torch.Tensor)
            next_atoms = rewards + self.gamma_n * self.target_net.support * (1 - dones)
            target_zone = next_atoms.clamp(self.q_net.v_min, self.q_net.v_max)

            b = (target_zone - self.q_net.v_min) / self.q_net.delta_z
            low = b.floor().clamp(0, self.n_atoms - 1)
            up = b.ceil().clamp(0, self.n_atoms - 1)

            d_m_low = (up.float() + (b == low).float() - b) * next_pmfs
            d_m_up = (b - low) * next_pmfs

            target_pmfs = torch.zeros_like(next_pmfs)
            for i in range(target_pmfs.size(0)):
                target_pmfs[i].index_add_(0, low[i].long(), d_m_low[i])
                target_pmfs[i].index_add_(0, up[i].long(), d_m_up[i])

        dist = self.q_net(states)
        pred_dist = dist[torch.arange(self.batch_size), actions]
        log_pred = torch.log(pred_dist.clamp(min=1e-5, max=1 - 1e-5))

        loss_per_sample = -(target_pmfs * log_pred).sum(dim=1)
        loss = (loss_per_sample * weights).mean()

        new_priorities = loss_per_sample.detach().cpu().numpy()
        self.memory.update_priorities(indices, new_priorities)
        self.memory.beta_anneal(self.beta_step)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.train_count % self.target_update_frequency == 0:
            for target_param, param in zip(self.target_net.parameters(), self.q_net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        wandb.log(
            {
                "Environment Steps": self.env_count,
                "Train/Steps": self.train_count,
                "Train/Loss": loss.item(),
                "Train/Importance Sampling beta": self.memory.beta,
            }
        )

    @torch.no_grad()
    def evaluate(self, num: int) -> tuple[float, float]:
        self.q_net.eval()

        total_reward = 0
        step_count = 0

        for _ in range(num):
            obs, _ = self.test_env.reset()
            state = self.preprocessor.reset(obs)
            done = False

            while not done:
                state = torch.from_numpy(state).float().unsqueeze_(dim=0).to(self.device)
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = self.test_env.step(action.item())
                done = terminated or truncated
                total_reward += float(reward)
                state = self.preprocessor.step(next_obs)
                step_count += 1

        return total_reward / num, step_count / num

    def _make_env(self, env_name: str, seed: int) -> None:
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.env.reset(seed=seed)
        self.test_env = gym.make(env_name, render_mode="rgb_array")
        self.test_env.reset(seed=seed)

    def _init_net(self) -> None:
        assert isinstance(self.env.action_space, gym.spaces.Discrete)
        self.q_net = NoisyDuelingDistributionalNet(self.env.action_space.n, self.n_atoms, -21, 21).to(self.device)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)

        if self.device.type.startswith("xpu"):
            import intel_extension_for_pytorch as ipex

            self.q_net, self.optimizer = ipex.optimize(self.q_net, torch.float32, self.optimizer, inplace=True)

        self.target_net = NoisyDuelingDistributionalNet(self.env.action_space.n, self.n_atoms, -21, 21).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

    def _init_hyperparam(self, args: argparse.Namespace) -> None:
        self.lr: float = args.learning_rate
        self.batch_size: int = args.batch_size
        self.gamma: float = args.gamma
        self.gamma_n: float = self.gamma**args.n_step
        self.max_episode_steps: int = args.max_episode_steps
        self.replay_start_size: int = args.replay_start_size
        self.target_update_frequency: int = args.target_update_frequency
        self.train_per_step: int = args.train_per_step
        self.tau: float = args.tau
        self.n_atoms: int = args.n_atoms
        self.beta_step: float = (1 - args.beta) / args.beta_anneal_steps if args.beta_anneal else 0.0

    def _init_count(self) -> None:
        self.env_count: int = 0
        self.train_count: int = 0
        self.best_reward: float = -21

    def _config_train(self, args) -> None:
        self.save_dir: Path = Path(args.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self._select_device(args.device)

        self.early_stop: int = args.early_stop
        self.eval_frequency: int = args.eval_frequency
        self.backup_frequency: int = args.backup_frequency

    def _select_device(self, device: str) -> None:
        if device.startswith("cuda"):
            device = device if torch.cuda.is_available() else "xpu"
        if device.startswith("xpu"):
            device = device if torch.xpu.is_available() else "cpu"
        self.device = torch.device(device)
        print("Using device:", self.device)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--save-dir", type=str, default="./weights")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--memory-size", type=int, default=40000)
    parser.add_argument("--learning-rate", "-lr", type=float, default=0.0000625)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--target-update-frequency", type=int, default=2000)
    parser.add_argument("--replay-start-size", type=int, default=5000)
    parser.add_argument("--max-episode-steps", type=int, default=10000)
    parser.add_argument("--train-per-step", type=int, default=4)

    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--beta-anneal", action="store_true")
    parser.add_argument("--beta-anneal-steps", type=int, default=200_000)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--n-step", type=int, default=3)

    parser.add_argument("--tau", type=float, default=0.002)
    parser.add_argument("--n-atoms", type=int, default=51)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-episodes", type=int, default=200)
    parser.add_argument("--eval-frequency", type=int, default=4)
    parser.add_argument("--backup-frequency", type=int, default=100)
    parser.add_argument("--early-stop", type=float, default=19)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--api-key", type=str, default="")

    return parser.parse_args()


def args_to_config(args: argparse.Namespace) -> dict:
    config = vars(args).copy()
    config.pop("save_dir")
    config.pop("num_episodes")
    config.pop("eval_frequency")
    config.pop("backup_frequency")
    config.pop("early_stop")
    config.pop("api_key")
    return config


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def train():
    args = get_args()
    set_seed(args.seed)

    group = "rainbow"
    project = "Pong-v5"
    enhance = "rainbow"
    dir = Path("wandb", project, enhance)
    config = args_to_config(args)

    wandb.init(group=group, dir=dir, project=project, config=config)

    assert wandb.run
    wandb.run.name = wandb.run.id

    args.save_dir = Path(args.save_dir, project, enhance, wandb.run.id)
    agent = Agent(args)
    agent.run(args.num_episodes)

    wandb.finish()


def main():
    args = get_args()
    if args.api_key:
        wandb.login(key=args.api_key)
    train()


if __name__ == "__main__":
    main()
