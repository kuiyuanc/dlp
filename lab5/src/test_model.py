import argparse
import os
import random
from collections import deque

import ale_py
import cv2
import gymnasium as gym
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# class DQN(nn.Module):
#     def __init__(self, input_channels, num_actions):
#         super(DQN, self).__init__()
#         self.network = nn.Sequential(
#             nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=4, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1),
#             nn.ReLU(),
#             nn.Flatten(),
#             nn.Linear(64 * 7 * 7, 512),
#             nn.ReLU(),
#             nn.Linear(512, num_actions),
#         )


#     def forward(self, x):
#         return self.network(x / 255.0)


class AtariPreprocessor:
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        if len(obs.shape) == 3 and obs.shape[2] == 3:
            gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        else:
            gray = obs
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame.copy())
        stacked = np.stack(self.frames, axis=0)
        return stacked


def evaluate(args, DQN: type, env_name: str, atari: bool) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = gym.make(f"ALE/{env_name}" if atari else env_name, render_mode="rgb_array")
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)

    preprocessor = AtariPreprocessor()
    assert isinstance(env.action_space, gym.spaces.Discrete)
    num_actions = env.action_space.n

    # model = DQN(4, num_actions).to(device)
    model = DQN(num_actions).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device)["q_net"])
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    sum_reward = 0
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        state = preprocessor.reset(obs) if atari else obs
        done = False
        total_reward = 0
        frames = []
        frame_idx = 0

        while not done:
            frame = env.render()
            frames.append(frame)

            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action = model(state_tensor).argmax().item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += float(reward)
            state = preprocessor.step(next_obs) if atari else next_obs
            frame_idx += 1

        out_path = os.path.join(args.output_dir, f"eval_ep{ep}.mp4")
        with imageio.get_writer(out_path, fps=30) as video:
            for f in frames:
                f = cv2.resize(f, (592, 400))  # for video compatibility with most codecs and players
                video.append_data(f)  # type: ignore
        print(f"Saved episode {ep:2d} with total reward {total_reward} → {out_path}")

        sum_reward += total_reward

    mean_reward = sum_reward / args.episodes
    print(f"Average reward over {args.episodes} episodes: {mean_reward:.3f}")

    return mean_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained .pt model")
    parser.add_argument("--output-dir", type=str, default="./eval_videos")
    # parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=313551076, help="Random seed for evaluation")
    args = parser.parse_args()
    evaluate(args)
