import random
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def append(self, transition: tuple) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size) -> list:
        return random.sample(self.buffer, batch_size)
