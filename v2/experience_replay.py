# Define memory for Experience Replay
from collections import deque
import random


class ReplayMemory:
    """
    A class to represent a Replay Memory for Experience Replay.

    This class stores transitions in a fixed-size buffer and allows for random sampling
    to improve learning stability in reinforcement learning algorithms.
    """

    def __init__(self, maxlen: int, seed: int = None) -> None:
        """
        Initialize the replay memory.

        Args:
            maxlen (int): Size of the deque buffer.
            seed (int, optional): Random seed for reproducibility. Default is None.
        """
        self.memory = deque([], maxlen=maxlen)

        if seed is not None:
            random.seed(seed)

    def append(self, transition) -> None:
        """
        Append experience to the replay memory.

        Args:
            transition: The tuple to store. (state, action, new_state, reward, terminated)
        """
        self.memory.append(transition)

    def sample(self, sample_size: int) -> list:
        """
        Sample a batch of transitions randomly from the memory.

        Args:
            sample_size (int): The number of transitions to be return sampled and returned.

        Returns:
            list: A list of randomly sampled transitions.
        """
        return random.sample(self.memory, sample_size)

    def __len__(self) -> int:
        """
        Get the current number of transitions stored in memory.

        Returns:
            int: The number of transitions currently in the memory.
        """
        return len(self.memory)
