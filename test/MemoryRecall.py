from collections import deque
import random

class MemoryRecall():
    """MemoryRecall module - needed to store and recall (and when needed for training randomly sample a batch of) data from the environment.
    """
    def __init__(self, memory_size: int) -> None:
        """Initialize the memory recall with the size of the memory.

        Args:
            memory_size (int): Size of the memory to store the data
        """
        self.memory_size = memory_size
        self.memory = deque(maxlen = self.memory_size)
    
    def cache(self, data) -> None:
        """Cache the data from the environment.

        Args:
            data: tuple of data to store in the memory
        """
        self.memory.append(data)

    def recall(self, batch_size: int):
        """Recall a batch of data from the memory.

        Args:
            batch_size (int): Size of the batch to recall

        Returns:
            list: Random sample of the memory
        """
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        """Return the length of the memory.

        Returns:
            int: Length of the memory
        """
        return len(self.memory)

