import torch
from torch import nn
import torch.nn.functional as F


class DQN(nn.Module):
    """ Deep Q-Network (DQN) class.

        Args:
            state_dim (int): dimension of the state space
            action_dim (int): dimension of the action space
            hidden_dim (int): dimension of the hidden layer
    """

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        """ Forward pass of the network.

            Args:
                x (torch.Tensor): input tensor (state)
            Returns:
                torch.Tensor: output tensor (Q-values)
        """

        x = F.relu(self.fc1(x))  # ReLU activation function for the first layer
        return self.fc2(x)  # Output layer


if __name__ == "__main__":
    # Create a DQN instance
    state_dim = 12
    action_dim = 2
    dqn = DQN(state_dim, action_dim)

    # Create a random state tensor
    state = torch.randn(1, state_dim)

    # Get the Q-values
    output = dqn(state)

    # Print the Q-values
    print(output)
