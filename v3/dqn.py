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

    def __init__(self, state_dim, action_dim, hidden_dim=256, enable_dueling_dqn=True):
        super(DQN, self).__init__()

        self.enable_dueling_dqn = enable_dueling_dqn

        # input layer implicit in pytorch
        self.fc1 = nn.Linear(state_dim, hidden_dim)  # Hidden layer

        if self.enable_dueling_dqn:
            # Value stream
            self.fc_value = nn.Linear(hidden_dim, 256)  # Value layer
            self.value = nn.Linear(256, 1)
            # Advantage stream
            self.fc_advantage = nn.Linear(hidden_dim, 256)  # Advantage layer
            self.advantage = nn.Linear(256, action_dim)

        else:
            self.output = nn.Linear(hidden_dim, action_dim)  # Output layer

    def forward(self, x):
        """ Forward pass of the network.

            Args:
                x (torch.Tensor): input tensor (state)
            Returns:
                torch.Tensor: output tensor (Q-values)
        """

        x = F.relu(self.fc1(x))  # ReLU activation function for the first layer

        if self.enable_dueling_dqn:
            # Value calculation
            v = F.relu(self.fc_value(x))
            V = self.value(v)
            # Advantage calculation
            a = F.relu(self.fc_advantage(x))
            A = self.advantage(a)

            # Calculate Q-values by combining value and advantage streams
            Q = V + A - torch.mean(A, dim=1, keepdim=True)
        else:
            Q = self.output(x)
        return Q


if __name__ == "__main__":
    # Create a DQN instance
    state_dim = 12
    action_dim = 2
    dqn = DQN(state_dim, action_dim)

    # Create a random state tensor
    state = torch.randn(10, state_dim)

    # Get the Q-values
    output = dqn(state)

    # Print the Q-values
    print(output)
