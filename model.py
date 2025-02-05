import torch
from torch import nn
import torch.nn.functional as F

class DQN(torch.nn.Module):
    """Neural network model for the DQN algorithm.
    A simple linear neural network is used to predict the Q value for each possible action.

    Args:
        torch (torch.nn.Module): Pytorch neural network module 
    """
    def __init__(self, input_dim: int, output_dim: int, network_type: str ='DQN', *args, **kwargs) -> None:
        """Initialize the neural network model for the DQN algorithm.

        Args:
            input_dim (int): Input dimensions of the network
            output_dim (int): Output dimensions of the network
            network_type (str, optional): Type of network to use. Defaults to 'DQN'.
        """

        super().__init__(*args, **kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.network_type = network_type
        self.layer1 = nn.Linear(input_dim,64)
        self.layer2 = nn.Linear(64, 128)
        self.layer3 = nn.Linear(128, 256)
        self.layer4 = nn.Linear(256, 512) 
        self.layer5 = nn.Linear(512, 512)

        if network_type == 'DuelingDQN':  
            # Predicts state value V(s)
            self.state_values = nn.Linear(512, 1)
            # Predicts the advantage of each possible action A(s,a)
            self.advantages = nn.Linear(512, output_dim)
        else:
            # Predicts Q-values directly
            self.output = nn.Linear(512, output_dim)

    def forward(self, x):
        """
        Forward pass to compute Q-values for the given input state.
        
        Args:
            x : Input tensor representing the current state

        Returns:
            Predicted Q-values for each possible action
        """
        x = F.relu6(self.layer1(x))
        x = F.relu6(self.layer2(x))
        x = F.relu6(self.layer3(x))
        x = F.relu6(self.layer4(x))
        x = F.relu6(self.layer5(x))

        if self.network_type == 'DuelingDQN':
            # Dueling DQN predicts both the value of the state and the advantage of each possible action
            # Best action should have advantage of 0
            # Outputs are combined to generate the Q values
            state_values = self.state_values(x)
            advantages = self.advantages(x)
            output = state_values + (advantages - torch.max((advantages), dim=1, keepdim=True)[0])
            return output
        else:
            return self.output(x)