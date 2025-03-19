import torch
import torch.nn as nn
import torch.optim as optim

class AcousticModel(nn.Module):
    """
    A simple neural network model for chord recognition.
    The model takes in a feature vector of size input_dim and outputs a log-softmax of the chord classes.
    Args:
        input_dim (int): The dimension of the input features.
        hidden_dim (int): The dimension of the hidden layer.
        num_chords (int): The number of possible chord classes.
    """
    def __init__(self, input_dim, hidden_dim, num_chords):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_chords),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):
        return self.layers(x)

