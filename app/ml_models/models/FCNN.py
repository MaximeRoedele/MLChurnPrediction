import torch
from torch import nn

# FCNN Binary Classifier
class FCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Define a fully connected Neural Network using nn.Sequential
        self.NN = nn.Sequential(
            nn.Linear(in_features = 24, out_features = 24),
            nn.ReLU(),
            nn.Linear(in_features = 24, out_features = 1)
        )

    def forward(self, x: torch.Tensor):
        return self.NN(x)