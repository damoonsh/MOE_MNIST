import torch
import torch.nn as nn
import torch.optim as optim

class Model(nn.Module):
    def __init__(self, input_features):
        super().__init__()

        # Feature extraction (CNN)
        self.features = nn.Sequential(
            nn.Conv2d(input_features, out_channels=32, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, out_channels=64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(in_features=64 * 7 * 7, out_features=128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=64, out_features=10),
            nn.Softmax()
        )

    def forward(self, x):
        # Feature extraction
        x = self.features(x)  # Shape: (batch_size, 64, 7, 7)

        # Flatten the output for the MoE layer
        x = x.view(x.size(0), -1)  # Shape: (batch_size, 64 * 7 * 7)

        # Pass through the classifier
        x_classified = self.classifier(x)  # Shape: (batch_size, 10)

        return x_classified # Return classifier output and usage counts

    def compute_total_loss(self, outputs, targets):
        """Compute total loss including load balance loss."""
        return  F.cross_entropy(outputs, targets) 