import torch
import torch.nn as nn
import torch.nn.functional as F

# Expert Network: Processes a specific patch of the image
class CNNSpatialExpert(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CNNSpatialExpert, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, out_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(out_channels * 16 * 16, 128)  # Adjust based on patch size

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

# Gating Network: Dynamically weights the experts based on input features
class CNNSpatialGatingNetwork(nn.Module):
    def __init__(self, in_channels, num_experts):
        super(CNNSpatialGatingNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64 * 16 * 16, num_experts)  # Adjust based on image size

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return F.softmax(x, dim=1)  # Softmax to get weights

# Mixture of Experts: Combines outputs of top-k experts using gating network weights
class CNNSpatialMoE(nn.Module):
    def __init__(self, in_channels, out_channels, num_experts, patch_size, top_k):
        super(CNNSpatialMoE, self).__init__()
        self.num_experts = num_experts
        self.patch_size = patch_size
        self.top_k = top_k
        self.experts = nn.ModuleList([CNNSpatialExpert(in_channels, out_channels) for _ in range(num_experts)])
        self.gating_network = CNNSpatialGatingNetwork(in_channels, num_experts)
        self.fc_final = nn.Linear(128, 10)  # Example: 10-class classification

    def forward(self, x):
        # Split image into patches
        patches = [x[:, :, i::self.patch_size, j::self.patch_size] for i in range(self.patch_size) for j in range(self.patch_size)]
        
        # Process each patch with its expert
        expert_outputs = [expert(patch) for expert, patch in zip(self.experts, patches)]
        expert_outputs = torch.stack(expert_outputs, dim=1)  # Shape: (batch_size, num_experts, 128)

        # Compute gating weights
        gating_weights = self.gating_network(x)  # Shape: (batch_size, num_experts)

        # Select top-k experts
        topk_weights, topk_indices = torch.topk(gating_weights, self.top_k, dim=1)  # Shapes: (batch_size, top_k)

        # Normalize top-k weights
        topk_weights = topk_weights / topk_weights.sum(dim=1, keepdim=True)  # Normalize to sum to 1

        # Gather outputs from top-k experts
        topk_expert_outputs = torch.gather(expert_outputs, 1, topk_indices.unsqueeze(-1).expand(-1, -1, expert_outputs.size(-1)))  # Shape: (batch_size, top_k, 128)

        # Weighted combination of top-k expert outputs
        weighted_outputs = torch.sum(topk_expert_outputs * topk_weights.unsqueeze(-1), dim=1)  # Shape: (batch_size, 128)

        # Final prediction
        output = self.fc_final(weighted_outputs)
        return output

# Example usage
model = CNNSpatialMoE(in_channels=3, out_channels=64, num_experts=4, patch_size=2, top_k=2)
input_image = torch.randn(1, 3, 64, 64)  # Example input
output = model(input_image)
print(output.shape)  # Expected output shape: (1, 10)