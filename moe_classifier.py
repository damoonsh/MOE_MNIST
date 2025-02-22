import torch
import torch.nn as nn
import torch.optim as optim

class Expert(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, input_size)
        self._initialize_weights()  # Initialize weights

    def _initialize_weights(self):
        # Initialize weights with normal distribution (mean=0, std=0.001)
        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.001)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.001)
        # Initialize biases to zero (optional, but common practice)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
        
class GatingNetwork(nn.Module):
    def __init__(self, input_size, num_experts):
        super(GatingNetwork, self).__init__()
        self.fc = nn.Linear(input_size, num_experts)
        self._initialize_weights()  # Initialize weights

    def _initialize_weights(self):
        # Initialize weights with normal distribution (mean=0, std=0.001)
        nn.init.normal_(self.fc.weight, mean=0.0, std=0.001)
        # Initialize biases to zero (optional, but common practice)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        return F.softmax(self.fc(x), dim=-1)

class MoELayer(nn.Module):
    def __init__(self, input_size, hidden_size, num_experts=cfg.num_experts, top_k=cfg.top_k):
        super(MoELayer, self).__init__()
        self.experts = nn.ModuleList([Expert(input_size, hidden_size) for _ in range(num_experts)])
        self.gating_network = GatingNetwork(input_size, num_experts)
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_usage_count = torch.zeros(num_experts, device=device)  # Track usage count

        self.batch_capaity = cfg.batch_size / cfg.num_experts

    def forward(self, x):
        # Get gating weights
        gating_weights = self.gating_network(x)  # Shape: (batch_size, num_experts)
        self.batch_importance_sum = gating_weights.sum(dim=0)  # Sum of gating weights for each expert
        entropy = -torch.sum(gating_weights * torch.log(gating_weights + 1e-10), dim=1)
        self.diversity_loss = -entropy.mean()

        # Select top-k experts
        top_k_weights, top_k_indices = torch.topk(gating_weights, self.top_k, dim=-1)  # Shapes: (batch_size, top_k)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)  # Normalize weights

        # Get outputs from only the top-k experts
        batch_size = x.size(0)
        top_k_expert_outputs = torch.zeros(batch_size, self.top_k, x.size(-1), device=x.device)  # Shape: (batch_size, top_k, output_size)

        for i in range(self.top_k):
            # Gather the outputs of the i-th top expert for all samples in the batch
            expert_indices = top_k_indices[:, i]  # Shape: (batch_size,)
            expert_outputs = torch.stack([self.experts[idx](x[b]) for b, idx in enumerate(expert_indices)])  # Shape: (batch_size, output_size)
            top_k_expert_outputs[:, i, :] = expert_outputs

        # Combine outputs using top-k weights with einsum
        combined_output = torch.einsum('ijk,ij->ik', top_k_expert_outputs, top_k_weights)  # Shape: (batch_size, output_size)

        # Update expert usage count
        usage_counts = torch.zeros(self.num_experts, device=device)
        for i in range(self.top_k):
            usage_counts.scatter_add_(0, top_k_indices[:, i], top_k_weights[:, i].detach())
        self.expert_usage_count += usage_counts.detach()  # Detach to avoid gradients affecting counts
        self.batch_usage_count = usage_counts

        self.batch_overflow_sum = torch.relu(usage_counts - self.batch_capaity).sum()  # Penalize experts that exceed capacity

        return combined_output, usage_counts  # Return combined output and usage counts

    def load_balance_loss(self):
        # Calculate load balancing loss based on usage counts
        mean_importance = self.batch_importance_sum.mean()
        importance_loss = F.mse_loss(self.batch_importance_sum, mean_importance * torch.ones_like(self.batch_importance_sum))
        
        # print(self.diversity_loss.item(), importance_loss.item(), self.batch_overflow_sum.item())

        load_balance_loss = self.batch_overflow_sum + importance_loss + self.diversity_loss
        
        return load_balance_loss

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

        # MoE Layer
        self.moe_layer = MoELayer(input_size=64 * 7 * 7, hidden_size=64, num_experts=5)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(in_features=64 * 7 * 7, out_features=128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=64, out_features=10)
        )

    def forward(self, x):
        # Feature extraction
        x = self.features(x)  # Shape: (batch_size, 64, 7, 7)

        # Flatten the output for the MoE layer
        x = x.view(x.size(0), -1)  # Shape: (batch_size, 64 * 7 * 7)

        # Pass through the MoE layer
        moe_output, usage_counts = self.moe_layer(x)  # Shape: (batch_size, 64 * 7 * 7)

        # Pass through the classifier
        x_classified = self.classifier(moe_output)  # Shape: (batch_size, 10)

        return x_classified, usage_counts  # Return classifier output and usage counts

    def compute_total_loss(self, outputs, targets):
        """Compute total loss including load balance loss."""
        task_loss = F.cross_entropy(outputs, targets)  # Cross-entropy loss for classification
        load_balance_loss = self.moe_layer.load_balance_loss()  # Load balance loss
        
        # Combine losses with a coefficient for load balancing
        total_loss = task_loss + cfg.load_balance_coef * load_balance_loss
        return total_loss