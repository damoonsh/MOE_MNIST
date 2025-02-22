class CNNExpert(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(CNNExpert, self).__init__()
        # Define the sequential layers
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding='same'),  # Use 'same' padding
            nn.ReLU(),
            # nn.BatchNorm2d(output_channels),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        return self.layers(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Initialize weights with mean=0 and std=0.01
                init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    # Initialize biases to 0
                    init.zeros_(m.bias)

class CNNGatingNetwork(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, input_size, num_experts=cfg.num_experts):
        super(CNNGatingNetwork, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, output_channels, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Global average pooling
        self.fc = nn.Linear(output_channels * input_size // 4, num_experts)  # Output gating weights

    def forward(self, x):
        # print(x.shape)
        x = F.relu(self.conv1(x)) # (batch_size, original_channel_size, height, width) -> (batch_size, hidden_channels, heigh, width)
        # print(x.shape)
        x = F.relu(self.conv2(x)) # (batch_size, hidden_channels, heigh, width) -> (batch_size, output_channels, heigh, width)
        # print(x.shape)
        x = self.pool(x) # (batch_size, hidden_channels, heigh, width) -> (batch_size, output_channels, heigh // 2, width // 2)
        # print(x.shape)
        x = x.view(x.size(0), -1)  # (batch_size, output_channels, heigh // 2, width // 2) -> (batch_size, output_channels * heigh * width // 4)
        # print(x.shape)
        x = self.fc(x)
        return F.softmax(x, dim=-1)  # Softmax to get gating weights

class MoECNN(nn.Module):
    def __init__(self, input_channels, output_channels, gating_hidden_channels, gating_output_channels, input_size, num_experts=cfg.num_experts, top_k=cfg.top_k):
        super(MoECNN, self).__init__()
        self.gating_network = CNNGatingNetwork(input_channels, gating_hidden_channels, gating_output_channels, input_size, num_experts)
        self.experts = nn.ModuleList([
            CNNExpert(input_channels, output_channels) for _ in range(num_experts)
        ])
        self.output_channels = output_channels
        self.num_experts = num_experts
        self.top_k = top_k
        # self.batch_size = cfg.batch_size
        self.batch_capacity = cfg.batch_size / num_experts  # Capacity per expert per batch

        # Track usage counts and importance
        self.expert_usage_count = torch.zeros(num_experts, device='cuda')  # Track usage count
        self.batch_importance_sum = torch.zeros(num_experts, device='cuda')  # Sum of gating weights per expert
        self.batch_usage_count = torch.zeros(num_experts, device='cuda')  # Usage count for the current batch
        self.batch_overflow_sum = 0.0  # Penalty for exceeding expert capacity
        self.diversity_loss = 0.0  # Penalty for lack of diversity in expert usage

    def forward(self, x):
        # Get gating weights
        gating_weights = self.gating_network(x)  # Shape: (batch_size, num_experts)
        self.batch_importance_sum = gating_weights.sum(dim=0)  # Sum of gating weights for each expert

        # Calculate diversity loss (entropy-based)
        entropy = -torch.sum(gating_weights * torch.log(gating_weights + 1e-10), dim=1)
        self.diversity_loss = -entropy.mean()

        # Select top-k experts
        top_k_weights, top_k_indices = torch.topk(gating_weights, self.top_k, dim=-1)  # Shapes: (batch_size, top_k)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)  # Normalize weights

        # Get outputs from top-k experts
        batch_size, l, height, width = x.shape
        # print(batch_size, l, height, width)
        expert_outputs = torch.zeros(batch_size, self.top_k, self.output_channels, height // 2, width // 2, device=x.device)

        for i in range(self.top_k):
            expert_index = top_k_indices[:, i]  # Shape: (batch_size,)
            expert_output = torch.stack([self.experts[idx](x[b]) for b, idx in enumerate(expert_index)])  # Shape: (batch_size, output_channels, height, width)
            expert_outputs[:, i] = expert_output

        # Combine outputs using top-k weights
        combined_output = torch.einsum('bijkl,bi->bjkl', expert_outputs, top_k_weights)  # Shape: (batch_size, output_channels, height, width)

        # Update expert usage count
        usage_counts = torch.zeros(self.num_experts, device=x.device)
        for i in range(self.top_k):
            usage_counts.scatter_add_(0, top_k_indices[:, i], top_k_weights[:, i].detach())
        self.expert_usage_count += usage_counts.detach()  # Detach to avoid gradients affecting counts
        self.batch_usage_count = usage_counts

        # Calculate overflow penalty
        self.batch_overflow_sum = torch.relu(usage_counts - self.batch_capacity).sum()  # Penalize experts that exceed capacity

        return combined_output, usage_counts  # Return combined output and usage counts

    def load_balance_loss(self):
        # Calculate importance loss (encourage equal importance across experts)
        mean_importance = self.batch_importance_sum.mean()
        importance_loss = F.mse_loss(self.batch_importance_sum, mean_importance * torch.ones_like(self.batch_importance_sum))

        # Combine losses: overflow penalty + importance loss + diversity loss
        load_balance_loss = self.batch_overflow_sum + importance_loss + self.diversity_loss

        return load_balance_loss