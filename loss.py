import torch
import torch.nn as nn
import torch.nn.functional as F

# Contrastive Loss: Encourages diversity among experts
def contrastive_loss(expert_outputs, margin=0.5):
    batch_size, num_experts, feat_dim = expert_outputs.shape

    # Normalize expert outputs to compute cosine similarity
    expert_outputs_norm = F.normalize(expert_outputs, p=2, dim=-1)  # Shape: (batch_size, num_experts, feat_dim)

    # Compute pairwise cosine similarity matrix
    sim_matrix = torch.bmm(expert_outputs_norm, expert_outputs_norm.transpose(1, 2))  # Shape: (batch_size, num_experts, num_experts)

    # Extract upper triangular part (excluding diagonal)
    mask = torch.triu(torch.ones(num_experts, num_experts), diagonal=1).bool()  # Upper triangular mask
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)  # Expand to batch size
    sim_matrix_upper = sim_matrix[mask].view(batch_size, -1)  # Shape: (batch_size, num_experts * (num_experts - 1) / 2)

    # Apply contrastive loss formula
    contrastive_loss = torch.mean(torch.clamp(sim_matrix_upper - margin, min=0))

    return contrastive_loss

def orthogonality_regularization(expert_weights):
    # Stack all expert weight matrices into a single tensor
    expert_weights_stacked = torch.stack(expert_weights, dim=0)  # Shape: (num_experts, in_features, out_features)

    # Compute pairwise dot products between all expert weight matrices
    dot_products = torch.matmul(expert_weights_stacked, expert_weights_stacked.transpose(1, 2))  # Shape: (num_experts, in_features, in_features)

    # Extract upper triangular part (excluding diagonal)
    num_experts = expert_weights_stacked.shape[0]
    mask = torch.triu(torch.ones(num_experts, num_experts), diagonal=1).bool()  # Upper triangular mask
    dot_products_upper = dot_products[mask]  # Shape: (num_pairs, in_features, in_features)

    # Compute Frobenius norm for each pair
    frobenius_norms = torch.norm(dot_products_upper, p='fro', dim=(1, 2))  # Shape: (num_pairs,)

    # Sum the Frobenius norms
    loss = torch.sum(frobenius_norms)
    return loss