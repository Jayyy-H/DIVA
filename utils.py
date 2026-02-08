import torch
import torch.nn as nn
import torch.nn.functional as F

class DIVAConfig:
    def __init__(self):
        self.stage = 2  
        self.middle_layer_idx = 16  
        self.hidden_dim = 2048 
        self.bottleneck_dim = 256 
        self.temp = 0.07 
        self.lambda_align = 0.1 
        self.lambda_dis = 0.01  

class TinyEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, input_dim) 
        )

    def forward(self, x):
        return self.net(x)

class CLUBDiscriminator(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x, y):
        # x: [B, D], y: [B, D]
        return self.net(torch.cat([x, y], dim=-1))

def info_nce_loss(features_a, features_b, temperature=0.07):
    """
    Shared Feature Alignment
    features_a: [Batch, Dim] (Understanding Shared)
    features_b: [Batch, Dim] (Generation Shared)
    """
    # normlization
    features_a = F.normalize(features_a, dim=1)
    features_b = F.normalize(features_b, dim=1)
    
    logits = torch.matmul(features_a, features_b.T) / temperature
    
    labels = torch.arange(logits.shape[0], device=logits.device)
    
    loss_a = F.cross_entropy(logits, labels)
    loss_b = F.cross_entropy(logits.T, labels)
    
    return (loss_a + loss_b) / 2

def orthogonality_loss(shared_feat, unique_feat):
    cosine_sim = F.cosine_similarity(shared_feat, unique_feat, dim=-1)
    return torch.mean(cosine_sim ** 2)
