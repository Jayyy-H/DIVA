import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DIVAConfig:
    def __init__(self):
        # config of stage
        self.stage = 2
        
        # layers
        self.middle_layer_idx = 16  
        self.hidden_dim = 2048      
        self.bottleneck_dim = 256   
        
        # loss weight
        self.temp = 0.07            
        self.lambda_align = 0.1      
        self.lambda_dis = 0.01      
        self.lr_club = 1e-4         

class GatedMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_ratio=2):
        super().__init__()
        
        hidden_features = int(input_dim * hidden_ratio)
        
        self.fc1 = nn.Linear(input_dim, hidden_features * 2)
        
        self.fc2 = nn.Linear(hidden_features, output_dim)
        
        self.act = nn.SiLU()

    def forward(self, x):
        x_up = self.fc1(x)
        
        gate, value = x_up.chunk(2, dim=-1)
        
        x_gated = self.act(gate) * value
        
        # Output projection
        return self.fc2(x_gated)

class CLUB(nn.Module):
    
    def __init__(self, x_dim, y_dim, hidden_size=512):
        super().__init__()
        # predict q(y|x) 
        # predict y (Unique) 
        self.p_mu = nn.Sequential(
            nn.Linear(x_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, y_dim)
        )
        
        self.p_logvar = nn.Sequential(
            nn.Linear(x_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, y_dim),
            nn.Tanh() # contraint logvar
        )

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def loglikeli(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        
        # Gaussian Log Likelihood:
        # -(1/2) * [log(2pi) + logvar + (y-mu)^2 / exp(logvar)]
        return (-(0.5 * (logvar + torch.pow(y_samples - mu, 2) / logvar.exp()))).sum(dim=1).mean()

    def mi_est(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        
        sample_size = x_samples.shape[0]
        random_index = torch.randperm(sample_size).long()
        
        # Positive pairs: (x_i, y_i) -> p(x,y)
        positive = - (mu - y_samples)**2 / logvar.exp()
        
        # Negative pairs: (x_i, y_j) -> p(x)p(y) (Shuffle y)
        prediction_1 = mu.unsqueeze(1)          # [B, 1, D]
        y_samples_1 = y_samples.unsqueeze(0)    # [1, B, D]

        # negative = - ((y_samples_1 - prediction_1)**2).mean(dim=1) / logvar.exp() 

        y_shuffle = y_samples[random_index]
        negative = - (mu - y_shuffle)**2 / logvar.exp()
        
        # CLUB Formula: E[log p(y|x)] - E[log p(y)]
        return (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()

def info_nce_loss(features_a, features_b, temperature=0.07):
    # L2 Normalize
    features_a = F.normalize(features_a, dim=1)
    features_b = F.normalize(features_b, dim=1)
    
    # Cosine Similarity Matrix: [B, B]
    logits = torch.matmul(features_a, features_b.T) / temperature
    
    # Labels: Diagonal elements are positive pairs
    labels = torch.arange(logits.shape[0], device=logits.device)
    
    # Symmetric Loss
    loss_a = F.cross_entropy(logits, labels)
    loss_b = F.cross_entropy(logits.T, labels)
    
    return (loss_a + loss_b) / 2

def orthogonality_loss(shared_feat, unique_feat):
    # orthogonality
    cosine_sim = F.cosine_similarity(shared_feat, unique_feat, dim=-1)
    return torch.mean(cosine_sim ** 2)
