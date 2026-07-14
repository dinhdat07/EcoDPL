import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math

class DWTExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # Haar Wavelet kernels
        # LL is low frequency (content), we discard it
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]])
        hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]])
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])
        
        # Shape: (out_channels, in_channels/groups, H, W)
        # We process grayscale image (1 channel) to get 3 high-freq subbands
        kernel = torch.stack([lh, hl, hh], dim=0).unsqueeze(1)
        
        self.conv = nn.Conv2d(1, 3, kernel_size=2, stride=2, padding=0, bias=False)
        self.conv.weight.data = kernel
        self.conv.weight.requires_grad = False
        
        # Load Frozen VGG16
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.feature_extractor = vgg.features[:16] # Up to conv3_3 or pool3
        
        # Freeze VGG
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        # x: [B, 3, H, W] RGB image
        with torch.no_grad():
            # Convert to grayscale
            gray = 0.2989 * x[:, 0:1, :, :] + 0.5870 * x[:, 1:2, :, :] + 0.1140 * x[:, 2:3, :, :]
            
            # Extract high-freq subbands: LH, HL, HH
            high_freqs = self.conv(gray) # [B, 3, H/2, W/2]
            
            # CRITICAL FIX: Haar DWT outputs negative values for edges in certain directions.
            # VGG's ReLU will kill 50% of the edge information if we feed it raw DWT outputs.
            # 1. Take absolute value to get edge magnitude.
            high_freqs = torch.abs(high_freqs)
            
            # 2. Normalize using ImageNet statistics to match VGG16's pre-training domain
            mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
            high_freqs = (high_freqs - mean) / std
            
            # Pass through VGG
            features = self.feature_extractor(high_freqs)
            
            # Global Average Pooling
            features = F.adaptive_avg_pool2d(features, (1, 1)).view(features.size(0), -1)
            
            # L2 Normalize
            features = F.normalize(features, p=2, dim=1)
            return features


class GMMStatisticalMemory(nn.Module):
    def __init__(self, feature_dim=256, num_tasks=2, n_components=3, momentum=0.99):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_tasks = num_tasks
        self.n_components = n_components
        self.momentum = momentum
        
        # Buffers for GMM parameters per task
        self.register_buffer("means", torch.zeros(num_tasks, n_components, feature_dim))
        self.register_buffer("inv_covs", torch.eye(feature_dim).unsqueeze(0).unsqueeze(0).repeat(num_tasks, n_components, 1, 1))
        self.register_buffer("weights", torch.ones(num_tasks, n_components) / n_components)
        self.register_buffer("is_fitted", torch.zeros(num_tasks, dtype=torch.bool))

        # Background distribution (for Relative Mahalanobis)
        self.register_buffer("bg_mean", torch.zeros(feature_dim))
        self.register_buffer("bg_inv_cov", torch.eye(feature_dim))
        
    @torch.no_grad()
    def fit(self, task_id, features):
        """Fit GMM for a specific task using K-Means initialization + EM."""
        if features.size(0) < self.n_components:
            raise ValueError(f"Not enough features ({features.size(0)}) to fit {self.n_components} components.")
            
        features = F.normalize(features, p=2, dim=1) # Ensure L2 norm
        B, D = features.shape
        K = self.n_components
        
        # 1. K-Means Initialization
        # Randomly select initial centroids
        indices = torch.randperm(B)[:K]
        centroids = features[indices].clone()
        assignments = torch.zeros(B, dtype=torch.long, device=features.device)
        
        for _ in range(10): # 10 iterations of K-Means
            # Distances: [B, K]
            dists = torch.cdist(features, centroids)
            assignments = torch.argmin(dists, dim=1)
            
            for k in range(K):
                if (assignments == k).sum() > 0:
                    centroids[k] = features[assignments == k].mean(dim=0)
            centroids = F.normalize(centroids, p=2, dim=1)
            
        # 2. Compute Covariance and Weights for each cluster
        eps = 1e-4
        for k in range(K):
            mask = (assignments == k)
            N_k = mask.sum().float()
            if N_k > 1:
                cluster_feats = features[mask]
                mean_k = cluster_feats.mean(dim=0)
                # Covariance: (X - \mu)^T (X - \mu) / (N - 1)
                centered = cluster_feats - mean_k
                cov_k = (centered.t() @ centered) / (N_k - 1) + torch.eye(D, device=features.device) * eps
                
                self.means[task_id, k] = mean_k
                self.inv_covs[task_id, k] = torch.linalg.inv(cov_k)
                self.weights[task_id, k] = N_k / B
            else:
                # Fallback if cluster is empty or has 1 sample
                self.means[task_id, k] = centroids[k]
                self.inv_covs[task_id, k] = torch.eye(D, device=features.device) / eps
                self.weights[task_id, k] = 1e-5
                
        # Normalize weights
        self.weights[task_id] /= self.weights[task_id].sum()
        self.is_fitted[task_id] = True
        
        # Update Background Distribution (Global running average)
        if not self.is_fitted.any():
            self.bg_mean.copy_(features.mean(dim=0))
            cov = torch.cov(features.t()) + torch.eye(D, device=features.device) * eps
            self.bg_inv_cov.copy_(torch.linalg.inv(cov))
        else:
            self.bg_mean = self.momentum * self.bg_mean + (1 - self.momentum) * features.mean(dim=0)
            # Recompute a naive global cov
            cov = torch.cov(features.t()) + torch.eye(D, device=features.device) * eps
            self.bg_inv_cov = self.momentum * self.bg_inv_cov + (1 - self.momentum) * torch.linalg.inv(cov)

    @torch.no_grad()
    def predict_task_probs(self, features, temperature=1.0):
        """Predict task probabilities for given features using Relative Mahalanobis."""
        features = F.normalize(features, p=2, dim=1)
        B, D = features.shape
        
        # Calculate background score
        # dist: (X-\mu)^T \Sigma^{-1} (X-\mu)
        diff_bg = features - self.bg_mean.unsqueeze(0) # [B, D]
        # (B, 1, D) @ (D, D) @ (B, D, 1) -> (B, 1, 1)
        mahal_bg = torch.bmm(diff_bg.unsqueeze(1), self.bg_inv_cov.unsqueeze(0).expand(B, -1, -1))
        mahal_bg = torch.bmm(mahal_bg, diff_bg.unsqueeze(2)).squeeze() # [B]
        # If B=1, squeeze() might remove the batch dimension, creating a scalar
        if B == 1:
            mahal_bg = mahal_bg.view(1)
            
        scores = []
        for t in range(self.num_tasks):
            if not self.is_fitted[t]:
                scores.append(torch.full((B,), -1e9, device=features.device))
                continue
                
            log_probs_k = []
            for k in range(self.n_components):
                diff = features - self.means[t, k].unsqueeze(0) # [B, D]
                inv_cov = self.inv_covs[t, k] # [D, D]
                
                mahal_k = torch.bmm(diff.unsqueeze(1), inv_cov.unsqueeze(0).expand(B, -1, -1))
                mahal_k = torch.bmm(mahal_k, diff.unsqueeze(2)).squeeze() # [B]
                if B == 1:
                    mahal_k = mahal_k.view(1)
                
                # Log-likelihood roughly proportional to -0.5 * mahal
                log_p = torch.log(self.weights[t, k].clamp_min(1e-9)) - 0.5 * mahal_k
                log_probs_k.append(log_p)
                
            # Marginalize over K components using LogSumExp
            log_probs_k = torch.stack(log_probs_k, dim=1) # [B, K]
            task_score = torch.logsumexp(log_probs_k, dim=1) # [B]
            
            # Relative score: Task Score - (-0.5 * mahal_bg)
            relative_score = task_score + 0.5 * mahal_bg
            scores.append(relative_score)
            
        scores = torch.stack(scores, dim=1) # [B, num_tasks]
        
        # If no tasks are fitted, return uniform distribution
        if not self.is_fitted.any():
            return torch.ones_like(scores) / self.num_tasks
            
        probs = F.softmax(scores / temperature, dim=1)
        return probs
