import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.model import (
    Downsample,
    OverlapPatchEmbed,
    TransformerBlock,
    Upsample,
)


# ---------------------------------------------------------------------------
# Fix #1: DWT (Haar Wavelet) for Content-Degradation Disentanglement
# ---------------------------------------------------------------------------

def haar_dwt2d(x):
    """Apply single-level 2D Haar Discrete Wavelet Transform.

    Args:
        x: Tensor of shape [B, C, H, W].

    Returns:
        ll, lh, hl, hh: Each of shape [B, C, H//2, W//2].
    """
    # Pad to even dimensions if necessary
    _, _, h, w = x.shape
    if h % 2 != 0:
        x = F.pad(x, (0, 0, 0, 1), mode="reflect")
    if w % 2 != 0:
        x = F.pad(x, (0, 1, 0, 0), mode="reflect")

    x_lo = x[:, :, 0::2, :] + x[:, :, 1::2, :]  # Row-wise low
    x_hi = x[:, :, 0::2, :] - x[:, :, 1::2, :]  # Row-wise high

    ll = (x_lo[:, :, :, 0::2] + x_lo[:, :, :, 1::2]) * 0.5
    lh = (x_lo[:, :, :, 0::2] - x_lo[:, :, :, 1::2]) * 0.5
    hl = (x_hi[:, :, :, 0::2] + x_hi[:, :, :, 1::2]) * 0.5
    hh = (x_hi[:, :, :, 0::2] - x_hi[:, :, :, 1::2]) * 0.5

    return ll, lh, hl, hh


def extract_degradation_features(x):
    """Extract degradation-only features by discarding LL (content) sub-band.

    Uses the LH, HL, HH sub-bands which capture directional high-frequency
    information (horizontal, vertical, diagonal rain streaks).

    Args:
        x: Rainy image tensor [B, 3, H, W], values in [0, 1].

    Returns:
        Tensor of shape [B, 3, H//2, W//2] containing degradation info only.
    """
    _, lh, hl, hh = haar_dwt2d(x)
    # Combine high-freq sub-bands: average across input channels, stack as 3ch
    # Each sub-band captures a different direction of rain streaks
    return torch.cat([
        lh.mean(dim=1, keepdim=True),  # Horizontal edges (rain direction)
        hl.mean(dim=1, keepdim=True),  # Vertical edges (rain direction)
        hh.mean(dim=1, keepdim=True),  # Diagonal edges
    ], dim=1)


# ---------------------------------------------------------------------------
# Fix #3: GMM Statistical Memory (K components + L2-Norm + Relative Distance)
# ---------------------------------------------------------------------------

class GMMStatisticalMemory(nn.Module):
    """Gaussian Mixture Model memory for task-free degradation routing.

    Stores K Gaussian components per task, each with its own mean, inverse
    covariance, and mixing weight. Uses L2-normalized features and relative
    Mahalanobis distance for robust task identification.

    References:
        - CoGaMiD (NeurIPS/OpenReview): GMM for continual segmentation
        - Mahalanobis++ (OpenReview 2024): L2-norm before Mahalanobis
        - RMD (UCL): Relative Mahalanobis Distance
    """

    def __init__(self, feature_dim, max_tasks=10, num_components=3):
        super().__init__()
        self.feature_dim = feature_dim
        self.max_tasks = max_tasks
        self.K = num_components

        # Per-task GMM parameters: K components each
        self.register_buffer(
            "means", torch.zeros(max_tasks, num_components, feature_dim))
        self.register_buffer(
            "inv_covs", torch.zeros(max_tasks, num_components, feature_dim, feature_dim))
        self.register_buffer(
            "mix_weights", torch.zeros(max_tasks, num_components))
        self.register_buffer(
            "task_count", torch.tensor(0, dtype=torch.long))

        # Background distribution for Relative Mahalanobis Distance
        self.register_buffer("bg_mean", torch.zeros(feature_dim))
        self.register_buffer("bg_inv_cov", torch.zeros(feature_dim, feature_dim))
        self.register_buffer("bg_valid", torch.tensor(False, dtype=torch.bool))

    @torch.no_grad()
    def update_statistics(self, task_id, features):
        """Fit K-component GMM to L2-normalized features for a task.

        Uses K-Means initialization followed by EM-style assignment to fit
        GMM components. This avoids sklearn dependency.
        """
        features = F.normalize(features, dim=1)
        N, D = features.shape

        # --- K-Means initialization ---
        indices = torch.randperm(N, device=features.device)[:self.K]
        centroids = features[indices].clone()

        for _ in range(20):  # K-Means iterations
            dists = torch.cdist(features, centroids)  # [N, K]
            assignments = dists.argmin(dim=1)  # [N]
            for k in range(self.K):
                mask = assignments == k
                if mask.sum() > 0:
                    centroids[k] = features[mask].mean(dim=0)

        # --- Compute per-component statistics ---
        final_dists = torch.cdist(features, centroids)
        assignments = final_dists.argmin(dim=1)

        for k in range(self.K):
            mask = assignments == k
            count = mask.sum().item()
            if count < D + 1:
                # Not enough samples, fall back to centroid + identity cov
                self.means[task_id, k] = centroids[k]
                self.inv_covs[task_id, k] = torch.eye(D, device=features.device)
                self.mix_weights[task_id, k] = max(count, 1) / N
                continue

            cluster_features = features[mask]
            mean = cluster_features.mean(dim=0)
            centered = cluster_features - mean
            cov = (centered.t() @ centered) / (count - 1 + 1e-6)
            cov += torch.eye(D, device=features.device) * 1e-4  # Ridge
            inv_cov = torch.linalg.inv(cov)

            self.means[task_id, k] = mean
            self.inv_covs[task_id, k] = inv_cov
            self.mix_weights[task_id, k] = count / N

        if task_id >= self.task_count.item():
            self.task_count.fill_(task_id + 1)

        # --- Update background distribution (all tasks combined) ---
        self._update_background(features)

    @torch.no_grad()
    def _update_background(self, new_features):
        """Update running background distribution for Relative Mahalanobis."""
        N, D = new_features.shape
        bg_mean = new_features.mean(dim=0)
        centered = new_features - bg_mean
        bg_cov = (centered.t() @ centered) / (N - 1 + 1e-6)
        bg_cov += torch.eye(D, device=new_features.device) * 1e-4

        if self.bg_valid.item():
            # Exponential moving average with existing background
            alpha = 0.5
            self.bg_mean.mul_(1 - alpha).add_(bg_mean * alpha)
            old_cov = torch.linalg.inv(self.bg_inv_cov)
            blended_cov = (1 - alpha) * old_cov + alpha * bg_cov
            self.bg_inv_cov.copy_(torch.linalg.inv(blended_cov))
        else:
            self.bg_mean.copy_(bg_mean)
            self.bg_inv_cov.copy_(torch.linalg.inv(bg_cov))
            self.bg_valid.fill_(True)

    def compute_task_scores(self, features):
        """Compute relative GMM log-likelihood scores for task routing.

        Returns:
            task_probs: [B, num_tasks] probability vector via softmax.
        """
        features = F.normalize(features, dim=1)
        B, D = features.shape
        num_tasks = self.task_count.item()
        if num_tasks == 0:
            return None

        # --- GMM log-likelihood per task ---
        task_scores = []
        for t in range(num_tasks):
            component_scores = []
            for k in range(self.K):
                w = self.mix_weights[t, k].clamp_min(1e-8)
                mean = self.means[t, k]
                inv_cov = self.inv_covs[t, k]
                diff = features - mean
                mahal = ((diff @ inv_cov) * diff).sum(dim=1)  # [B]
                log_prob = torch.log(w) - 0.5 * mahal  # [B]
                component_scores.append(log_prob.unsqueeze(1))
            # LogSumExp over K components: marginalize
            stacked = torch.cat(component_scores, dim=1)  # [B, K]
            task_score = torch.logsumexp(stacked, dim=1)   # [B]
            task_scores.append(task_score.unsqueeze(1))

        task_scores = torch.cat(task_scores, dim=1)  # [B, num_tasks]

        # --- Relative Mahalanobis: subtract background score ---
        if self.bg_valid.item():
            diff_bg = features - self.bg_mean
            bg_mahal = ((diff_bg @ self.bg_inv_cov) * diff_bg).sum(dim=1, keepdim=True)
            bg_score = -0.5 * bg_mahal  # [B, 1]
            task_scores = task_scores - bg_score

        # Temperature-scaled softmax
        temperature = max(D, 1)
        task_probs = F.softmax(task_scores / temperature, dim=1)
        return task_probs

class SFTAdapter(nn.Module):
    """Spatial Feature Transform (SFT) for adapting frozen backbone.
    
    Transforms prompts into affine transformation parameters (gamma, beta)
    to modulate the backbone features.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv_gamma = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1)
        )
        self.conv_beta = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1)
        )

    def forward(self, x, prompt):
        gamma = self.conv_gamma(prompt)
        beta = self.conv_beta(prompt)
        return x * (1 + gamma) + beta


# ---------------------------------------------------------------------------
# Fix #2: PromptFuser with Soft Mask Gating (replaces Logit Fusion)
# ---------------------------------------------------------------------------

class PromptFuser(nn.Module):
    """Adaptive prompt fusion with Soft Mask Gating.

    Instead of additive Logit Fusion (which contradicts instance-adaptive
    selection), we use multiplicative Soft Mask Gating. The statistical
    memory produces a task probability vector, which is converted to a
    per-prompt soft mask. This mask gates (multiplies) the P-Fuser logits,
    preserving the instance-adaptive nature of cosine similarity matching
    while constraining prompts to the correct task region.

    References:
        - BID-LoRA: Mask-based pathway protection
        - CODA-Prompt (CVPR 2023): Attention-based prompt assembly
    """

    def __init__(self, num_prompts, query_dim, value_shape, temperature=1.0):
        super().__init__()
        self.num_prompts = num_prompts
        self.query_dim = query_dim
        self.temperature = temperature
        value_dim = math.prod(value_shape)

        self.keys = nn.Parameter(torch.randn(num_prompts, query_dim) * 0.02)
        self.attention = nn.Parameter(torch.ones(num_prompts, query_dim))
        self.values = nn.Parameter(torch.randn(num_prompts, *value_shape) * 0.02)
        self.register_buffer("frequency", torch.ones(num_prompts), persistent=True)
        self.register_buffer("active", torch.ones(num_prompts, dtype=torch.bool), persistent=False)

        # Soft Mask Gating: stores normalized frequency profile per task
        # task_prompt_mask[t, p] = how relevant prompt p is to task t
        self.register_buffer(
            "task_prompt_mask", torch.zeros(10, num_prompts), persistent=True)
            
        # Hard-freezing buffer: stores which prompts belong exclusively to old tasks
        self.register_buffer(
            "protected_mask", torch.zeros(num_prompts, dtype=torch.bool), persistent=True)
            
        self.backup_keys = None
        self.backup_values = None
        self.backup_attention = None

    @torch.no_grad()
    def backup_protected_prompts(self):
        """Backup protected prompts before training."""
        self.backup_keys = self.keys.data.clone()
        self.backup_values = self.values.data.clone()
        self.backup_attention = self.attention.data.clone()

    @torch.no_grad()
    def restore_protected_prompts(self):
        """Restore protected prompts to combat optimizer weight decay."""
        if self.backup_keys is not None and self.protected_mask.any():
            mask = self.protected_mask
            self.keys.data[mask] = self.backup_keys[mask]
            self.values.data[mask] = self.backup_values[mask]
            self.attention.data[mask] = self.backup_attention[mask]

    def forward(self, query, update_frequency=False, soft_mask=None):
        """Forward pass with optional soft mask gating.

        Args:
            query: [B, C] feature query.
            update_frequency: Whether to update prompt usage counts.
            soft_mask: [B, num_prompts] soft mask from statistical router.
                       Values in [0, 1]. If None, no gating is applied.
        """
        if query.dim() != 2:
            raise ValueError(f"PromptFuser expects [B, C] query, got {tuple(query.shape)}")

        query = F.normalize(query, dim=1)
        keys = F.normalize(self.keys, dim=1)
        attended = F.normalize(query[:, None, :] * self.attention[None, :, :], dim=2)
        logits = (attended * keys[None, :, :]).sum(dim=2) / self.temperature
        if self.active is not None:
            logits = logits.masked_fill(~self.active.to(logits.device)[None, :], -1e4)

        raw_logits = logits.clone()

        # --- Additive Soft Mask Gating (Fix #2 + Fix Flaw) ---
        # Additive log-masking mathematically zeros out Softmax for irrelevant tasks
        # preserving P-Fuser's instance-adaptive selection cleanly.
        if soft_mask is not None:
            # soft_mask: [B, num_prompts], values typically in [0.0, 1.0]
            # Add log of mask. If mask -> 0, penalty -> -inf (blocks prompt entirely)
            mask_penalty = torch.log(soft_mask.clamp(min=1e-6))
            logits = logits + mask_penalty

        weights = F.softmax(logits, dim=1)
        fused = torch.einsum("bm,m...->b...", weights, self.values)

        # Distance surrogate from the paper, weighted by the fused prompt weights.
        cosine_distance = 1.0 - raw_logits.clamp(-1.0, 1.0)
        distance_loss = (weights.detach() * cosine_distance).sum(dim=1).mean()

        top_indices = torch.argmax(weights.detach(), dim=1)
        if update_frequency and self.training:
            with torch.no_grad():
                counts = torch.bincount(top_indices, minlength=self.num_prompts)
                self.frequency.add_(counts.to(self.frequency.device))

        return fused, {
            "weights": weights,
            "logits": raw_logits,
            "top_indices": top_indices,
            "distance_loss": distance_loss,
        }

    @torch.no_grad()
    def update_task_mask(self, task_id):
        """Build task-prompt mask from accumulated frequency counts.

        Normalizes the current frequency buffer to [0, 1] and stores it
        as the affinity profile for this task. Also locks heavily used prompts.
        """
        freq = self.frequency.float()
        max_freq = freq.max().clamp_min(1.0)
        self.task_prompt_mask[task_id] = freq / max_freq
        
        # Lock heavily used prompts (e.g. > 50% relative usage) to prevent weight decay
        new_protected = (freq / max_freq) > 0.5
        self.protected_mask = self.protected_mask | new_protected
        
        # Reset frequency for the next task
        self.frequency.fill_(1.0)

    def compute_soft_mask(self, task_probs):
        """Convert task probabilities to per-prompt soft mask.

        Args:
            task_probs: [B, num_tasks] from GMMStatisticalMemory.

        Returns:
            soft_mask: [B, num_prompts] values in [0, 1].
        """
        T = task_probs.shape[1]
        # Weighted combination of per-task prompt masks
        mask = task_probs @ self.task_prompt_mask[:T, :].to(task_probs.device)
        # Normalize to [0, 1]
        mask_max = mask.max(dim=1, keepdim=True).values.clamp_min(1e-6)
        return mask / mask_max

    @torch.no_grad()
    def set_active_range(self, start=None, end=None):
        self.active.zero_()
        start = 0 if start is None else max(0, int(start))
        end = self.num_prompts if end is None else min(self.num_prompts, int(end))
        if end <= start:
            raise ValueError(f"Invalid prompt range: {start}:{end}")
        self.active[start:end] = True

    @torch.no_grad()
    def clear_active_range(self):
        self.active.fill_(True)


# ---------------------------------------------------------------------------
# Main Model: EcoDPL + all 3 fixes integrated
# ---------------------------------------------------------------------------

class EcoDPLPromptIR(nn.Module):
    """EcoSMH: EcoDPL enhanced with Statistical Memory and 3 vulnerability fixes.

    Fix #1: DWT Haar Wavelet extracts degradation-only features (discards LL).
    Fix #2: Soft Mask Gating replaces additive Logit Fusion in PromptFuser.
    Fix #3: GMM (K=3) + L2-Norm + Relative Mahalanobis replaces single Gaussian.

    The backbone follows the PromptIR/Restormer-style implementation already in
    this repository, while the continual-learning interface mirrors the TIP
    paper: image prompts, feature prompts, P-Fuser, frequency tables,
    Grad-Tuner, and optional parameter regularization from the trainer.
    """

    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=None,
        num_refinement_blocks=4,
        heads=None,
        ffn_expansion_factor=2.66,
        bias=False,
        layer_norm_type="WithBias",
        num_prompts=100,
        image_prompt_size=32,
        grad_tuner_components=25,
        gmm_components=3,
    ):
        super().__init__()
        if num_blocks is None:
            num_blocks = [4, 6, 6, 8]
        if heads is None:
            heads = [1, 2, 4, 8]

        self.num_prompts = num_prompts
        self.grad_tuner_components = grad_tuner_components

        # Fix #3: GMM Statistical Memory (replaces single-Gaussian StatisticalMemory)
        self.stat_memory = GMMStatisticalMemory(
            feature_dim=256, max_tasks=10, num_components=gmm_components)

        # Fix #1: Frozen VGG16 for degradation feature extraction
        # Input will be DWT high-freq sub-bands instead of raw images
        from torchvision.models import VGG16_Weights, vgg16
        self.frozen_extractor = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16].eval()
        for param in self.frozen_extractor.parameters():
            param.requires_grad = False

        self.register_buffer("vgg_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("vgg_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.patch_embed_query = OverlapPatchEmbed(inp_channels, dim)
        self.image_fuser = PromptFuser(
            num_prompts=num_prompts,
            query_dim=dim,
            value_shape=(inp_channels, image_prompt_size, image_prompt_size),
        )
        self.image_prompt_adapter = SFTAdapter(inp_channels)

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=layer_norm_type)
            for _ in range(num_blocks[0])
        ])

        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=layer_norm_type)
            for _ in range(num_blocks[1])
        ])

        self.down2_3 = Downsample(int(dim * 2))
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 4), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=layer_norm_type)
            for _ in range(num_blocks[2])
        ])

        self.down3_4 = Downsample(int(dim * 4))
        latent_dim = int(dim * 8)
        self.latent = nn.Sequential(*[
            TransformerBlock(dim=latent_dim, num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=layer_norm_type)
            for _ in range(num_blocks[3])
        ])

        self.feature_fuser = PromptFuser(
            num_prompts=num_prompts,
            query_dim=latent_dim,
            value_shape=(latent_dim, 1, 1),
        )
        self.feature_prompt_adapter = SFTAdapter(latent_dim)

        self.up4_3 = Upsample(latent_dim)
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 8), int(dim * 4), 1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 4), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=layer_norm_type)
            for _ in range(num_blocks[2])
        ])

        self.up3_2 = Upsample(int(dim * 4))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 4), int(dim * 2), 1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=layer_norm_type)
            for _ in range(num_blocks[1])
        ])

        self.up2_1 = Upsample(int(dim * 2))
        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=layer_norm_type)
            for _ in range(num_blocks[0])
        ])

        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=layer_norm_type)
            for _ in range(num_refinement_blocks)
        ])
        self.output = nn.Conv2d(int(dim * 2), out_channels, 3, stride=1, padding=1, bias=bias)
        self.last_aux = {}

    def _extract_degradation_vector(self, inp_img):
        """Extract degradation-only feature vector using DWT + VGG16.

        Pipeline: Image -> DWT (discard LL) -> VGG16 -> GAP -> L2-norm -> 256-d
        """
        with torch.no_grad():
            # Fix #1: DWT to separate degradation from content
            hf_img = extract_degradation_features(inp_img.clamp(0, 1))
            # Normalize for VGG16 (pretrained on ImageNet)
            hf_norm = (hf_img - self.vgg_mean) / self.vgg_std
            deg_features = self.frozen_extractor(hf_norm).mean(dim=(-2, -1))
        return deg_features

    @staticmethod
    def _match_skip(x, skip):
        if x.shape[-2:] == skip.shape[-2:]:
            return x
        return F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, inp_img, return_aux=False):
        b, _, h, w = inp_img.shape

        # --- Statistical Routing (Fixes #1 + #3) ---
        soft_mask_img = None
        soft_mask_feat = None
        deg_features = self._extract_degradation_vector(inp_img)

        if not self.training:
            task_probs = self.stat_memory.compute_task_scores(deg_features)
            if task_probs is not None:
                # Fix #2: Generate soft masks instead of logit offsets
                soft_mask_img = self.image_fuser.compute_soft_mask(task_probs)
                soft_mask_feat = self.feature_fuser.compute_soft_mask(task_probs)

        # --- P-Fuser with Soft Mask Gating (Fix #2) ---
        query_feature = self.patch_embed_query(inp_img).mean(dim=(-2, -1))
        image_prompt, image_aux = self.image_fuser(
            query_feature, update_frequency=True, soft_mask=soft_mask_img)
        image_prompt = F.interpolate(image_prompt, size=(h, w), mode="bilinear", align_corners=False)
        prompted_img = self.image_prompt_adapter(inp_img, image_prompt)

        out_enc_level1 = self.encoder_level1(self.patch_embed(prompted_img))
        out_enc_level2 = self.encoder_level2(self.down1_2(out_enc_level1))
        out_enc_level3 = self.encoder_level3(self.down2_3(out_enc_level2))
        latent = self.latent(self.down3_4(out_enc_level3))

        feature_query = latent.mean(dim=(-2, -1))
        feature_prompt, feature_aux = self.feature_fuser(
            feature_query, update_frequency=True, soft_mask=soft_mask_feat)
        feature_prompt = feature_prompt.expand(b, -1, latent.shape[-2], latent.shape[-1])
        latent = self.feature_prompt_adapter(latent, feature_prompt)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = self._match_skip(inp_dec_level3, out_enc_level3)
        inp_dec_level3 = self.reduce_chan_level3(torch.cat([inp_dec_level3, out_enc_level3], dim=1))
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = self._match_skip(inp_dec_level2, out_enc_level2)
        inp_dec_level2 = self.reduce_chan_level2(torch.cat([inp_dec_level2, out_enc_level2], dim=1))
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = self._match_skip(inp_dec_level1, out_enc_level1)
        out_dec_level1 = self.decoder_level1(torch.cat([inp_dec_level1, out_enc_level1], dim=1))
        out_dec_level1 = self.refinement(out_dec_level1)

        restored = self.output(out_dec_level1) + inp_img
        self.last_aux = {
            "image_distance": image_aux["distance_loss"],
            "feature_distance": feature_aux["distance_loss"],
            "image_top_indices": image_aux["top_indices"],
            "feature_top_indices": feature_aux["top_indices"],
            "image_logits": image_aux["logits"],
            "feature_logits": feature_aux["logits"],
            "deg_features": deg_features,
        }

        if return_aux:
            return restored, self.last_aux
        return restored

    def prompt_regularization_loss(self):
        image_keys = F.normalize(self.image_fuser.keys, dim=1)
        feature_keys = F.normalize(self.feature_fuser.keys, dim=1)
        image_eye = torch.eye(self.num_prompts, device=image_keys.device)
        feature_eye = torch.eye(self.num_prompts, device=feature_keys.device)
        return (
            (image_keys @ image_keys.t() - image_eye).pow(2).mean()
            + (feature_keys @ feature_keys.t() - feature_eye).pow(2).mean()
        )

    @torch.no_grad()
    def grad_tune_prompts(self):
        self.image_fuser.grad_tune(self.grad_tuner_components)
        self.feature_fuser.grad_tune(self.grad_tuner_components)

    @torch.no_grad()
    def set_active_prompt_range(self, start=None, end=None):
        self.image_fuser.set_active_range(start, end)
        self.feature_fuser.set_active_range(start, end)

    @torch.no_grad()
    def clear_active_prompt_range(self):
        self.image_fuser.clear_active_range()
        self.feature_fuser.clear_active_range()

    def zero_protected_prompt_grads(self):
        self.image_fuser.zero_protected_grads()
        self.feature_fuser.zero_protected_grads()

    @torch.no_grad()
    def backup_protected_prompts(self):
        self.image_fuser.backup_protected_prompts()
        self.feature_fuser.backup_protected_prompts()

    @torch.no_grad()
    def restore_protected_prompts(self):
        self.image_fuser.restore_protected_prompts()
        self.feature_fuser.restore_protected_prompts()

    @torch.no_grad()
    def update_task_statistics(self, task_id, dataloader, device):
        """Compute and store GMM anchors + prompt masks after training a task."""
        training_state = self.training
        self.eval()
        all_deg_features = []

        for batch in dataloader:
            if len(batch) == 3:
                name, degraded, clean = batch
            elif len(batch) == 2:
                degraded, clean = batch
            else:
                degraded = batch[0]
            degraded = degraded.to(device)
            deg_features = self._extract_degradation_vector(degraded)
            all_deg_features.append(deg_features)

        all_deg_features = torch.cat(all_deg_features, dim=0)

        # Fix #3: Fit GMM to degradation features
        self.stat_memory.update_statistics(task_id, all_deg_features)

        # Fix #2: Store prompt usage mask for this task
        self.image_fuser.update_task_mask(task_id)
        self.feature_fuser.update_task_mask(task_id)

        # Reset frequency counters for next task
        self.image_fuser.frequency.fill_(1.0)
        self.feature_fuser.frequency.fill_(1.0)

        self.train(training_state)
