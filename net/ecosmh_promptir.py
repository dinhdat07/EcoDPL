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
    """Extract a high-frequency degradation proxy by discarding the LL band.

    Uses the LH, HL, HH sub-bands which capture directional high-frequency
    information (horizontal, vertical, diagonal rain streaks).

    Args:
        x: Rainy image tensor [B, 3, H, W], values in [0, 1].

    Returns:
        Tensor of shape [B, 3, H//2, W//2]. High-frequency scene texture is
        still present, so this representation is not content-free.
    """
    _, lh, hl, hh = haar_dwt2d(x)
    # Combine high-freq sub-bands: average across input channels, stack as 3ch
    # Each sub-band captures a different direction of rain streaks
    return torch.cat([
        lh.abs().mean(dim=1, keepdim=True),
        hl.abs().mean(dim=1, keepdim=True),
        hh.abs().mean(dim=1, keepdim=True),
    ], dim=1)


# ---------------------------------------------------------------------------
# GMM statistical memory (K components + consistent L2 normalization)
# ---------------------------------------------------------------------------


def _regularized_covariance(scatter, dof, shrinkage, ridge):
    dim = scatter.shape[0]
    covariance = scatter / max(float(dof), 1.0)
    scale = covariance.diagonal().mean().clamp_min(ridge)
    eye = torch.eye(dim, device=scatter.device, dtype=scatter.dtype)
    covariance = (1.0 - shrinkage) * covariance + shrinkage * scale * eye
    covariance = covariance + ridge * eye
    chol, info = torch.linalg.cholesky_ex(covariance)
    if int(info.max().item()) == 0:
        return torch.cholesky_inverse(chol), 2.0 * torch.log(chol.diagonal()).sum()
    sign, log_det = torch.linalg.slogdet(covariance)
    if sign <= 0:
        raise RuntimeError("Regularized covariance is not positive definite")
    return torch.linalg.pinv(covariance, hermitian=True), log_det


class GMMStatisticalMemory(nn.Module):
    """Hard-assignment GMM memory for task-agnostic inference routing.

    Components within a task share a shrinkage covariance. Feature
    normalization is identical during fitting and inference, and Gaussian
    log-determinants keep likelihoods from different tasks comparable.

    References:
        - CoGaMiD (NeurIPS/OpenReview): GMM for continual segmentation
        - Mahalanobis++ (OpenReview 2024): L2-norm before Mahalanobis
    """

    def __init__(
        self,
        feature_dim,
        max_tasks=10,
        num_components=3,
        covariance_shrinkage=0.1,
        covariance_ridge=1e-4,
        temperature=1.0,
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.feature_dim = feature_dim
        self.max_tasks = max_tasks
        self.K = num_components
        self.covariance_shrinkage = covariance_shrinkage
        self.covariance_ridge = covariance_ridge
        self.temperature = temperature

        # Per-task GMM parameters: K components each
        self.register_buffer(
            "means", torch.zeros(max_tasks, num_components, feature_dim))
        self.register_buffer(
            "inv_covs", torch.zeros(max_tasks, num_components, feature_dim, feature_dim))
        self.register_buffer(
            "mix_weights", torch.zeros(max_tasks, num_components))
        self.register_buffer(
            "log_dets", torch.zeros(max_tasks, num_components))
        self.register_buffer(
            "task_valid", torch.zeros(max_tasks, dtype=torch.bool))
        self.register_buffer(
            "task_count", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def update_statistics(self, task_id, features):
        """Fit a hard-assignment GMM on consistently normalized features."""
        if not 0 <= int(task_id) < self.max_tasks:
            raise IndexError(f"task_id {task_id} is outside [0, {self.max_tasks})")
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected features [N, {self.feature_dim}], got {tuple(features.shape)}"
            )
        if features.shape[0] == 0:
            raise ValueError("Cannot fit statistical memory with no features")

        features = F.normalize(features.float(), dim=1)
        N, D = features.shape
        component_count = min(self.K, N)

        # Deterministic farthest-point initialization.
        first = torch.cdist(features, features.mean(dim=0, keepdim=True)).argmax()
        centroids = [features[first].clone()]
        for _ in range(1, component_count):
            dists = torch.cdist(features, torch.stack(centroids))
            centroids.append(features[dists.min(dim=1).values.argmax()].clone())
        centroids = torch.stack(centroids)
        assignments = torch.full((N,), -1, dtype=torch.long, device=features.device)

        for _ in range(25):
            dists = torch.cdist(features, centroids)
            new_assignments = dists.argmin(dim=1)
            if torch.equal(assignments, new_assignments):
                break
            assignments = new_assignments
            for k in range(component_count):
                mask = assignments == k
                if mask.any():
                    centroids[k] = features[mask].mean(dim=0)

        # Pool within-component scatter and regularize it. This remains positive
        # definite even when the sample count is below the feature dimension.
        counts = torch.zeros(self.K, device=features.device)
        scatter = torch.zeros(D, D, device=features.device)
        self.mix_weights[task_id].zero_()
        for k in range(component_count):
            mask = assignments == k
            count = int(mask.sum().item())
            if count > 0:
                component_features = features[mask]
                mean = component_features.mean(dim=0)
                centered = component_features - mean
                scatter.add_(centered.t() @ centered)
                self.means[task_id, k].copy_(mean)
                counts[k] = count

        valid_components = int((counts > 0).sum().item())
        precision, log_det = _regularized_covariance(
            scatter,
            N - valid_components,
            self.covariance_shrinkage,
            self.covariance_ridge,
        )
        for k in range(self.K):
            self.inv_covs[task_id, k].copy_(precision)
            self.log_dets[task_id, k].copy_(log_det)
        self.mix_weights[task_id].copy_(counts / counts.sum().clamp_min(1.0))
        self.task_valid[task_id] = True

        if task_id >= self.task_count.item():
            self.task_count.fill_(task_id + 1)

    def compute_task_logits(self, features):
        """Compute proper Gaussian-mixture log likelihoods for each task."""
        num_tasks = self.task_count.item()
        if num_tasks == 0:
            return None
        features = F.normalize(features.float(), dim=1)

        task_scores = []
        for t in range(num_tasks):
            if not self.task_valid[t]:
                task_scores.append(
                    torch.full(
                        (features.shape[0],),
                        -torch.inf,
                        device=features.device,
                        dtype=features.dtype,
                    )
                )
                continue
            component_scores = []
            for k in range(self.K):
                w = self.mix_weights[t, k]
                if w <= 0:
                    component_scores.append(
                        torch.full(
                            (features.shape[0],),
                            -torch.inf,
                            device=features.device,
                            dtype=features.dtype,
                        )
                    )
                    continue
                mean = self.means[t, k]
                inv_cov = self.inv_covs[t, k]
                diff = features - mean
                mahal = ((diff @ inv_cov) * diff).sum(dim=1)
                component_scores.append(
                    torch.log(w) - 0.5 * (mahal + self.log_dets[t, k])
                )
            task_scores.append(
                torch.logsumexp(torch.stack(component_scores, dim=1), dim=1)
            )
        return torch.stack(task_scores, dim=1)

    def compute_task_scores(self, features, temperature=None):
        logits = self.compute_task_logits(features)
        if logits is None:
            return None
        temperature = self.temperature if temperature is None else temperature
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        return F.softmax(logits / temperature, dim=1)

class SFTAdapter(nn.Module):
    """Per-image FiLM adapter driven by a global prompt vector."""
    def __init__(self, in_channels, prompt_channels=None):
        super().__init__()
        if prompt_channels is None:
            prompt_channels = in_channels
        self.conv_gamma = nn.Sequential(
            nn.Conv2d(prompt_channels, in_channels, 1, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels, in_channels, 1, padding=0, bias=False)
        )
        self.conv_beta = nn.Sequential(
            nn.Conv2d(prompt_channels, in_channels, 1, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels, in_channels, 1, padding=0, bias=False)
        )
        
        # Fix #6: Adapter Initialization Trap. Must zero-initialize the final layers
        # so that the adapter acts as an identity mapping at the start of training.
        nn.init.zeros_(self.conv_gamma[2].weight)
        nn.init.zeros_(self.conv_beta[2].weight)

        self.register_buffer(
            "nsp_input_cov", torch.zeros(prompt_channels, prompt_channels)
        )
        self.register_buffer(
            "nsp_gamma_hidden_cov", torch.zeros(in_channels, in_channels)
        )
        self.register_buffer(
            "nsp_beta_hidden_cov", torch.zeros(in_channels, in_channels)
        )
        self.register_buffer("nsp_input_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer(
            "nsp_gamma_hidden_count", torch.tensor(0, dtype=torch.long)
        )
        self.register_buffer(
            "nsp_beta_hidden_count", torch.tensor(0, dtype=torch.long)
        )
        self._collect_nsp = False

    @torch.no_grad()
    def set_nsp_collection(self, enabled):
        self._collect_nsp = bool(enabled)

    @torch.no_grad()
    def _accumulate_covariance(self, values, covariance, count):
        # Prompts are spatially constant in this architecture. Averaging avoids
        # counting the same broadcast vector once per pixel.
        vectors = values.detach().float().mean(dim=(-2, -1))
        covariance.add_(vectors.t() @ vectors)
        count.add_(vectors.shape[0])

    def forward(self, x, prompt):
        gamma_hidden = self.conv_gamma[1](self.conv_gamma[0](prompt))
        beta_hidden = self.conv_beta[1](self.conv_beta[0](prompt))
        if self._collect_nsp:
            self._accumulate_covariance(
                prompt, self.nsp_input_cov, self.nsp_input_count
            )
            self._accumulate_covariance(
                gamma_hidden,
                self.nsp_gamma_hidden_cov,
                self.nsp_gamma_hidden_count,
            )
            self._accumulate_covariance(
                beta_hidden,
                self.nsp_beta_hidden_cov,
                self.nsp_beta_hidden_count,
            )
        gamma = self.conv_gamma[2](gamma_hidden)
        beta = self.conv_beta[2](beta_hidden)
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

    def __init__(
        self,
        num_prompts,
        query_dim,
        value_shape,
        temperature=1.0,
        max_tasks=10,
    ):
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
            "task_prompt_mask",
            torch.zeros(max_tasks, num_prompts),
            persistent=True,
        )
            
        # Hard-freezing buffer: stores which prompts belong exclusively to old tasks
        self.register_buffer(
            "protected_mask", torch.zeros(num_prompts, dtype=torch.bool), persistent=True)
            
        self.backup_keys = None
        self.backup_values = None
        self.backup_attention = None

    @torch.no_grad()
    def backup_protected_prompts(self):
        """Backup all prompt slices that must remain immutable this task."""
        self.backup_keys = self.keys.data.clone()
        self.backup_values = self.values.data.clone()
        self.backup_attention = self.attention.data.clone()

    def frozen_mask(self):
        return self.protected_mask | ~self.active

    @torch.no_grad()
    def zero_protected_grads(self):
        """Zero gradients for old prompts and future, inactive prompt slots."""
        mask = self.frozen_mask()
        if mask.any():
            if self.keys.grad is not None:
                self.keys.grad[mask] = 0.0
            if self.values.grad is not None:
                self.values.grad[mask] = 0.0
            if self.attention.grad is not None:
                self.attention.grad[mask] = 0.0

    @torch.no_grad()
    def restore_protected_prompts(self):
        """Restore frozen slices after an optimizer step, including weight decay."""
        mask = self.frozen_mask()
        if self.backup_keys is not None and mask.any():
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
        if self.training and self.active is not None:
            logits = logits.masked_fill(~self.active.to(logits.device)[None, :], -1e4)

        raw_logits = logits.clone()

        # Treat the routing mask as a multiplicative prior on attention weights:
        # softmax(logits + log(mask)) is proportional to exp(logits) * mask.
        if soft_mask is not None:
            mask_penalty = torch.log(soft_mask.clamp(min=1e-8))
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
        """Assign the current private prompt block to a task and freeze it."""
        if not 0 <= int(task_id) < self.task_prompt_mask.shape[0]:
            raise IndexError(f"task_id {task_id} exceeds prompt memory capacity")
        self.task_prompt_mask[task_id].zero_()
        self.task_prompt_mask[task_id, self.active] = 1.0
        self.protected_mask |= self.active
        self.frequency.fill_(1.0)

    def task_mask(self, task_id, batch_size):
        if not 0 <= int(task_id) < self.task_prompt_mask.shape[0]:
            raise IndexError(f"task_id {task_id} exceeds prompt memory capacity")
        mask = self.task_prompt_mask[task_id]
        if not bool(mask.any().item()):
            mask = self.active.float()
        return mask.to(self.values.device).unsqueeze(0).expand(batch_size, -1)

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
    """EcoSMH continual deraining model.

    Training is boundary-aware and rehearsal-free. Inference can be
    task-agnostic through the statistical router, or use an oracle ``task_id``
    for diagnostics. Null-space protection is applied only to the FiLM adapter
    weights; private prompt blocks are protected exactly by the trainer.
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
        max_tasks=10,
        router_temperature=1.0,
        covariance_shrinkage=0.1,
    ):
        super().__init__()
        if num_blocks is None:
            num_blocks = [4, 6, 6, 8]
        if heads is None:
            heads = [1, 2, 4, 8]

        self.num_prompts = num_prompts
        self.max_tasks = max_tasks
        self.grad_tuner_components = grad_tuner_components

        self.stat_memory = GMMStatisticalMemory(
            feature_dim=256,
            max_tasks=max_tasks,
            num_components=gmm_components,
            covariance_shrinkage=covariance_shrinkage,
            temperature=router_temperature,
        )

        # The router sees high-frequency magnitudes, not a content-free signal.
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
            value_shape=(dim, 1, 1),
            max_tasks=max_tasks,
        )
        self.image_prompt_adapter = SFTAdapter(inp_channels, prompt_channels=dim)

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=layer_norm_type)
            for _ in range(num_blocks[0])
        ])
        self.adapter_enc_level1 = SFTAdapter(in_channels=dim, prompt_channels=dim)

        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=layer_norm_type)
            for _ in range(num_blocks[1])
        ])
        self.adapter_enc_level2 = SFTAdapter(in_channels=int(dim * 2), prompt_channels=dim)

        self.down2_3 = Downsample(int(dim * 2))
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 4), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=layer_norm_type)
            for _ in range(num_blocks[2])
        ])
        self.adapter_enc_level3 = SFTAdapter(in_channels=int(dim * 4), prompt_channels=dim)

        self.down3_4 = Downsample(int(dim * 4))
        self.latent_dim = int(dim * 8)
        self.latent = nn.Sequential(*[
            TransformerBlock(dim=self.latent_dim, num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=layer_norm_type)
            for _ in range(num_blocks[3])
        ])

        self.feature_fuser = PromptFuser(
            num_prompts=num_prompts,
            query_dim=self.latent_dim,
            value_shape=(self.latent_dim, 1, 1),
            max_tasks=max_tasks,
        )
        self.feature_prompt_adapter = SFTAdapter(self.latent_dim)

        self.up4_3 = Upsample(self.latent_dim)
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 8), int(dim * 4), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 4), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=layer_norm_type)
            for _ in range(num_blocks[2])
        ])
        self.adapter_dec_level3 = SFTAdapter(in_channels=int(dim * 4), prompt_channels=self.latent_dim)

        self.up3_2 = Upsample(int(dim * 4))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 4), int(dim * 2), 1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=layer_norm_type)
            for _ in range(num_blocks[1])
        ])
        self.adapter_dec_level2 = SFTAdapter(in_channels=int(dim * 2), prompt_channels=self.latent_dim)

        self.up2_1 = Upsample(int(dim * 2))
        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=layer_norm_type)
            for _ in range(num_blocks[0])
        ])
        self.adapter_dec_level1 = SFTAdapter(in_channels=int(dim * 2), prompt_channels=self.latent_dim)

        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=layer_norm_type)
            for _ in range(num_refinement_blocks)
        ])
        self.output = nn.Conv2d(int(dim * 2), out_channels, 3, stride=1, padding=1, bias=bias)
        self.last_aux = {}

    def _extract_degradation_vector(self, inp_img):
        """Extract a high-frequency VGG descriptor used only by the router.

        Pipeline: Image -> DWT (discard LL) -> VGG16 -> GAP -> L2-norm -> 256-d
        """
        self.frozen_extractor.eval()
        with torch.no_grad():
            # Fix #1: DWT to separate degradation from content
            hf_img = extract_degradation_features(inp_img.clamp(0, 1))
            # Normalize for VGG16 (pretrained on ImageNet)
            hf_norm = (hf_img - self.vgg_mean) / self.vgg_std
            deg_features = self.frozen_extractor(hf_norm).mean(dim=(-2, -1))
        return deg_features

    @torch.no_grad()
    def route_task_probs(self, inp_img, max_side=256):
        """Route once per image; callers can reuse the result for all tiles."""
        if self.stat_memory.task_count.item() == 0:
            return None
        height, width = inp_img.shape[-2:]
        if max(height, width) > max_side:
            scale = max_side / max(height, width)
            inp_img = F.interpolate(
                inp_img,
                size=(max(8, round(height * scale)), max(8, round(width * scale))),
                mode="bilinear",
                align_corners=False,
            )
        return self.stat_memory.compute_task_scores(
            self._extract_degradation_vector(inp_img)
        )

    @staticmethod
    def _match_skip(x, skip):
        if x.shape[-2:] == skip.shape[-2:]:
            return x
        return F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

    def forward(
        self,
        inp_img,
        return_aux=False,
        task_id=None,
        routing_probs=None,
    ):
        b, _, h, w = inp_img.shape
        if task_id is not None and routing_probs is not None:
            raise ValueError("Specify task_id or routing_probs, not both")

        # Route only at inference. Training uses the current private block.
        soft_mask_img = None
        soft_mask_feat = None
        deg_features = None
        if task_id is not None:
            soft_mask_img = self.image_fuser.task_mask(task_id, b)
            soft_mask_feat = self.feature_fuser.task_mask(task_id, b)
        else:
            if routing_probs is None and not self.training:
                routing_probs = self.route_task_probs(inp_img)
            if routing_probs is not None:
                soft_mask_img = self.image_fuser.compute_soft_mask(routing_probs)
                soft_mask_feat = self.feature_fuser.compute_soft_mask(routing_probs)

        query_feature = self.patch_embed_query(inp_img).mean(dim=(-2, -1))
        image_prompt, image_aux = self.image_fuser(
            query_feature, update_frequency=True, soft_mask=soft_mask_img)
        image_prompt = image_prompt.expand(-1, -1, h, w)
        prompted_img = self.image_prompt_adapter(inp_img, image_prompt)

        out_enc_level1 = self.encoder_level1(self.patch_embed(prompted_img))
        
        prompt_level1 = F.interpolate(image_prompt, size=out_enc_level1.shape[-2:], mode="bilinear", align_corners=False)
        out_enc_level1 = self.adapter_enc_level1(out_enc_level1, prompt_level1)
        
        out_enc_level2 = self.encoder_level2(self.down1_2(out_enc_level1))
        
        prompt_level2 = F.interpolate(image_prompt, size=out_enc_level2.shape[-2:], mode="bilinear", align_corners=False)
        out_enc_level2 = self.adapter_enc_level2(out_enc_level2, prompt_level2)
        
        out_enc_level3 = self.encoder_level3(self.down2_3(out_enc_level2))
        
        prompt_level3 = F.interpolate(image_prompt, size=out_enc_level3.shape[-2:], mode="bilinear", align_corners=False)
        out_enc_level3 = self.adapter_enc_level3(out_enc_level3, prompt_level3)
        
        latent = self.latent(self.down3_4(out_enc_level3))

        feature_query = latent.mean(dim=(-2, -1))

        feature_prompt, feature_aux = self.feature_fuser(
            feature_query, update_frequency=True, soft_mask=soft_mask_feat)
        feature_prompt = feature_prompt.expand(b, -1, latent.shape[-2], latent.shape[-1])
        prompted_latent = self.feature_prompt_adapter(latent, feature_prompt)
        
        # Decoder passes
        out_dec_level3 = self.up4_3(prompted_latent)
        out_dec_level3 = torch.cat([out_dec_level3, out_enc_level3], 1)
        out_dec_level3 = self.reduce_chan_level3(out_dec_level3)
        out_dec_level3 = self.decoder_level3(out_dec_level3)
        prompt_dec3 = F.interpolate(feature_prompt, size=out_dec_level3.shape[-2:], mode="bilinear", align_corners=False)
        out_dec_level3 = self.adapter_dec_level3(out_dec_level3, prompt_dec3)

        out_dec_level2 = self.up3_2(out_dec_level3)
        out_dec_level2 = torch.cat([out_dec_level2, out_enc_level2], 1)
        out_dec_level2 = self.reduce_chan_level2(out_dec_level2)
        out_dec_level2 = self.decoder_level2(out_dec_level2)
        prompt_dec2 = F.interpolate(feature_prompt, size=out_dec_level2.shape[-2:], mode="bilinear", align_corners=False)
        out_dec_level2 = self.adapter_dec_level2(out_dec_level2, prompt_dec2)

        out_dec_level1 = self.up2_1(out_dec_level2)
        out_dec_level1 = torch.cat([out_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(out_dec_level1)
        prompt_dec1 = F.interpolate(feature_prompt, size=out_dec_level1.shape[-2:], mode="bilinear", align_corners=False)
        out_dec_level1 = self.adapter_dec_level1(out_dec_level1, prompt_dec1)

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
            "latent_query": feature_query,
        }

        if return_aux:
            return restored, self.last_aux
        return restored

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
    def set_nsp_collection(self, enabled):
        for module in self.modules():
            if isinstance(module, SFTAdapter):
                module.set_nsp_collection(enabled)

    @torch.no_grad()
    def update_task_statistics(self, task_id, dataloader, device):
        """Consolidate router statistics, prompt ownership and NSP covariances."""
        training_state = self.training
        all_deg_features = []
        # Train mode intentionally bypasses statistical routing and selects only
        # the active private prompt block. No gradients are recorded.
        self.train(True)
        self.set_nsp_collection(True)
        try:
            for batch in dataloader:
                degraded = batch[1] if len(batch) == 3 else batch[0]
                degraded = degraded.to(device, non_blocking=True)
                pad_h = (8 - degraded.size(2) % 8) % 8
                pad_w = (8 - degraded.size(3) % 8) % 8
                if pad_h > 0 or pad_w > 0:
                    degraded = F.pad(
                        degraded, (0, pad_w, 0, pad_h), mode="reflect"
                    )

                self.forward(degraded)
                all_deg_features.append(self._extract_degradation_vector(degraded))
        finally:
            self.set_nsp_collection(False)
            self.train(training_state)

        if not all_deg_features:
            raise ValueError("Cannot consolidate an empty task loader")
        self.stat_memory.update_statistics(task_id, torch.cat(all_deg_features, dim=0))
        self.image_fuser.update_task_mask(task_id)
        self.feature_fuser.update_task_mask(task_id)
