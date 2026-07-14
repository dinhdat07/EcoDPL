import torch
import torch.nn.functional as F

@torch.no_grad()
def compute_null_space_projectors(model, patch_size, threshold=0.03):
    """
    Compute Null Space projectors for all SFTAdapters to prevent Catastrophic Forgetting.
    """
    P_null_dict = {}
    
    # Get protected prompts
    img_fuser = model.image_fuser
    feat_fuser = model.feature_fuser
    
    if not img_fuser.protected_mask.any():
        return P_null_dict # No protected prompts yet
        
    img_protected_vals = img_fuser.values[img_fuser.protected_mask] # [K, 3, 32, 32]
    feat_protected_vals = feat_fuser.values[feat_fuser.protected_mask] # [K, latent_dim, 1, 1]
    
    # Helper to compute P_null for a Conv2d layer
    def compute_P_null_for_tensor(tensor, conv):
        # Extract patches
        # tensor is [K, C, H, W]
        # unfold to [K, C * 9, L]
        unfolded = F.unfold(tensor, kernel_size=conv.kernel_size, padding=conv.padding, stride=conv.stride)
        # Reshape to [K * L, C * 9]
        X = unfolded.transpose(1, 2).reshape(-1, unfolded.shape[1])
        
        # Compute uncentered covariance
        # To avoid memory issues with large K*L, we compute X.T @ X
        cov = (X.T @ X) / X.shape[0]
        
        # SVD
        U, S, V = torch.linalg.svd(cov, full_matrices=True)
        
        # Find Null Space
        zero_idx = S <= S[0] * threshold
        if not zero_idx.any():
            # If no null space, return identity
            return torch.eye(X.shape[1], device=X.device)
            
        U_null = U[:, zero_idx]
        P_null = U_null @ U_null.T
        return P_null

    def get_P_nulls(adapter, prompts, spatial_size):
        # prompts: [K, C, H_p, W_p]
        # Interpolate to the spatial size the adapter sees during training
        if prompts.shape[-2:] != (spatial_size, spatial_size):
            prompts = F.interpolate(prompts, size=(spatial_size, spatial_size), mode="bilinear", align_corners=False)
            
        # P_null for first layer
        conv0 = adapter.conv_gamma[0]
        P_null_0 = compute_P_null_for_tensor(prompts, conv0)
        
        # Compute activation for second layer
        with torch.no_grad():
            H_gamma = adapter.conv_gamma[0](prompts)
            H_gamma = adapter.conv_gamma[1](H_gamma) # LeakyReLU
            
            H_beta = adapter.conv_beta[0](prompts)
            H_beta = adapter.conv_beta[1](H_beta) # LeakyReLU
            
        conv2_gamma = adapter.conv_gamma[2]
        P_null_gamma_2 = compute_P_null_for_tensor(H_gamma, conv2_gamma)
        
        conv2_beta = adapter.conv_beta[2]
        P_null_beta_2 = compute_P_null_for_tensor(H_beta, conv2_beta)
        
        return {
            "gamma_0": P_null_0,
            "beta_0": P_null_0, # Beta and Gamma 0 have the same input
            "gamma_2": P_null_gamma_2,
            "beta_2": P_null_beta_2,
        }

    # Map adapters to their spatial sizes (assuming input patch_size)
    sizes = {
        "image_prompt_adapter": patch_size,
        "adapter_enc_level1": patch_size,
        "adapter_enc_level2": patch_size // 2,
        "adapter_enc_level3": patch_size // 4,
        "feature_prompt_adapter": patch_size // 8,
        "adapter_dec_level3": patch_size // 4,
        "adapter_dec_level2": patch_size // 2,
        "adapter_dec_level1": patch_size,
    }
    
    for name, adapter in model.named_modules():
        if name in sizes:
            # feature_prompt_adapter and decoder adapters use feature_prompt
            if "dec" in name or "feature" in name:
                prompts = feat_protected_vals
            else:
                prompts = img_protected_vals
                
            P_null_dict[name] = get_P_nulls(adapter, prompts, sizes[name])
            
    return P_null_dict

def apply_null_space_projection(model, P_null_dict, old_weights):
    """
    Project the actual weight update (Delta W) into the Null Space.
    This guarantees mathematically that AdamW's element-wise scaling 
    does not violate the Null Space constraints.
    """
    for name, adapter in model.named_modules():
        if name in P_null_dict:
            P_nulls = P_null_dict[name]
            for branch_name in ["gamma", "beta"]:
                branch = getattr(adapter, f"conv_{branch_name}")
                
                # Layer 0
                conv0 = branch[0]
                full_name_0 = f"{name}.conv_{branch_name}.0.weight"
                if full_name_0 in old_weights:
                    old_w = old_weights[full_name_0]
                    delta = conv0.weight.data - old_w
                    shape = delta.shape
                    delta_flat = delta.view(shape[0], -1)
                    delta_proj = delta_flat @ P_nulls[f"{branch_name}_0"]
                    conv0.weight.data.copy_(old_w + delta_proj.view(shape))
                    
                # Layer 2
                conv2 = branch[2]
                full_name_2 = f"{name}.conv_{branch_name}.2.weight"
                if full_name_2 in old_weights:
                    old_w = old_weights[full_name_2]
                    delta = conv2.weight.data - old_w
                    shape = delta.shape
                    delta_flat = delta.view(shape[0], -1)
                    delta_proj = delta_flat @ P_nulls[f"{branch_name}_2"]
                    conv2.weight.data.copy_(old_w + delta_proj.view(shape))
