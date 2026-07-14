import torch


@torch.no_grad()
def projector_from_covariance(covariance, count, threshold=0.03, strength=1.0):
    """Build a right projector onto low-energy directions of old inputs.

    A strength of one is a hard approximate null-space constraint. Smaller
    values interpolate with the identity and intentionally trade stability for
    plasticity. If old inputs span the full space, the hard projector is zero.
    """
    if not 0.0 <= threshold < 1.0:
        raise ValueError("threshold must be in [0, 1)")
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be in [0, 1]")
    if int(count) <= 0:
        raise ValueError("Cannot build a projector before collecting activations")

    covariance = covariance.float() / float(count)
    covariance = 0.5 * (covariance + covariance.t())
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    largest = eigenvalues[-1].clamp_min(0.0)
    identity = torch.eye(
        covariance.shape[0], device=covariance.device, dtype=covariance.dtype
    )
    if largest <= torch.finfo(covariance.dtype).eps:
        null_projector = identity
        nullity = covariance.shape[0]
    else:
        low_energy = eigenvalues <= largest * threshold
        basis = eigenvectors[:, low_energy]
        null_projector = basis @ basis.t()
        nullity = int(low_energy.sum().item())
    projector = strength * null_projector + (1.0 - strength) * identity
    return projector, {
        "dimension": covariance.shape[0],
        "nullity": nullity,
        "largest_eigenvalue": float(largest.item()),
    }


@torch.no_grad()
def compute_null_space_projectors(
    model, patch_size=None, threshold=0.03, strength=1.0
):
    """Build adapter projectors from cumulative observed activation covariance."""
    del patch_size  # Compatibility with older training commands.
    projectors = {}
    for name, adapter in model.named_modules():
        required = (
            "nsp_input_cov",
            "nsp_gamma_hidden_cov",
            "nsp_beta_hidden_cov",
        )
        if not all(hasattr(adapter, attr) for attr in required):
            continue
        if int(adapter.nsp_input_count.item()) == 0:
            continue

        input_projector, input_stats = projector_from_covariance(
            adapter.nsp_input_cov,
            adapter.nsp_input_count,
            threshold=threshold,
            strength=strength,
        )
        gamma_projector, gamma_stats = projector_from_covariance(
            adapter.nsp_gamma_hidden_cov,
            adapter.nsp_gamma_hidden_count,
            threshold=threshold,
            strength=strength,
        )
        beta_projector, beta_stats = projector_from_covariance(
            adapter.nsp_beta_hidden_cov,
            adapter.nsp_beta_hidden_count,
            threshold=threshold,
            strength=strength,
        )
        projectors[name] = {
            "gamma_0": input_projector,
            "beta_0": input_projector,
            "gamma_2": gamma_projector,
            "beta_2": beta_projector,
            "stats": {
                "input": input_stats,
                "gamma_hidden": gamma_stats,
                "beta_hidden": beta_stats,
            },
        }
    return projectors


@torch.no_grad()
def snapshot_projected_weights(model, projectors):
    """Clone only weights constrained after the optimizer step."""
    prefixes = tuple(f"{name}.conv_" for name in projectors)
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if prefixes and name.startswith(prefixes) and name.endswith(".weight")
    }


@torch.no_grad()
def apply_null_space_projection(model, projectors, old_weights):
    """Project the actual AdamW candidate update on old activation null spaces.

    This preserves each constrained linear map on sampled old activations up to
    the chosen eigenvalue threshold. It is not a whole-network zero-forgetting
    guarantee.
    """
    for name, adapter in model.named_modules():
        if name not in projectors:
            continue
        module_projectors = projectors[name]
        for branch_name in ("gamma", "beta"):
            branch = getattr(adapter, f"conv_{branch_name}")
            for layer_index in (0, 2):
                full_name = f"{name}.conv_{branch_name}.{layer_index}.weight"
                if full_name not in old_weights:
                    continue
                weight = branch[layer_index].weight
                old_weight = old_weights[full_name]
                delta = weight.data - old_weight
                shape = delta.shape
                projector = module_projectors[f"{branch_name}_{layer_index}"]
                projected = delta.reshape(shape[0], -1).float() @ projector
                weight.data.copy_(
                    old_weight + projected.to(weight.dtype).reshape(shape)
                )
