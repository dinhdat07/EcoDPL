import argparse
import csv
import json
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.models import VGG16_Weights, vgg16
from tqdm import tqdm

from net.ecosmh_promptir import EcoDPLPromptIR
from utils.derain_release import (
    H5DerainDataset,
    ImagePairDataset,
    calculate_psnr,
    calculate_ssim,
    pad_to_multiple,
    set_seed,
    tensor_to_rgb,
    tiled_forward,
)


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        weights = VGG16_Weights.IMAGENET1K_V1
        self.features = vgg16(weights=weights).features[:16].eval()
        for param in self.features.parameters():
            param.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        pred = (pred.clamp(0, 1) - self.mean) / self.std
        target = (target.clamp(0, 1) - self.mean) / self.std
        return F.mse_loss(self.features(pred), self.features(target))


def _split_indices(length, validation_fraction, seed):
    if length < 2:
        raise ValueError("At least two training samples are required for validation")
    validation_size = min(
        length - 1, max(1, int(round(length * validation_fraction)))
    )
    order = torch.randperm(length, generator=torch.Generator().manual_seed(seed)).tolist()
    return order[validation_size:], order[:validation_size]


def build_loaders(args, task, task_index):
    train_set = H5DerainDataset(
        args.data_root, task, patch_size=args.patch_size, augment=True
    )
    deterministic_set = H5DerainDataset(
        args.data_root, task, patch_size=args.patch_size, augment=False
    )
    split_seed = args.seed + task_index * 1009
    train_indices, validation_indices = _split_indices(
        len(train_set), args.validation_fraction, split_seed
    )
    train_loader = DataLoader(
        Subset(train_set, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )
    validation_loader = DataLoader(
        Subset(deterministic_set, validation_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    statistics_loader = DataLoader(
        Subset(deterministic_set, train_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    return train_loader, validation_loader, statistics_loader


@torch.no_grad()
def evaluate_validation(model, loader, device, task_id, limit=None, no_progress=False):
    was_training = model.training
    model.eval()
    psnr_values = []
    for index, (degraded, clean) in enumerate(
        tqdm(loader, desc="validation", leave=False, disable=no_progress)
    ):
        degraded = degraded.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)
        original_shape = degraded.shape[-2:]
        degraded, _ = pad_to_multiple(degraded, multiple=8)
        restored = crop_to_shape(model(degraded, task_id=task_id), original_shape)
        mse = (restored.clamp(0, 1) - clean).square().flatten(1).mean(1)
        psnr_values.extend((-10.0 * torch.log10(mse.clamp_min(1e-12))).tolist())
        if limit is not None and index + 1 >= limit:
            break
    model.train(was_training)
    return sum(psnr_values) / len(psnr_values)


@torch.no_grad()
def evaluate_test(
    model,
    loader,
    device,
    task_id=None,
    expected_task_id=None,
    limit=None,
    tile_size=384,
    tile_overlap=32,
    no_progress=False,
):
    was_training = model.training
    model.eval()
    psnr_values = []
    ssim_values = []
    route_hits = []
    for index, (_, degraded, clean_np) in enumerate(tqdm(loader, desc="eval", leave=False, disable=no_progress)):
        degraded = degraded.to(device, non_blocking=True)
        if task_id is None:
            routing_probs = model.route_task_probs(degraded)
            forward_kwargs = {"routing_probs": routing_probs}
            if routing_probs is not None and expected_task_id is not None:
                route_hits.extend(
                    (routing_probs.argmax(dim=1) == expected_task_id).float().tolist()
                )
        else:
            forward_kwargs = {"task_id": task_id}
        restored = tiled_forward(
            model,
            degraded,
            tile_size=tile_size,
            overlap=tile_overlap,
            multiple=8,
            forward_kwargs=forward_kwargs,
        )
        restored_np = tensor_to_rgb(restored)
        clean = clean_np.numpy()[0]
        psnr_values.append(calculate_psnr(restored_np, clean))
        ssim_values.append(calculate_ssim(restored_np, clean))
        if limit is not None and index + 1 >= limit:
            break
    if was_training:
        model.train()
    route_accuracy = (
        sum(route_hits) / len(route_hits) if route_hits else float("nan")
    )
    return (
        sum(psnr_values) / len(psnr_values),
        sum(ssim_values) / len(ssim_values),
        route_accuracy,
    )


def crop_to_shape(tensor, shape):
    h, w = shape
    return tensor[..., :h, :w]


def build_grad_scaler(device, enabled):
    enabled = enabled and device.type == "cuda"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler(device.type, enabled=enabled)
        except TypeError:
            return torch.cuda.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(device, enabled):
    enabled = enabled and device.type == "cuda"
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def tensor_summary(tensor):
    if tensor is None:
        return None
    detached = tensor.detach().float()
    finite = torch.isfinite(detached)
    summary = {
        "shape": list(detached.shape),
        "finite_fraction": float(finite.float().mean().item()),
    }
    if finite.any():
        values = detached[finite]
        summary.update(
            min=float(values.min().item()),
            max=float(values.max().item()),
            mean=float(values.mean().item()),
        )
    return summary


def write_numerical_diagnostic(
    args, task, task_index, epoch, step, scaler, tensors, model
):
    first_bad_parameter = None
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter.detach()).all():
            first_bad_parameter = name
            break
    payload = {
        "task": task,
        "task_index": task_index,
        "epoch": epoch,
        "step": step,
        "amp_scale": float(scaler.get_scale()) if scaler.is_enabled() else None,
        "first_nonfinite_parameter": first_bad_parameter,
        "tensors": {
            name: tensor_summary(tensor) for name, tensor in tensors.items()
        },
    }
    path = os.path.join(
        args.output_dir,
        f"nonfinite_{task}_epoch{epoch}_step{step}.json",
    )
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path


def first_nonfinite_optimizer_state(optimizer):
    for parameter_index, state in enumerate(optimizer.state.values()):
        for state_name, value in state.items():
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                return f"parameter[{parameter_index}].{state_name}"
    return None


def load_compatible_state(model, state):
    result = model.load_state_dict(state, strict=False)
    # Allow statistical memory missing keys when loading from standard models
    allowed_missing = {
        "statistical_memory.background_mean",
        "statistical_memory.background_cov",
        "statistical_memory.task_means",
        "statistical_memory.task_covs"
    }
    unexpected = list(result.unexpected_keys)
    missing = [key for key in result.missing_keys if key not in allowed_missing]
    if missing or unexpected:
        print(f"Checkpoint mismatch warning. Missing: {missing}; unexpected: {unexpected}")


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    task_index,
    epoch,
    best_metric,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "task_index": task_index,
            "epoch": epoch,
            "best_metric": best_metric,
        },
        path,
    )


def load_model_weights(path, model, device):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    for name, value in state.items():
        if (
            torch.is_tensor(value)
            and (value.is_floating_point() or value.is_complex())
            and not torch.isfinite(value).all()
        ):
            raise FloatingPointError(
                f"Checkpoint {path} contains non-finite model tensor: {name}"
            )
    load_compatible_state(model, state)
    return checkpoint


def configure_trainable_parameters(model, scope, task_index):
    for param in model.parameters():
        param.requires_grad = False

    if scope == "all":
        print(f"Scope is 'all' - Training full network for task {task_index}.")
        for param in model.parameters():
            param.requires_grad = True
        for param in model.frozen_extractor.parameters():
            param.requires_grad = False
        return

    trainable_markers = {
        "prompts": ("image_fuser", "feature_fuser"),
        "prompts_adapters": ("image_fuser", "feature_fuser", "image_prompt_adapter", "feature_prompt_adapter"),
        "ppa_scope": (
            "image_fuser",
            "feature_fuser",
            "image_prompt_adapter",
            "adapter_enc_level1",
            "adapter_enc_level2",
            "adapter_enc_level3",
            "feature_prompt_adapter",
            "adapter_dec_level3",
            "adapter_dec_level2",
            "adapter_dec_level1",
        ),
    }[scope]
    
    for name, param in model.named_parameters():
        if any(marker in name for marker in trainable_markers):
            param.requires_grad = True


def append_metric(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def log_metric(row):
    parts = [f"{key}={value}" for key, value in row.items() if value != ""]
    print("[metric] " + " ".join(parts), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Train EcoSMH for rehearsal-free continual image deraining.")
    parser.add_argument("--data-root", default="/mnt/netdisk/liumh/workspace/Image-deraining")
    parser.add_argument("--output-dir", default="runs/ecosmh_release")
    parser.add_argument("--tasks", nargs="+", default=["Rain800", "Rain100H"])
    parser.add_argument("--initial-task-index", type=int, default=0)
    parser.add_argument("--resume-state", default=None)
    parser.add_argument(
        "--resume-weights-only",
        action="store_true",
        help="Load model weights but reset optimizer, scheduler and epoch count.",
    )
    parser.add_argument(
        "--trainable-scope",
        choices=["all", "prompts", "prompts_adapters", "ppa_scope", "auto"],
        default="auto",
    )

    parser.add_argument("--epochs-per-task", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=100)
    parser.add_argument("--train-pad-multiple", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler-t-max", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--perceptual-weight", type=float, default=0.04)
    
    # Loss scaling parameters for aux distillation
    parser.add_argument("--zeta", type=float, default=1e-5)
    parser.add_argument("--eta", type=float, default=1e-5)
    
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--max-tasks", type=int, default=10)
    parser.add_argument("--prompts-per-task", type=int, default=None)
    parser.add_argument("--router-temperature", type=float, default=1.0)
    parser.add_argument("--covariance-shrinkage", type=float, default=0.1)
    parser.add_argument("--nsp-threshold", type=float, default=0.03)
    parser.add_argument("--nsp-strength", type=float, default=1.0)
    parser.add_argument("--adapter-modulation-limit", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-nonfinite-skips", type=int, default=3)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--tile-size", type=int, default=384)
    parser.add_argument("--tile-overlap", type=int, default=32)
    parser.add_argument("--max-steps-per-epoch", type=int, default=None)
    
    # Checkpointing arguments
    parser.set_defaults(restore_best_before_consolidation=True)
    parser.add_argument("--restore-best-before-consolidation", dest="restore_best_before_consolidation", action="store_true")
    parser.add_argument("--no-restore-best-before-consolidation", dest="restore_best_before_consolidation", action="store_false")
    parser.set_defaults(save_latest=True)
    parser.add_argument("--no-save-latest", dest="save_latest", action="store_false")
    
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--no-perceptual", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--skip-continual-eval", action="store_true")
    args = parser.parse_args()

    if args.initial_task_index + len(args.tasks) > args.max_tasks:
        parser.error("Task sequence exceeds --max-tasks")
    if not 0.0 < args.validation_fraction < 1.0:
        parser.error("--validation-fraction must be in (0, 1)")
    if args.adapter_modulation_limit <= 0:
        parser.error("--adapter-modulation-limit must be positive")
    if args.max_grad_norm <= 0:
        parser.error("--max-grad-norm must be positive")
    if args.max_nonfinite_skips < 0:
        parser.error("--max-nonfinite-skips cannot be negative")
    prompts_per_task = args.prompts_per_task
    if prompts_per_task is None:
        prompts_per_task = args.num_prompts // args.max_tasks
    if prompts_per_task <= 0:
        parser.error("--prompts-per-task must be positive")
    if (args.initial_task_index + len(args.tasks)) * prompts_per_task > args.num_prompts:
        parser.error("Private prompt blocks exceed --num-prompts capacity")
    args.prompts_per_task = prompts_per_task

    set_seed(args.seed)
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    
    model = EcoDPLPromptIR(
        num_prompts=args.num_prompts,
        max_tasks=args.max_tasks,
        router_temperature=args.router_temperature,
        covariance_shrinkage=args.covariance_shrinkage,
        adapter_modulation_limit=args.adapter_modulation_limit,
    ).to(device)
    
    resume_checkpoint = None
    if args.resume_state:
        resume_checkpoint = load_model_weights(args.resume_state, model, device)
        print(f"[checkpoint] resumed {args.resume_state}", flush=True)
        
    perceptual = None if args.no_perceptual else VGGPerceptualLoss().to(device)
    scaler = build_grad_scaler(device, args.amp)
    if (
        not args.resume_weights_only
        and
        scaler.is_enabled()
        and
        isinstance(resume_checkpoint, dict)
        and resume_checkpoint.get("scaler") is not None
    ):
        scaler.load_state_dict(resume_checkpoint["scaler"])
    from null_space import (
        apply_null_space_projection,
        compute_null_space_projectors,
        snapshot_projected_weights,
    )

    metrics_path = os.path.join(args.output_dir, "metrics.csv")
    continual_metrics_path = os.path.join(
        args.output_dir, "continual_test_metrics.csv"
    )
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "args.json"), "w") as handle:
        json.dump(vars(args), handle, indent=2, sort_keys=True)
    global_epoch = 0

    for local_task_index, task in enumerate(args.tasks):
        task_index = args.initial_task_index + local_task_index
        train_loader, validation_loader, statistics_loader = build_loaders(
            args, task, task_index
        )
        best_metric = -1.0
        start_epoch = 1
        block_start = task_index * prompts_per_task
        block_end = block_start + prompts_per_task
        model.set_active_prompt_range(block_start, block_end)
        print(
            f"[Prompts] Private range {block_start}:{block_end} for task {task_index}",
            flush=True,
        )

        # --- Task-specific optimizer and trainable parameters ---
        current_scope = args.trainable_scope
        if current_scope == "auto":
            current_scope = "all" if task_index == 0 else "ppa_scope"
            
        configure_trainable_parameters(model, current_scope, task_index)
        trainable_parameters = [param for param in model.parameters() if param.requires_grad]
        if not trainable_parameters:
            raise ValueError(f"No trainable parameters for trainable scope: {args.trainable_scope}")
            
        optimizer = torch.optim.AdamW(trainable_parameters, lr=args.lr, weight_decay=args.weight_decay)
        # CosineAnnealing per task
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, args.scheduler_t_max or args.epochs_per_task),
            eta_min=args.lr * 0.01,
        )
        if (
            not args.resume_weights_only
            and
            local_task_index == 0
            and isinstance(resume_checkpoint, dict)
            and resume_checkpoint.get("task_index") == task_index
            and resume_checkpoint.get("optimizer") is not None
        ):
            optimizer.load_state_dict(resume_checkpoint["optimizer"])
            bad_optimizer_state = first_nonfinite_optimizer_state(optimizer)
            if bad_optimizer_state is not None:
                raise FloatingPointError(
                    "Checkpoint contains non-finite optimizer state at "
                    f"{bad_optimizer_state}. Use an earlier best checkpoint "
                    "or --resume-weights-only."
                )
            if resume_checkpoint.get("scheduler") is not None:
                scheduler.load_state_dict(resume_checkpoint["scheduler"])
            start_epoch = int(resume_checkpoint.get("epoch", 0)) + 1
            best_metric = float(resume_checkpoint.get("best_metric", -1.0))
            global_epoch = (
                task_index * args.epochs_per_task + start_epoch - 1
            )
            print(
                f"[checkpoint] continuing task {task_index} at epoch "
                f"{start_epoch}/{args.epochs_per_task}",
                flush=True,
            )
        if start_epoch > args.epochs_per_task:
            raise ValueError(
                f"Checkpoint already reached epoch {start_epoch - 1}; "
                f"--epochs-per-task is {args.epochs_per_task}"
            )

        # Before training a new task, backup protected prompts
        model.backup_protected_prompts()
        
        # Compute Null Space projectors for SFTAdapters
        if task_index > 0:
            P_null_dict = compute_null_space_projectors(
                model,
                threshold=args.nsp_threshold,
                strength=args.nsp_strength,
            )
            nullities = {
                name: values["stats"]["input"]["nullity"]
                for name, values in P_null_dict.items()
            }
            print(
                f"[NSP] Computed {len(P_null_dict)} adapter projectors; "
                f"input nullities={nullities}."
            )
        else:
            P_null_dict = {}

        for epoch in range(start_epoch, args.epochs_per_task + 1):
            model.train()
            running_loss = 0.0
            steps_this_epoch = 0
            optimizer_steps = 0
            nonfinite_skips = 0
            progress = tqdm(train_loader, desc=f"{task} epoch {epoch}/{args.epochs_per_task}", disable=args.no_progress)
            
            for step, (degraded, clean) in enumerate(progress, start=1):
                degraded = degraded.to(device, non_blocking=True)
                clean = clean.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                original_shape = degraded.shape[-2:]
                
                if args.train_pad_multiple and args.train_pad_multiple > 1:
                    degraded, _ = pad_to_multiple(degraded, multiple=args.train_pad_multiple)

                with autocast_context(device, scaler.is_enabled()):
                    restored, aux = model(degraded, return_aux=True)
                    restored = crop_to_shape(restored, original_shape)
                    reconstruction_loss = F.smooth_l1_loss(restored, clean)
                    image_distance_loss = aux["image_distance"]
                    feature_distance_loss = aux["feature_distance"]
                    loss = args.alpha * reconstruction_loss
                    loss = loss + args.zeta * image_distance_loss
                    loss = loss + args.eta * feature_distance_loss

                perceptual_loss = torch.zeros((), device=device)
                if perceptual is not None and args.perceptual_weight > 0:
                    with autocast_context(device, False):
                        perceptual_loss = perceptual(
                            restored.float(), clean.float()
                        )
                        loss = loss + args.perceptual_weight * perceptual_loss

                loss_tensors = {
                    "degraded": degraded,
                    "clean": clean,
                    "restored": restored,
                    "loss": loss,
                    "reconstruction_loss": reconstruction_loss,
                    "image_distance_loss": image_distance_loss,
                    "feature_distance_loss": feature_distance_loss,
                    "perceptual_loss": perceptual_loss,
                    "image_logits": aux["image_logits"],
                    "feature_logits": aux["feature_logits"],
                }
                if not torch.isfinite(loss):
                    diagnostic_path = write_numerical_diagnostic(
                        args,
                        task,
                        task_index,
                        epoch,
                        step,
                        scaler,
                        loss_tensors,
                        model,
                    )
                    raise FloatingPointError(
                        "Non-finite forward loss. Training stopped before "
                        f"backward; diagnostics: {diagnostic_path}"
                    )

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        trainable_parameters, args.max_grad_norm
                    )
                    if not torch.isfinite(grad_norm):
                        reduced_scale = max(float(scaler.get_scale()) / 2.0, 1.0)
                        optimizer.zero_grad(set_to_none=True)
                        scaler.update(new_scale=reduced_scale)
                        nonfinite_skips += 1
                        progress.set_postfix(
                            loss=f"{loss.item():.4f}",
                            skipped_nonfinite=nonfinite_skips,
                        )
                        if nonfinite_skips > args.max_nonfinite_skips:
                            diagnostic_path = write_numerical_diagnostic(
                                args,
                                task,
                                task_index,
                                epoch,
                                step,
                                scaler,
                                {
                                    **loss_tensors,
                                    "gradient_norm": grad_norm,
                                },
                                model,
                            )
                            raise FloatingPointError(
                                "Repeated non-finite AMP gradients; "
                                f"diagnostics: {diagnostic_path}"
                            )
                        continue
                    
                    # Frozen prompt slices must not receive optimizer updates.
                    model.zero_protected_prompt_grads()
                    
                    if P_null_dict:
                        old_weights = snapshot_projected_weights(
                            model, P_null_dict
                        )
                        
                    scaler.step(optimizer)
                    scaler.update()
                    
                    if P_null_dict:
                        apply_null_space_projection(model, P_null_dict, old_weights)
                        
                    optimizer_steps += 1
                else:
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        trainable_parameters, args.max_grad_norm
                    )
                    if not torch.isfinite(grad_norm):
                        diagnostic_path = write_numerical_diagnostic(
                            args,
                            task,
                            task_index,
                            epoch,
                            step,
                            scaler,
                            {
                                **loss_tensors,
                                "gradient_norm": grad_norm,
                            },
                            model,
                        )
                        raise FloatingPointError(
                            "Non-finite gradients without AMP; "
                            f"diagnostics: {diagnostic_path}"
                        )
                    
                    # Frozen prompt slices must not receive optimizer updates.
                    model.zero_protected_prompt_grads()
                    
                    if P_null_dict:
                        old_weights = snapshot_projected_weights(
                            model, P_null_dict
                        )
                        
                    optimizer.step()
                    
                    if P_null_dict:
                        apply_null_space_projection(model, P_null_dict, old_weights)
                        
                    optimizer_steps += 1

                # Restore protected prompts to reverse any weight decay from AdamW
                model.restore_protected_prompts()

                running_loss += loss.item()
                steps_this_epoch = step
                postfix_dict = {"loss": f"{loss.item():.4f}"}
                progress.set_postfix(**postfix_dict)
                if args.max_steps_per_epoch is not None and step >= args.max_steps_per_epoch:
                    break

            if optimizer_steps > 0:
                scheduler.step()
            global_epoch += 1

            row = {
                "task": task,
                "task_index": task_index,
                "epoch": epoch,
                "global_epoch": global_epoch,
                "train_loss": running_loss / max(1, steps_this_epoch),
                "lr": optimizer.param_groups[0]["lr"],
                "validation_psnr": "",
            }

            should_eval = epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs_per_task
            if should_eval:
                validation_psnr = evaluate_validation(
                    model,
                    validation_loader,
                    device,
                    task_id=task_index,
                    limit=args.eval_limit,
                    no_progress=args.no_progress,
                )
                row["validation_psnr"] = f"{validation_psnr:.4f}"
                metric = validation_psnr
                if metric > best_metric:
                    best_metric = metric
                    save_checkpoint(
                        os.path.join(args.output_dir, f"best_{task}.pth"),
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        task_index,
                        epoch,
                        best_metric,
                    )
            append_metric(metrics_path, row)
            log_metric(row)
            if args.save_latest:
                save_checkpoint(
                    os.path.join(args.output_dir, "latest.pth"),
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    task_index,
                    epoch,
                    best_metric,
                )

        best_path = os.path.join(args.output_dir, f"best_{task}.pth")
        
        if args.restore_best_before_consolidation and os.path.exists(best_path):
            load_model_weights(best_path, model, device)
            print(f"[checkpoint] restored {best_path} before SMH update", flush=True)

        print(f"[SMH] Updating Statistical Memory for task {task_index}...", flush=True)
        model.update_task_statistics(task_index, statistics_loader, device)

        save_checkpoint(
            os.path.join(args.output_dir, f"after_{task}.pth"),
            model,
            optimizer,
            scheduler,
            scaler,
            task_index,
            args.epochs_per_task,
            best_metric,
        )

        if not args.skip_continual_eval:
            for seen_offset, seen_task in enumerate(args.tasks[: local_task_index + 1]):
                seen_task_id = args.initial_task_index + seen_offset
                test_loader = DataLoader(
                    ImagePairDataset(args.data_root, seen_task),
                    batch_size=1,
                    shuffle=False,
                    num_workers=0,
                )
                for mode in ("oracle", "routed"):
                    oracle_task_id = seen_task_id if mode == "oracle" else None
                    psnr, ssim, route_accuracy = evaluate_test(
                        model,
                        test_loader,
                        device,
                        task_id=oracle_task_id,
                        expected_task_id=seen_task_id,
                        limit=args.eval_limit,
                        tile_size=args.tile_size,
                        tile_overlap=args.tile_overlap,
                        no_progress=args.no_progress,
                    )
                    append_metric(
                        continual_metrics_path,
                        {
                            "after_task": task,
                            "after_task_index": task_index,
                            "eval_task": seen_task,
                            "eval_task_index": seen_task_id,
                            "mode": mode,
                            "psnr": f"{psnr:.4f}",
                            "ssim": f"{ssim:.4f}",
                            "router_accuracy": (
                                "" if mode == "oracle" else f"{route_accuracy:.4f}"
                            ),
                        },
                    )


if __name__ == "__main__":
    main()
