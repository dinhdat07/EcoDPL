import argparse
import csv
import json
import os
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
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


class ParameterRegularizer:
    def __init__(self, device, normalize_importance=True, mode="l1", importance_floor=0.0):
        self.device = device
        self.normalize_importance = normalize_importance
        self.mode = mode
        self.importance_floor = importance_floor
        self.star = None
        self.importance = None

    def penalty(self, model):
        if self.star is None or self.importance is None:
            return torch.tensor(0.0, device=self.device)
        total = torch.tensor(0.0, device=self.device)
        count = 0
        for param, star, importance in zip(model.parameters(), self.star, self.importance):
            if not param.requires_grad:
                continue
            importance = importance.to(param.device)
            if self.normalize_importance:
                importance = importance / importance.detach().mean().clamp_min(1e-12)
            if self.importance_floor > 0:
                importance = importance + self.importance_floor
            diff = param - star.to(param.device)
            if self.mode == "l2":
                total = total + torch.mean(importance * diff.square())
            else:
                total = total + torch.mean(importance * diff.abs())
            count += 1
        return total / max(count, 1)

    @torch.no_grad()
    def consolidate(self, model, loader, max_batches=50, pad_multiple=8, microbatch_size=4, use_amp=False, no_progress=False):
        model.train()
        importance = [torch.zeros_like(param, device=self.device) for param in model.parameters()]
        seen = 0
        for degraded, clean in tqdm(loader, total=min(len(loader), max_batches), desc="importance", leave=False, disable=no_progress):
            degraded = degraded.to(self.device, non_blocking=True)
            clean = clean.to(self.device, non_blocking=True)
            model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                if microbatch_size is None or microbatch_size <= 0:
                    microbatch_size = degraded.shape[0]
                degraded_chunks = degraded.split(microbatch_size)
                clean_chunks = clean.split(microbatch_size)
                chunk_count = len(degraded_chunks)
                for degraded_chunk, clean_chunk in zip(degraded_chunks, clean_chunks):
                    original_shape = degraded_chunk.shape[-2:]
                    if pad_multiple and pad_multiple > 1:
                        degraded_chunk, _ = pad_to_multiple(degraded_chunk, multiple=pad_multiple)
                    with autocast_context(self.device, use_amp):
                        restored = model(degraded_chunk)
                        restored = crop_to_shape(restored, original_shape)
                        loss = F.smooth_l1_loss(restored, clean_chunk) / chunk_count
                    loss.backward()
            for slot, param in zip(importance, model.parameters()):
                if param.grad is not None:
                    slot.add_(param.grad.detach().abs())
            seen += 1
            if seen >= max_batches:
                break
        if seen > 0:
            importance = [slot / seen for slot in importance]
        self.importance = importance
        self.star = [param.detach().clone() for param in model.parameters()]
        model.zero_grad(set_to_none=True)


def build_train_loader(args, task, batch_size=None):
    train_set = H5DerainDataset(args.data_root, task, patch_size=args.patch_size, augment=True)
    return DataLoader(
        train_set,
        batch_size=batch_size or args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )


def build_loaders(args, task):
    train_loader = build_train_loader(args, task)
    eval_set = ImagePairDataset(args.data_root, task)
    eval_loader = DataLoader(eval_set, batch_size=1, shuffle=False, num_workers=0)
    return train_loader, eval_loader


@torch.no_grad()
def evaluate(model, loader, device, limit=None, tile_size=384, tile_overlap=32, no_progress=False):
    was_training = model.training
    model.eval()
    psnr_values = []
    ssim_values = []
    for index, (_, degraded, clean_np) in enumerate(tqdm(loader, desc="eval", leave=False, disable=no_progress)):
        degraded = degraded.to(device, non_blocking=True)
        restored = tiled_forward(model, degraded, tile_size=tile_size, overlap=tile_overlap, multiple=8)
        restored_np = tensor_to_rgb(restored)
        clean = clean_np.numpy()[0]
        psnr_values.append(calculate_psnr(restored_np, clean))
        ssim_values.append(calculate_ssim(restored_np, clean))
        if limit is not None and index + 1 >= limit:
            break
    if was_training:
        model.train()
    return sum(psnr_values) / len(psnr_values), sum(ssim_values) / len(ssim_values)


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


def save_checkpoint(path, model, optimizer, scheduler, task_index, epoch, best_metric):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
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
    load_compatible_state(model, state)
    return checkpoint


def configure_trainable_parameters(model, scope, task_index):
    for param in model.parameters():
        param.requires_grad = False

    if scope == "all":
        print(f"Scope is 'all' - Training full network for task {task_index}.")
        for param in model.parameters():
            param.requires_grad = True
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
        "--trainable-scope",
        choices=["all", "prompts", "prompts_adapters", "decoder_prompts", "auto"],
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
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    
    model = EcoDPLPromptIR(num_prompts=args.num_prompts).to(device)
    
    if args.resume_state:
        checkpoint = load_model_weights(args.resume_state, model, device)
        print(f"[checkpoint] resumed {args.resume_state}", flush=True)
        
    perceptual = None if args.no_perceptual else VGGPerceptualLoss().to(device)
    scaler = build_grad_scaler(device, args.amp)
    from null_space import compute_null_space_projectors, apply_null_space_projection

    metrics_path = os.path.join(args.output_dir, "metrics.csv")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "args.json"), "w") as handle:
        json.dump(vars(args), handle, indent=2, sort_keys=True)
    global_epoch = 0

    for local_task_index, task in enumerate(args.tasks):
        task_index = args.initial_task_index + local_task_index
        train_loader, eval_loader = build_loaders(args, task)
        best_metric = -1.0
        # --- Allocate active prompts progressively ---
        total_tasks = 5 # EcoDPL has 5 tasks usually
        prompts_per_task = args.num_prompts // total_tasks
        # Cap the max prompts to num_prompts
        current_active_end = min(args.num_prompts, (task_index + 1) * prompts_per_task)
        model.set_active_prompt_range(0, current_active_end)
        print(f"[Prompts] Activated range 0 to {current_active_end} for task {task_index}", flush=True)

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
            T_max=max(1, args.epochs_per_task),
            eta_min=args.lr * 0.01,
        )

        # Before training a new task, backup protected prompts
        model.backup_protected_prompts()
        
        # Compute Null Space projectors for SFTAdapters
        if task_index > 0:
            P_null_dict = compute_null_space_projectors(model, args.patch_size)
            print(f"[NSP] Computed {len(P_null_dict)} projectors.")
        else:
            P_null_dict = {}

        for epoch in range(1, args.epochs_per_task + 1):
            model.train()
            running_loss = 0.0
            steps_this_epoch = 0
            optimizer_steps = 0
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
                    loss = args.alpha * F.smooth_l1_loss(restored, clean)
                    loss = loss + args.zeta * aux["image_distance"] + args.eta * aux["feature_distance"]

                if perceptual is not None and args.perceptual_weight > 0:
                    with autocast_context(device, False):
                        loss = loss + args.perceptual_weight * perceptual(restored.float(), clean.float())

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    
                    # Prevent AdamW momentum explosion for frozen protected prompts
                    if task_index > 0:
                        model.zero_protected_prompt_grads()
                    
                    if task_index > 0:
                        old_weights = {name: p.data.clone() for name, p in model.named_parameters() if p.requires_grad}
                        
                    scaler.step(optimizer)
                    scaler.update()
                    
                    if task_index > 0:
                        apply_null_space_projection(model, P_null_dict, old_weights)
                        
                    optimizer_steps += 1
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    
                    # Prevent AdamW momentum explosion for frozen protected prompts
                    if task_index > 0:
                        model.zero_protected_prompt_grads()
                    
                    if task_index > 0:
                        old_weights = {name: p.data.clone() for name, p in model.named_parameters() if p.requires_grad}
                        
                    optimizer.step()
                    
                    if task_index > 0:
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
                "psnr": "",
                "ssim": "",
            }

            should_eval = epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs_per_task
            if should_eval:
                psnr, ssim = evaluate(
                    model,
                    eval_loader,
                    device,
                    limit=args.eval_limit,
                    tile_size=args.tile_size,
                    tile_overlap=args.tile_overlap,
                    no_progress=args.no_progress,
                )
                row["psnr"] = f"{psnr:.4f}"
                row["ssim"] = f"{ssim:.4f}"
                metric = psnr
                if metric > best_metric:
                    best_metric = metric
                    save_checkpoint(
                        os.path.join(args.output_dir, f"best_{task}.pth"),
                        model,
                        optimizer,
                        scheduler,
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
                    task_index,
                    epoch,
                    best_metric,
                )

        best_path = os.path.join(args.output_dir, f"best_{task}.pth")
        
        if args.restore_best_before_consolidation and os.path.exists(best_path):
            load_model_weights(best_path, model, device)
            print(f"[checkpoint] restored {best_path} before SMH update", flush=True)

        print(f"[SMH] Updating Statistical Memory for task {task_index}...", flush=True)
        model.update_task_statistics(task_index, train_loader, device)

        save_checkpoint(
            os.path.join(args.output_dir, f"after_{task}.pth"),
            model,
            optimizer,
            scheduler,
            task_index,
            args.epochs_per_task,
            best_metric,
        )


if __name__ == "__main__":
    main()
