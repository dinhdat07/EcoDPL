import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from net.ecosmh_promptir import EcoDPLPromptIR
from utils.derain_release import (
    ImagePairDataset,
    calculate_psnr,
    calculate_ssim,
    save_rgb,
    tensor_to_rgb,
    tiled_forward,
)


def load_compatible_state(model, state):
    for name, value in state.items():
        if (
            torch.is_tensor(value)
            and (value.is_floating_point() or value.is_complex())
            and not torch.isfinite(value).all()
        ):
            raise FloatingPointError(
                f"Checkpoint contains non-finite model tensor: {name}"
            )
    result = model.load_state_dict(state, strict=False)
    allowed_missing_fragments = (
        "nsp_input_cov",
        "nsp_gamma_hidden_cov",
        "nsp_beta_hidden_cov",
        "nsp_input_count",
        "nsp_gamma_hidden_count",
        "nsp_beta_hidden_count",
        "stat_memory.log_dets",
        "stat_memory.task_valid",
        "protected_mask",
        "task_prompt_mask",
    )
    legacy_unexpected_fragments = (
        "stat_memory.bg_",
        ".protected",
        ".dictionary",
    )
    unexpected = [
        key
        for key in result.unexpected_keys
        if not any(fragment in key for fragment in legacy_unexpected_fragments)
    ]
    missing = [
        key
        for key in result.missing_keys
        if not any(fragment in key for fragment in allowed_missing_fragments)
    ]
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch. Missing: {missing}; unexpected: {unexpected}")


def parse_prompt_range(value):
    if not value:
        return None
    start, end = value.split(":", 1)
    return int(start) if start else None, int(end) if end else None


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Evaluate EcoSMH deraining checkpoints.")
    parser.add_argument("--data-root", default="/mnt/netdisk/liumh/workspace/Image-deraining")
    parser.add_argument("--task", default="Rain800")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--max-tasks", type=int, default=10)
    parser.add_argument("--adapter-modulation-limit", type=float, default=1.0)
    parser.add_argument(
        "--task-id",
        type=int,
        default=None,
        help="Use oracle routing for diagnostics; omit for task-agnostic routing.",
    )
    parser.add_argument("--prompt-range", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tile-size", type=int, default=384)
    parser.add_argument("--tile-overlap", type=int, default=32)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    model = EcoDPLPromptIR(
        num_prompts=args.num_prompts,
        max_tasks=args.max_tasks,
        adapter_modulation_limit=args.adapter_modulation_limit,
    ).to(device)
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location=device)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    load_compatible_state(model, state)
    prompt_range = parse_prompt_range(args.prompt_range)
    if prompt_range is not None:
        model.set_active_prompt_range(*prompt_range)
    model.eval()

    dataset = ImagePairDataset(args.data_root, args.task)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    psnr_values = []
    ssim_values = []
    routed_tasks = []

    for index, (name, degraded, clean_np) in enumerate(tqdm(loader, desc=f"eval {args.task}", disable=args.no_progress)):
        degraded = degraded.to(device)
        if args.task_id is None:
            routing_probs = model.route_task_probs(degraded)
            forward_kwargs = {"routing_probs": routing_probs}
            if routing_probs is not None:
                routed_tasks.extend(routing_probs.argmax(dim=1).tolist())
        else:
            forward_kwargs = {"task_id": args.task_id}
        restored = tiled_forward(
            model,
            degraded,
            tile_size=args.tile_size,
            overlap=args.tile_overlap,
            multiple=8,
            forward_kwargs=forward_kwargs,
        )
        restored_np = tensor_to_rgb(restored)
        clean = clean_np.numpy()[0]

        psnr_values.append(calculate_psnr(restored_np, clean))
        ssim_values.append(calculate_ssim(restored_np, clean))

        if args.output_dir:
            save_rgb(os.path.join(args.output_dir, name[0]), restored_np)
        if args.limit is not None and index + 1 >= args.limit:
            break

    print(f"{args.task}: PSNR={sum(psnr_values) / len(psnr_values):.4f}, SSIM={sum(ssim_values) / len(ssim_values):.4f}, N={len(psnr_values)}")
    if routed_tasks:
        counts = {
            task_id: routed_tasks.count(task_id) for task_id in sorted(set(routed_tasks))
        }
        print(f"Router task counts: {counts}")


if __name__ == "__main__":
    main()
