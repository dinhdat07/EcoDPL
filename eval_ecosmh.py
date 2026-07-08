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
    result = model.load_state_dict(state, strict=False)
    allowed_missing = {
        "image_fuser.protected",
        "feature_fuser.protected",
        "image_fuser.dictionary",
        "feature_fuser.dictionary",
    }
    unexpected = list(result.unexpected_keys)
    missing = [key for key in result.missing_keys if key not in allowed_missing]
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
    parser.add_argument("--prompt-range", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tile-size", type=int, default=384)
    parser.add_argument("--tile-overlap", type=int, default=32)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    model = EcoDPLPromptIR(num_prompts=args.num_prompts).to(device)
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

    for index, (name, degraded, clean_np) in enumerate(tqdm(loader, desc=f"eval {args.task}", disable=args.no_progress)):
        degraded = degraded.to(device)
        restored = tiled_forward(model, degraded, tile_size=args.tile_size, overlap=args.tile_overlap, multiple=8)
        restored_np = tensor_to_rgb(restored)
        clean = clean_np.numpy()[0]

        psnr_values.append(calculate_psnr(restored_np, clean))
        ssim_values.append(calculate_ssim(restored_np, clean))

        if args.output_dir:
            save_rgb(os.path.join(args.output_dir, name[0]), restored_np)
        if args.limit is not None and index + 1 >= args.limit:
            break

    print(f"{args.task}: PSNR={sum(psnr_values) / len(psnr_values):.4f}, SSIM={sum(ssim_values) / len(ssim_values):.4f}, N={len(psnr_values)}")


if __name__ == "__main__":
    main()
