import os
import random
from dataclasses import dataclass

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


@dataclass(frozen=True)
class DerainTask:
    name: str
    train_dir: str
    test_input: str
    test_target: str
    naming: str


TASKS = {
    "Rain800": DerainTask("Rain800", "Rain800", "Rain800/inputTest", "Rain800/targetTest", "same"),
    "Rain100H": DerainTask("Rain100H", "RainTrainH", "RainTestH/rain", "RainTestH/norain", "rain_to_norain"),
    "RainTrainH": DerainTask("Rain100H", "RainTrainH", "RainTestH/rain", "RainTestH/norain", "rain_to_norain"),
    "Rain100L": DerainTask("Rain100L", "RainTrainL", "RainTestL/rain", "RainTestL/norain", "rain_to_norain"),
    "RainTrainL": DerainTask("Rain100L", "RainTrainL", "RainTestL/rain", "RainTestL/norain", "rain_to_norain"),
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class H5DerainDataset(Dataset):
    def __init__(self, root, task, patch_size=96, augment=True):
        if task not in TASKS:
            raise KeyError(f"Unknown task {task}. Available: {sorted(TASKS)}")
        self.root = root
        self.task = TASKS[task]
        self.patch_size = patch_size
        self.augment = augment
        self.input_path = os.path.join(root, self.task.train_dir, "train_input.h5")
        self.target_path = os.path.join(root, self.task.train_dir, "train_target.h5")
        self._input_h5 = None
        self._target_h5 = None

        with h5py.File(self.input_path, "r") as h5_file:
            self.length = len(h5_file)

    def _ensure_open(self):
        if self._input_h5 is None:
            self._input_h5 = h5py.File(self.input_path, "r")
            self._target_h5 = h5py.File(self.target_path, "r")

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        self._ensure_open()
        key = str(index)
        degraded = np.asarray(self._input_h5[key], dtype=np.float32)
        clean = np.asarray(self._target_h5[key], dtype=np.float32)

        _, h, w = degraded.shape
        if self.patch_size and (h < self.patch_size or w < self.patch_size):
            pad_h = max(0, self.patch_size - h)
            pad_w = max(0, self.patch_size - w)
            degraded = np.pad(
                degraded, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect"
            )
            clean = np.pad(
                clean, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect"
            )
            _, h, w = degraded.shape
        if self.patch_size and (h > self.patch_size or w > self.patch_size):
            if self.augment:
                top = random.randint(0, h - self.patch_size)
                left = random.randint(0, w - self.patch_size)
            else:
                top = max(0, (h - self.patch_size) // 2)
                left = max(0, (w - self.patch_size) // 2)
            degraded = degraded[:, top:top + self.patch_size, left:left + self.patch_size]
            clean = clean[:, top:top + self.patch_size, left:left + self.patch_size]

        if self.augment:
            if random.random() < 0.5:
                degraded = degraded[:, :, ::-1].copy()
                clean = clean[:, :, ::-1].copy()
            if random.random() < 0.5:
                degraded = degraded[:, ::-1, :].copy()
                clean = clean[:, ::-1, :].copy()

        return torch.from_numpy(degraded.copy()), torch.from_numpy(clean.copy())


class ImagePairDataset(Dataset):
    def __init__(self, root, task):
        if task not in TASKS:
            raise KeyError(f"Unknown task {task}. Available: {sorted(TASKS)}")
        self.root = root
        self.task = TASKS[task]
        self.input_dir = os.path.join(root, self.task.test_input)
        self.target_dir = os.path.join(root, self.task.test_target)
        self.names = sorted([name for name in os.listdir(self.input_dir) if name.lower().endswith(IMAGE_EXTENSIONS)])

    def __len__(self):
        return len(self.names)

    def _target_name(self, name):
        if self.task.naming == "same":
            return name
        return name.replace("rain-", "norain-", 1)

    def __getitem__(self, index):
        name = self.names[index]
        degraded = read_rgb(os.path.join(self.input_dir, name))
        clean = read_rgb(os.path.join(self.target_dir, self._target_name(name)))
        h = min(degraded.shape[0], clean.shape[0])
        w = min(degraded.shape[1], clean.shape[1])
        degraded = degraded[:h, :w]
        clean = clean[:h, :w]
        tensor = torch.from_numpy(degraded.transpose(2, 0, 1)).float()
        return name, tensor, clean


def read_rgb(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image.astype(np.float32) / 255.0


def pad_to_multiple(tensor, multiple=8):
    _, _, h, w = tensor.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return tensor, (h, w)
    padded = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
    return padded, (h, w)


def tensor_to_rgb(output, shape=None):
    if shape is not None:
        h, w = shape
        output = output[..., :h, :w]
    array = output.detach().clamp(0, 1).cpu().numpy()[0].transpose(1, 2, 0)
    return array


@torch.no_grad()
def tiled_forward(
    model, image, tile_size=384, overlap=32, multiple=8, forward_kwargs=None
):
    forward_kwargs = {} if forward_kwargs is None else dict(forward_kwargs)
    _, _, h, w = image.shape
    if tile_size is None or tile_size <= 0 or (h <= tile_size and w <= tile_size):
        padded, original_shape = pad_to_multiple(image, multiple=multiple)
        return model(padded, **forward_kwargs)[..., :original_shape[0], :original_shape[1]]

    stride = max(1, tile_size - overlap)
    ys = list(range(0, max(h - tile_size, 0) + 1, stride))
    xs = list(range(0, max(w - tile_size, 0) + 1, stride))
    if not ys or ys[-1] != h - tile_size:
        ys.append(max(h - tile_size, 0))
    if not xs or xs[-1] != w - tile_size:
        xs.append(max(w - tile_size, 0))

    output = torch.zeros_like(image)
    weight = torch.zeros((image.shape[0], 1, h, w), dtype=image.dtype, device=image.device)
    for top in ys:
        for left in xs:
            patch = image[..., top:top + tile_size, left:left + tile_size]
            padded, patch_shape = pad_to_multiple(patch, multiple=multiple)
            restored = model(padded, **forward_kwargs)[..., :patch_shape[0], :patch_shape[1]]
            output[..., top:top + patch_shape[0], left:left + patch_shape[1]] += restored
            weight[..., top:top + patch_shape[0], left:left + patch_shape[1]] += 1.0
    return output / weight.clamp_min(1.0)


def calculate_psnr(img1, img2):
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse <= 1e-12:
        return float("inf")
    return 20.0 * np.log10(1.0 / np.sqrt(mse))


def calculate_ssim(img1, img2):
    from skimage.metrics import structural_similarity

    win_size = min(7, img1.shape[0], img1.shape[1])
    if win_size % 2 == 0:
        win_size -= 1
    win_size = max(win_size, 3)
    try:
        return structural_similarity(img1, img2, data_range=1.0, channel_axis=2, win_size=win_size)
    except (TypeError, ValueError):
        return structural_similarity(img1, img2, data_range=1.0, multichannel=True, win_size=win_size)


def save_rgb(path, image):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image = np.uint8(np.clip(image, 0, 1) * 255.0)
    cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
