import os
import h5py
import numpy as np
import cv2

def create_h5(path, num_samples, shape=(3, 256, 256)):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, 'w') as f:
        for i in range(num_samples):
            dummy_img = np.random.randint(0, 255, size=shape, dtype=np.uint8)
            f.create_dataset(str(i), data=dummy_img)

def create_images(dir_path, num_samples, shape=(256, 256, 3)):
    os.makedirs(dir_path, exist_ok=True)
    for i in range(num_samples):
        dummy_img = np.random.randint(0, 255, size=shape, dtype=np.uint8)
        cv2.imwrite(os.path.join(dir_path, f"{i}.png"), dummy_img)

def main():
    base_dir = "dummy_datasets"
    num_train = 4
    num_test = 2

    # Rain800
    create_h5(os.path.join(base_dir, "Rain800", "train_input.h5"), num_train)
    create_h5(os.path.join(base_dir, "Rain800", "train_target.h5"), num_train)
    create_images(os.path.join(base_dir, "Rain800", "inputTest"), num_test)
    create_images(os.path.join(base_dir, "Rain800", "targetTest"), num_test)

    # Rain100H
    create_h5(os.path.join(base_dir, "RainTrainH", "train_input.h5"), num_train)
    create_h5(os.path.join(base_dir, "RainTrainH", "train_target.h5"), num_train)
    create_images(os.path.join(base_dir, "RainTestH", "rain"), num_test)
    create_images(os.path.join(base_dir, "RainTestH", "norain"), num_test)

if __name__ == "__main__":
    main()
