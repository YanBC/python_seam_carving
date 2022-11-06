import webp
import cv2 as cv
import os
import numpy as np
from typing import List

from carver.carve_slow import minimum_seam, backward_energy_opencv


def generate_webp(file_path: str, images: List[np.ndarray], max_size: int = 600) -> None:
    height, width, _ = images[0].shape
    scale_ratio = max_size / max(height, width)
    resized = []
    if scale_ratio < 1.:
        for image in images:
            img = cv.resize(image, (0, 0), fx=scale_ratio, fy=scale_ratio)
            img = cv.cvtColor(img, cv.COLOR_RGBA2BGR)
            resized.append(img)
    webp.mimwrite(file_path, resized, fps=24)


if __name__ == '__main__':
    demo_image = "images/Broadway_tower_edit.jpg"
    image = cv.imread(demo_image)
    height, width, _ = image.shape

    save_dir = "temp"
    # for i in range(width - height):
    #     empty = np.zeros((height, width, 3), dtype=np.uint8)
    #     energy = backward_energy_opencv(image)
    #     M, backtrack = minimum_seam(energy)

    #     carve_once = np.stack([energy] * 3, axis=2)
    #     seam_color = np.array([0, 255, 0], dtype=np.uint8).reshape((1, 1, 3))

    #     col = np.argmin(M[-1])
    #     for row in reversed(range(height)):
    #         carve_once[row, col, :] = seam_color
    #         col = backtrack[row, col]
    #     # cv.imwrite("carve_once.jpg", carve_once)

    #     once_height, once_width, _ = carve_once.shape
    #     empty[:once_height, :once_width, :] = carve_once
    #     cv.imwrite(f"{save_dir}/{i}.jpg", empty)

    #     r, c, _ = image.shape
    #     mask = np.ones((r, c), dtype=np.bool)
    #     j = np.argmin(M[-1])
    #     for i in reversed(range(r)):
    #         mask[i, j] = False
    #         j = backtrack[i, j]
    #     mask = np.stack([mask] * 3, axis=2)
    #     image = image[mask].reshape((r, c - 1, 3))

    files = sorted(os.listdir(save_dir), key=lambda x: int(x.split(".")[0]))

    images = []
    for file in files:
        image_path = os.path.join(save_dir, file)
        images.append(cv.imread(image_path))

    generate_webp("process.webp", images)
