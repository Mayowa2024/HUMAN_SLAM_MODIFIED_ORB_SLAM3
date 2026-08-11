#!/usr/bin/env python3

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np


def apply_gamma(img, gamma):
    if gamma == 1.0:
        return img

    inv_gamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255
        for i in range(256)
    ]).astype("uint8")

    return cv2.LUT(img, table)


def apply_vignette(img, strength):
    if strength <= 0:
        return img

    h, w = img.shape[:2]

    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    X, Y = np.meshgrid(x, y)

    dist = np.sqrt(X**2 + Y**2)
    mask = 1 - strength * dist
    mask = np.clip(mask, 0.15, 1.0)

    if len(img.shape) == 2:
        out = img.astype(np.float32) * mask
    else:
        out = img.astype(np.float32) * mask[:, :, None]

    return np.clip(out, 0, 255).astype(np.uint8)


def perturb_image(img, brightness, contrast, gamma, blur, noise_std, vignette):
    # contrast + brightness: output = contrast * image + brightness
    out = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)

    out = apply_gamma(out, gamma)

    if blur > 0:
        # kernel must be odd
        k = blur if blur % 2 == 1 else blur + 1
        out = cv2.GaussianBlur(out, (k, k), 0)

    if noise_std > 0:
        noise = np.random.normal(0, noise_std, out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    out = apply_vignette(out, vignette)

    return out


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--src_seq", required=True, help="Original KITTI sequence folder, e.g. sequences/06")
    parser.add_argument("--out_seq", required=True, help="Output perturbed sequence folder")

    parser.add_argument("--start", type=int, default=0, help="First frame to perturb")
    parser.add_argument("--end", type=int, default=-1, help="Last frame to perturb, inclusive. -1 means all frames")

    parser.add_argument("--brightness", type=float, default=0.0, help="Additive brightness beta")
    parser.add_argument("--contrast", type=float, default=1.0, help="Contrast alpha")
    parser.add_argument("--gamma", type=float, default=1.0, help="Gamma correction")
    parser.add_argument("--blur", type=int, default=0, help="Gaussian blur kernel size, e.g. 5")
    parser.add_argument("--noise_std", type=float, default=0.0, help="Gaussian noise std")
    parser.add_argument("--vignette", type=float, default=0.0, help="Vignette strength, e.g. 0.4")

    args = parser.parse_args()

    src_seq = Path(args.src_seq).expanduser()
    out_seq = Path(args.out_seq).expanduser()

    if not src_seq.exists():
        raise FileNotFoundError(f"Source sequence not found: {src_seq}")

    if out_seq.exists():
        raise FileExistsError(f"Output already exists, refusing to overwrite: {out_seq}")

    print(f"Copying sequence:\n  from: {src_seq}\n  to:   {out_seq}")
    shutil.copytree(src_seq, out_seq)

    for cam in ["image_0", "image_1"]:
        img_dir = out_seq / cam
        images = sorted(img_dir.glob("*.png"))

        if not images:
            print(f"No images found in {img_dir}")
            continue

        end = args.end if args.end >= 0 else len(images) - 1
        end = min(end, len(images) - 1)

        print(f"Perturbing {cam}: frames {args.start} to {end}")

        for idx in range(args.start, end + 1):
            path = images[idx]

            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue

            out = perturb_image(
                img,
                brightness=args.brightness,
                contrast=args.contrast,
                gamma=args.gamma,
                blur=args.blur,
                noise_std=args.noise_std,
                vignette=args.vignette,
            )

            cv2.imwrite(str(path), out)

    print("Done.")


if __name__ == "__main__":
    main()
