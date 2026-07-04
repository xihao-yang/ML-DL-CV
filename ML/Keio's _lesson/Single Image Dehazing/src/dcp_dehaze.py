"""Dark Channel Prior single image dehazing.

This is a compact, dependency-light implementation of the method described in
He, Sun, and Tang's Dark Channel Prior paper. It uses Pillow for image I/O and
minimum filtering, plus a grayscale guided filter to refine the transmission
map.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


EPS = 1e-6


@dataclass
class DCPConfig:
    patch_size: int = 15
    omega: float = 0.95
    top_percent: float = 0.001
    transmission_floor: float = 0.10
    guided_radius: int = 40
    guided_eps: float = 1e-3


def load_rgb(path: str | Path) -> np.ndarray:
    """Load an image as float32 RGB in [0, 1]."""
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.float32) / 255.0


def save_rgb(array: np.ndarray, path: str | Path) -> None:
    """Save a float RGB image in [0, 1]."""
    output = np.clip(array * 255.0 + 0.5, 0, 255).astype(np.uint8)
    Image.fromarray(output, mode="RGB").save(path)


def save_gray(array: np.ndarray, path: str | Path) -> None:
    output = np.clip(array * 255.0 + 0.5, 0, 255).astype(np.uint8)
    Image.fromarray(output, mode="L").save(path)


def _ensure_odd(value: int) -> int:
    value = int(value)
    if value < 3:
        return 3
    return value if value % 2 == 1 else value + 1


def min_filter_2d(array: np.ndarray, size: int) -> np.ndarray:
    """Apply a 2-D minimum filter using Pillow's efficient MinFilter."""
    size = _ensure_odd(size)
    img = Image.fromarray(np.clip(array * 255.0 + 0.5, 0, 255).astype(np.uint8), mode="L")
    filtered = img.filter(ImageFilter.MinFilter(size=size))
    return np.asarray(filtered, dtype=np.float32) / 255.0


def dark_channel(image: np.ndarray, patch_size: int = 15) -> np.ndarray:
    """Compute the dark channel: local minimum over RGB channels and patch."""
    channel_min = np.min(image, axis=2)
    return min_filter_2d(channel_min, patch_size)


def estimate_atmospheric_light(
    image: np.ndarray, dark: np.ndarray, top_percent: float = 0.001
) -> np.ndarray:
    """Estimate global atmospheric light from the brightest dark-channel pixels."""
    flat_dark = dark.reshape(-1)
    flat_image = image.reshape(-1, 3)
    count = max(1, int(flat_dark.size * top_percent))
    candidate_indices = np.argpartition(flat_dark, -count)[-count:]
    candidates = flat_image[candidate_indices]
    brightest = np.argmax(np.sum(candidates, axis=1))
    airlight = candidates[brightest]
    return np.maximum(airlight, EPS)


def estimate_transmission(
    image: np.ndarray, airlight: np.ndarray, patch_size: int = 15, omega: float = 0.95
) -> np.ndarray:
    normalized = image / np.maximum(airlight.reshape(1, 1, 3), EPS)
    return 1.0 - omega * dark_channel(np.clip(normalized, 0.0, 1.0), patch_size)


def box_filter(array: np.ndarray, radius: int) -> np.ndarray:
    """Sum over a square window with boundary clipping."""
    radius = int(radius)
    height, width = array.shape
    integral = np.pad(array, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    ys = np.arange(height)
    xs = np.arange(width)
    y0 = np.maximum(ys - radius, 0)
    y1 = np.minimum(ys + radius + 1, height)
    x0 = np.maximum(xs - radius, 0)
    x1 = np.minimum(xs + radius + 1, width)
    return (
        integral[y1[:, None], x1[None, :]]
        - integral[y0[:, None], x1[None, :]]
        - integral[y1[:, None], x0[None, :]]
        + integral[y0[:, None], x0[None, :]]
    )


def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int = 40, eps: float = 1e-3) -> np.ndarray:
    """Single-channel guided filter for transmission refinement."""
    guide = guide.astype(np.float32)
    src = src.astype(np.float32)
    ones = np.ones_like(guide, dtype=np.float32)
    n = box_filter(ones, radius)

    mean_i = box_filter(guide, radius) / n
    mean_p = box_filter(src, radius) / n
    corr_i = box_filter(guide * guide, radius) / n
    corr_ip = box_filter(guide * src, radius) / n

    var_i = corr_i - mean_i * mean_i
    cov_ip = corr_ip - mean_i * mean_p

    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i

    mean_a = box_filter(a, radius) / n
    mean_b = box_filter(b, radius) / n
    return np.clip(mean_a * guide + mean_b, 0.0, 1.0)


def recover_radiance(
    image: np.ndarray, transmission: np.ndarray, airlight: np.ndarray, transmission_floor: float = 0.10
) -> np.ndarray:
    t = np.maximum(transmission, transmission_floor)[..., None]
    return np.clip((image - airlight.reshape(1, 1, 3)) / t + airlight.reshape(1, 1, 3), 0.0, 1.0)


def dehaze_array(image: np.ndarray, config: DCPConfig | None = None) -> dict[str, np.ndarray | dict]:
    """Run the full DCP pipeline and return all intermediate outputs."""
    config = config or DCPConfig()
    dark = dark_channel(image, config.patch_size)
    airlight = estimate_atmospheric_light(image, dark, config.top_percent)
    raw_t = estimate_transmission(image, airlight, config.patch_size, config.omega)
    guide = np.mean(image, axis=2)
    refined_t = guided_filter(guide, raw_t, config.guided_radius, config.guided_eps)
    dehazed = recover_radiance(image, refined_t, airlight, config.transmission_floor)
    return {
        "dehazed": dehazed,
        "dark_channel": dark,
        "transmission_raw": raw_t,
        "transmission": refined_t,
        "airlight": airlight,
        "config": asdict(config),
    }


def dehaze_file(
    input_path: str | Path,
    output_path: str | Path,
    config: DCPConfig | None = None,
    transmission_path: str | Path | None = None,
    dark_channel_path: str | Path | None = None,
) -> dict[str, np.ndarray | dict]:
    image = load_rgb(input_path)
    result = dehaze_array(image, config)
    save_rgb(result["dehazed"], output_path)
    if transmission_path:
        save_gray(result["transmission"], transmission_path)
    if dark_channel_path:
        save_gray(result["dark_channel"], dark_channel_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Dark Channel Prior image dehazing")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--patch-size", type=int, default=15)
    parser.add_argument("--omega", type=float, default=0.95)
    parser.add_argument("--guided-radius", type=int, default=40)
    parser.add_argument("--guided-eps", type=float, default=1e-3)
    parser.add_argument("--transmission", type=Path, default=None)
    parser.add_argument("--dark-channel", type=Path, default=None)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    config = DCPConfig(
        patch_size=args.patch_size,
        omega=args.omega,
        guided_radius=args.guided_radius,
        guided_eps=args.guided_eps,
    )
    result = dehaze_file(args.input, args.output, config, args.transmission, args.dark_channel)
    airlight = ", ".join(f"{value:.3f}" for value in result["airlight"])
    print(f"saved {args.output}")
    print(f"estimated atmospheric light A = [{airlight}]")


if __name__ == "__main__":
    main()
