"""Run DCP dehazing on all generated or user-provided inputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from dcp_dehaze import DCPConfig, dehaze_array, load_rgb, save_gray, save_rgb
from generate_samples import generate


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUT = ROOT / "output"


def psnr(reference: np.ndarray, estimate: np.ndarray) -> float:
    mse = float(np.mean((reference - estimate) ** 2))
    if mse <= 1e-12:
        return 99.0
    return 10.0 * np.log10(1.0 / mse)


def mae(reference: np.ndarray, estimate: np.ndarray) -> float:
    return float(np.mean(np.abs(reference - estimate)))


def load_manifest() -> list[dict]:
    manifest_path = DATA / "scene_manifest.json"
    if not manifest_path.exists():
        generate()
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def label_image(image: Image.Image, label: str, target_size: tuple[int, int]) -> Image.Image:
    image = ImageOps.contain(image.convert("RGB"), target_size)
    canvas = Image.new("RGB", (target_size[0], target_size[1] + 34), "white")
    x = (target_size[0] - image.width) // 2
    canvas.paste(image, (x, 34))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    draw.text((10, 8), label, fill=(25, 25, 25), font=font)
    return canvas


def make_comparison(scene: dict, dehazed_path: Path, comparison_path: Path) -> None:
    clean = Image.open(ROOT / scene["clean"]).convert("RGB")
    hazy = Image.open(ROOT / scene["hazy"]).convert("RGB")
    dehazed = Image.open(dehazed_path).convert("RGB")

    panel_size = (360, 225)
    panels = [
        label_image(clean, "Clean reference", panel_size),
        label_image(hazy, "Before: hazy input", panel_size),
        label_image(dehazed, "After: DCP output", panel_size),
    ]
    gap = 16
    width = sum(p.width for p in panels) + gap * (len(panels) - 1)
    height = max(p.height for p in panels)
    canvas = Image.new("RGB", (width, height), "white")
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width + gap
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(comparison_path)


def run() -> None:
    manifest = load_manifest()
    (OUTPUT / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT / "comparisons").mkdir(parents=True, exist_ok=True)

    config = DCPConfig(patch_size=15, omega=0.95, top_percent=0.001, transmission_floor=0.10, guided_radius=35)
    results = []
    for scene in manifest:
        name = scene["name"]
        hazy_path = ROOT / scene["hazy"]
        clean_path = ROOT / scene["clean"]
        hazy = load_rgb(hazy_path)
        clean = load_rgb(clean_path)

        result = dehaze_array(hazy, config)
        dehazed_path = OUTPUT / "images" / f"{name}_dcp.png"
        dark_path = OUTPUT / "images" / f"{name}_dark_channel.png"
        transmission_path = OUTPUT / "images" / f"{name}_transmission.png"
        comparison_path = OUTPUT / "comparisons" / f"{name}_comparison.png"
        save_rgb(result["dehazed"], dehazed_path)
        save_gray(result["dark_channel"], dark_path)
        save_gray(result["transmission"], transmission_path)
        make_comparison(scene, dehazed_path, comparison_path)

        hazy_psnr = psnr(clean, hazy)
        dcp_psnr = psnr(clean, result["dehazed"])
        row = {
            "name": name,
            "title": scene["title"],
            "hazy_psnr": round(hazy_psnr, 2),
            "dcp_psnr": round(dcp_psnr, 2),
            "psnr_delta": round(dcp_psnr - hazy_psnr, 2),
            "hazy_mae": round(mae(clean, hazy), 4),
            "dcp_mae": round(mae(clean, result["dehazed"]), 4),
            "estimated_airlight": [round(float(v), 3) for v in result["airlight"]],
            "mean_transmission": round(float(np.mean(result["transmission"])), 3),
            "comparison": str(comparison_path.relative_to(ROOT)),
            "dehazed": str(dehazed_path.relative_to(ROOT)),
            "dark_channel": str(dark_path.relative_to(ROOT)),
            "transmission": str(transmission_path.relative_to(ROOT)),
            "expected": scene["expected"],
        }
        results.append(row)
        print(f"{name}: PSNR hazy={row['hazy_psnr']} dcp={row['dcp_psnr']} delta={row['psnr_delta']}")

    metrics_json = OUTPUT / "metrics.json"
    with metrics_json.open("w", encoding="utf-8") as handle:
        json.dump({"config": config.__dict__, "results": results}, handle, indent=2)

    metrics_csv = OUTPUT / "metrics.csv"
    with metrics_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "name",
                "hazy_psnr",
                "dcp_psnr",
                "psnr_delta",
                "hazy_mae",
                "dcp_mae",
                "estimated_airlight",
                "mean_transmission",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow({field: row[field] for field in writer.fieldnames})
    print(f"saved metrics to {metrics_json} and {metrics_csv}")


if __name__ == "__main__":
    run()
