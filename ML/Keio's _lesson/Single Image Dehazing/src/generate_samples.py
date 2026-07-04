"""Generate controlled sample scenes for the dehazing exercise.

The assignment asks for multiple scene types. The workspace did not contain
input photographs, so this script creates deterministic synthetic scenes with
known clean images and depth maps. That lets the experiment measure PSNR before
and after dehazing while still covering the requested categories.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SIZE = (640, 400)


def vertical_gradient(width: int, height: int, top: tuple[int, int, int], bottom: tuple[int, int, int]) -> Image.Image:
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    top_arr = np.array(top, dtype=np.float32)
    bottom_arr = np.array(bottom, dtype=np.float32)
    for y in range(height):
        ratio = y / max(1, height - 1)
        arr[y, :, :] = (top_arr * (1 - ratio) + bottom_arr * ratio).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def depth_gradient(width: int, height: int, near: float = 0.15, far: float = 1.0) -> Image.Image:
    arr = np.zeros((height, width), dtype=np.uint8)
    for y in range(height):
        ratio = y / max(1, height - 1)
        depth = far * (1 - ratio) + near * ratio
        arr[y, :] = int(np.clip(depth, 0, 1) * 255)
    return Image.fromarray(arr, mode="L")


def apply_haze(clean: Image.Image, depth: Image.Image, beta: float, airlight: tuple[float, float, float]) -> Image.Image:
    clean_arr = np.asarray(clean, dtype=np.float32) / 255.0
    depth_arr = np.asarray(depth, dtype=np.float32) / 255.0
    t = np.exp(-beta * depth_arr)[..., None]
    a = np.array(airlight, dtype=np.float32).reshape(1, 1, 3)
    hazy = clean_arr * t + a * (1.0 - t)
    return Image.fromarray(np.clip(hazy * 255.0 + 0.5, 0, 255).astype(np.uint8), mode="RGB")


def draw_landscape() -> tuple[Image.Image, Image.Image]:
    w, h = SIZE
    img = vertical_gradient(w, h, (122, 180, 232), (226, 240, 250))
    depth = depth_gradient(w, h, near=0.12, far=1.0)
    draw = ImageDraw.Draw(img)
    ddraw = ImageDraw.Draw(depth)

    # Far mountain layers.
    draw.polygon([(0, 235), (90, 105), (185, 225), (290, 110), (435, 235)], fill=(83, 111, 123))
    draw.polygon([(220, 245), (385, 95), (520, 245), (640, 135), (640, 255)], fill=(66, 94, 113))
    ddraw.polygon([(0, 235), (90, 105), (185, 225), (290, 110), (435, 235)], fill=205)
    ddraw.polygon([(220, 245), (385, 95), (520, 245), (640, 135), (640, 255)], fill=220)

    # Mid forest and lake.
    draw.rectangle([0, 230, w, 300], fill=(63, 126, 86))
    ddraw.rectangle([0, 230, w, 300], fill=115)
    for x in range(10, w, 32):
        height = 28 + (x * 7) % 30
        draw.polygon([(x, 255), (x + 15, 255 - height), (x + 30, 255)], fill=(28, 79, 55))
        ddraw.polygon([(x, 255), (x + 15, 255 - height), (x + 30, 255)], fill=92)
    draw.rectangle([0, 300, w, h], fill=(49, 104, 135))
    ddraw.rectangle([0, 300, w, h], fill=65)
    draw.polygon([(0, 350), (120, 310), (260, 400), (0, 400)], fill=(40, 84, 59))
    ddraw.polygon([(0, 350), (120, 310), (260, 400), (0, 400)], fill=38)
    return img, depth


def draw_cityscape() -> tuple[Image.Image, Image.Image]:
    w, h = SIZE
    img = vertical_gradient(w, h, (135, 184, 225), (235, 238, 232))
    depth = depth_gradient(w, h, near=0.20, far=1.0)
    draw = ImageDraw.Draw(img)
    ddraw = ImageDraw.Draw(depth)

    buildings = [
        (30, 145, 105, 320, (73, 84, 95), 170),
        (125, 95, 205, 320, (92, 101, 110), 205),
        (230, 130, 320, 320, (69, 82, 94), 180),
        (345, 75, 440, 320, (98, 104, 111), 225),
        (465, 120, 585, 320, (80, 88, 98), 190),
    ]
    for x0, y0, x1, y1, color, dep in buildings:
        draw.rectangle([x0, y0, x1, y1], fill=color)
        ddraw.rectangle([x0, y0, x1, y1], fill=dep)
        for wx in range(x0 + 12, x1 - 10, 22):
            for wy in range(y0 + 15, y1 - 18, 30):
                window = (203, 214, 218) if (wx + wy) % 3 else (45, 58, 68)
                draw.rectangle([wx, wy, wx + 9, wy + 14], fill=window)

    draw.polygon([(0, h), (260, 315), (380, 315), (640, h)], fill=(55, 58, 62))
    ddraw.polygon([(0, h), (260, 315), (380, 315), (640, h)], fill=60)
    draw.line([(320, 330), (320, 400)], fill=(240, 225, 80), width=4)
    draw.rectangle([0, 318, 640, 335], fill=(45, 94, 69))
    ddraw.rectangle([0, 318, 640, 335], fill=85)
    return img, depth


def draw_indoor() -> tuple[Image.Image, Image.Image]:
    w, h = SIZE
    img = vertical_gradient(w, h, (205, 210, 205), (175, 160, 145))
    depth = depth_gradient(w, h, near=0.12, far=0.65)
    draw = ImageDraw.Draw(img)
    ddraw = ImageDraw.Draw(depth)

    draw.rectangle([0, 250, w, h], fill=(136, 105, 78))
    ddraw.rectangle([0, 250, w, h], fill=45)
    draw.polygon([(0, 250), (90, 190), (550, 190), (640, 250)], fill=(186, 181, 169))
    ddraw.polygon([(0, 250), (90, 190), (550, 190), (640, 250)], fill=130)

    draw.rectangle([430, 70, 590, 210], fill=(225, 238, 246), outline=(105, 118, 128), width=5)
    ddraw.rectangle([430, 70, 590, 210], fill=220)
    draw.line([(510, 70), (510, 210)], fill=(105, 118, 128), width=4)
    draw.line([(430, 140), (590, 140)], fill=(105, 118, 128), width=4)

    draw.rectangle([70, 150, 245, 250], fill=(87, 65, 52))
    ddraw.rectangle([70, 150, 245, 250], fill=85)
    for y in [172, 204, 236]:
        draw.line([(80, y), (235, y)], fill=(45, 35, 30), width=4)
    for x, col in [(90, (60, 85, 120)), (125, (120, 70, 58)), (160, (45, 105, 75)), (197, (130, 122, 60))]:
        draw.rectangle([x, 158, x + 20, 235], fill=col)

    draw.rectangle([300, 210, 410, 285], fill=(52, 83, 112))
    ddraw.rectangle([300, 210, 410, 285], fill=70)
    draw.rectangle([285, 285, 425, 305], fill=(47, 48, 50))
    ddraw.rectangle([285, 285, 425, 305], fill=35)
    return img, depth


def draw_bright_regions() -> tuple[Image.Image, Image.Image]:
    w, h = SIZE
    img = vertical_gradient(w, h, (220, 234, 246), (248, 250, 248))
    depth = depth_gradient(w, h, near=0.12, far=1.0)
    draw = ImageDraw.Draw(img)
    ddraw = ImageDraw.Draw(depth)

    # Large high-albedo areas deliberately violate the DCP assumption.
    draw.ellipse([55, 55, 245, 130], fill=(248, 248, 244))
    draw.ellipse([210, 75, 430, 155], fill=(246, 248, 248))
    draw.ellipse([385, 45, 610, 125], fill=(250, 250, 247))
    ddraw.ellipse([55, 55, 245, 130], fill=240)
    ddraw.ellipse([210, 75, 430, 155], fill=240)
    ddraw.ellipse([385, 45, 610, 125], fill=240)

    draw.rectangle([0, 235, w, h], fill=(238, 240, 236))
    ddraw.rectangle([0, 235, w, h], fill=90)
    draw.polygon([(0, 280), (140, 220), (290, 300), (430, 225), (640, 300), (640, 400), (0, 400)], fill=(248, 248, 245))
    ddraw.polygon([(0, 280), (140, 220), (290, 300), (430, 225), (640, 300), (640, 400), (0, 400)], fill=120)
    draw.rectangle([250, 165, 390, 255], fill=(236, 236, 230), outline=(200, 205, 205), width=3)
    ddraw.rectangle([250, 165, 390, 255], fill=160)
    draw.rectangle([275, 190, 315, 232], fill=(215, 228, 235))
    draw.rectangle([330, 190, 370, 232], fill=(215, 228, 235))
    return img, depth


SCENES = [
    {
        "name": "landscape",
        "title": "Landscape with mountains and vegetation",
        "builder": draw_landscape,
        "beta": 1.45,
        "airlight": (0.92, 0.95, 1.00),
        "expected": "Works well because vegetation and shadows provide dark pixels inside most local patches.",
    },
    {
        "name": "cityscape",
        "title": "Cityscape with buildings and road",
        "builder": draw_cityscape,
        "beta": 1.35,
        "airlight": (0.91, 0.94, 0.98),
        "expected": "Works reasonably well on buildings and road; sky remains a weak region.",
    },
    {
        "name": "indoor",
        "title": "Indoor room with window and furniture",
        "builder": draw_indoor,
        "beta": 1.05,
        "airlight": (0.94, 0.93, 0.90),
        "expected": "Mixed result because indoor haze and lighting do not match the outdoor atmospheric model.",
    },
    {
        "name": "bright_regions",
        "title": "Scene dominated by white clouds, snow, and bright walls",
        "builder": draw_bright_regions,
        "beta": 1.55,
        "airlight": (0.96, 0.98, 1.00),
        "expected": "Failure case because white regions naturally have no dark channel even when haze-free.",
    },
]


def generate() -> None:
    for folder in ["clean", "input", "depth"]:
        (DATA / folder).mkdir(parents=True, exist_ok=True)

    manifest = []
    for scene in SCENES:
        clean, depth = scene["builder"]()
        hazy = apply_haze(clean, depth, scene["beta"], scene["airlight"])
        clean_path = DATA / "clean" / f"{scene['name']}_clean.png"
        hazy_path = DATA / "input" / f"{scene['name']}_hazy.png"
        depth_path = DATA / "depth" / f"{scene['name']}_depth.png"
        clean.save(clean_path)
        hazy.save(hazy_path)
        depth.save(depth_path)
        manifest.append(
            {
                "name": scene["name"],
                "title": scene["title"],
                "clean": str(clean_path.relative_to(ROOT)),
                "hazy": str(hazy_path.relative_to(ROOT)),
                "depth": str(depth_path.relative_to(ROOT)),
                "beta": scene["beta"],
                "airlight": scene["airlight"],
                "expected": scene["expected"],
            }
        )

    with (DATA / "scene_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"generated {len(manifest)} scenes in {DATA}")


if __name__ == "__main__":
    generate()
