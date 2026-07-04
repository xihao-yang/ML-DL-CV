"""Build the final PDF report for the dehazing exercise."""

from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"
PDF_PATH = OUTPUT / "pdf" / "single_image_dehazing_report.pdf"


def load_metrics() -> dict:
    metrics_path = OUTPUT / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError("Run src/run_experiment.py before building the report.")
    with metrics_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def image_flowable(path: Path, max_width: float, max_height: float | None = None) -> Image:
    img = Image(str(path))
    width, height = img.drawWidth, img.drawHeight
    scale = max_width / width
    if max_height is not None:
        scale = min(scale, max_height / height)
    img.drawWidth = width * scale
    img.drawHeight = height * scale
    return img


def build() -> Path:
    OUTPUT.joinpath("pdf").mkdir(parents=True, exist_ok=True)
    metrics = load_metrics()
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
        title="Single Image Dehazing Report",
    )
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="Small",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8.4,
            leading=10.3,
            spaceAfter=5,
        )
    )
    styles.add(
        ParagraphStyle(
            name="TightBody",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.2,
            leading=11.4,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Heading",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=15,
            spaceBefore=8,
            spaceAfter=6,
        )
    )

    story = []
    story.append(Paragraph("Computational Photography: Single Image Dehazing", styles["Title"]))
    story.append(Paragraph("Baseline: Dark Channel Prior (DCP). Deadline: July 7, 2026.", styles["TightBody"]))
    story.append(
        Paragraph(
            "This report implements and evaluates the Dark Channel Prior algorithm on four image categories: "
            "landscape, cityscape, indoor, and a deliberately difficult bright-white scene. Because no input "
            "photographs were present in the workspace, the test images are deterministic synthetic scenes with "
            "known clean references and depth maps. The same code can be run on real photos by placing images in "
            "data/input and adapting the manifest.",
            styles["TightBody"],
        )
    )

    story.append(Paragraph("1. Baseline Algorithm", styles["Heading"]))
    story.append(
        Paragraph(
            "DCP assumes the haze model I(x) = J(x)t(x) + A(1 - t(x)), where I is the observed hazy image, "
            "J is the recovered scene radiance, A is global atmospheric light, and t is the transmission. "
            "For most non-sky natural patches, at least one color channel contains very low intensity in the "
            "haze-free image. This prior estimates transmission as t(x) = 1 - omega * dark(I/A), then refines "
            "it with a guided filter before recovering J. The implementation saves the dehazed result, dark "
            "channel, and refined transmission map for every input.",
            styles["TightBody"],
        )
    )
    config = metrics["config"]
    story.append(
        Paragraph(
            f"Parameters used: patch_size={config['patch_size']}, omega={config['omega']}, "
            f"transmission_floor={config['transmission_floor']}, guided_radius={config['guided_radius']}.",
            styles["Small"],
        )
    )

    story.append(Paragraph("2. Output Result Comparisons", styles["Heading"]))
    story.append(
        Paragraph(
            "Each row below shows clean reference, hazy input, and DCP output. The clean image is included for "
            "quantitative evaluation; the required before-vs-after comparison is the middle and right panel.",
            styles["TightBody"],
        )
    )
    for result in metrics["results"]:
        story.append(Paragraph(result["title"], styles["Heading3"]))
        story.append(image_flowable(ROOT / result["comparison"], doc.width, max_height=2.0 * inch))
        story.append(
            Paragraph(
                f"PSNR before: {result['hazy_psnr']} dB; PSNR after DCP: {result['dcp_psnr']} dB; "
                f"delta: {result['psnr_delta']} dB. Estimated A: {result['estimated_airlight']}.",
                styles["Small"],
            )
        )

    story.append(PageBreak())
    story.append(Paragraph("3. Quantitative Summary", styles["Heading"]))
    table_data = [["Scene", "Before PSNR", "After PSNR", "Delta", "Before MAE", "After MAE"]]
    for result in metrics["results"]:
        table_data.append(
            [
                result["name"],
                f"{result['hazy_psnr']:.2f}",
                f"{result['dcp_psnr']:.2f}",
                f"{result['psnr_delta']:.2f}",
                f"{result['hazy_mae']:.4f}",
                f"{result['dcp_mae']:.4f}",
            ]
        )
    table = Table(table_data, colWidths=[1.55 * inch, 1.0 * inch, 1.0 * inch, 0.75 * inch, 1.0 * inch, 1.0 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#263238")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.6),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B0BEC5")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F7F8")]),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 0.12 * inch))

    story.append(Paragraph("4. Technical Discussion: When DCP Works and Fails", styles["Heading"]))
    story.append(
        Paragraph(
            "Landscape and cityscape scenes usually work well because vegetation, roads, windows, building edges, "
            "and shadows create low values in at least one RGB channel. Those pixels make the dark channel close "
            "to zero in haze-free patches, so the DCP transmission estimate is meaningful. The method recovers "
            "contrast and color most clearly in those non-sky regions.",
            styles["TightBody"],
        )
    )
    story.append(
        Paragraph(
            "Sky, snow, white clouds, white walls, and other high-albedo objects are the main failure cases. "
            "They naturally have high intensity in all color channels even without haze, so the dark-channel "
            "assumption is false. DCP can then confuse true object brightness with airlight, underestimate "
            "transmission, and over-darken or color-shift the restored result. A bright object may also be "
            "selected as atmospheric light A, which further biases the recovery.",
            styles["TightBody"],
        )
    )
    story.append(
        Paragraph(
            "Indoor scenes are also unstable because the physical model assumes outdoor participating media with "
            "a mostly global atmospheric light. Interior scenes can contain local artificial lights, reflective "
            "surfaces, flat walls, and weak depth-haze correlation. The guided filter reduces block artifacts, "
            "but it cannot repair an incorrect prior; near depth discontinuities it may still leave halos.",
            styles["TightBody"],
        )
    )

    story.append(Paragraph("5. Modern Approaches (Bonus Discussion)", styles["Heading"]))
    story.append(
        Paragraph(
            "Recent learning-based dehazing methods, such as DehazeFormer and later zero-shot Gaussian-based "
            "approaches, replace the hand-crafted dark-channel assumption with learned or optimized image priors. "
            "They tend to handle sky and bright regions better because they can learn semantic and multi-scale "
            "context rather than relying on a local minimum color channel. Their drawbacks are dependence on "
            "training data or pretrained weights, higher compute cost, possible domain shift, and reduced "
            "interpretability. In this local submission I did not execute a pretrained SOTA model because the "
            "environment has no PyTorch package, no downloaded weights, and no GPU; the bonus comparison is "
            "therefore a literature-level technical comparison, while all DCP results are locally reproducible.",
            styles["TightBody"],
        )
    )

    story.append(Paragraph("6. Reproducibility and Files", styles["Heading"]))
    story.append(
        Paragraph(
            "Run order: src/generate_samples.py creates clean/hazy/depth inputs; src/run_experiment.py runs DCP "
            "and writes output/metrics.json plus comparison images; src/build_report.py builds this PDF. "
            "The key deliverables are output/pdf/single_image_dehazing_report.pdf and output/comparisons/*.png.",
            styles["TightBody"],
        )
    )

    story.append(Paragraph("References", styles["Heading"]))
    refs = [
        "K. He, J. Sun, and X. Tang. Single Image Haze Removal Using Dark Channel Prior. CVPR 2009 / TPAMI 2011.",
        "Reference implementation repository: https://github.com/He-Zhang/image_dehaze",
        "DehazeFormer: Vision Transformer for Single Image Dehazing. https://arxiv.org/abs/2204.03883",
        "Dehaze-Gaussian: Single Image Dehazing via Physical Model-based Gaussian Splatting. https://arxiv.org/abs/2606.16163",
        "Papers with Code image dehazing task page: https://paperswithcode.com/task/image-dehazing",
    ]
    for ref in refs:
        story.append(Paragraph(ref, styles["Small"]))

    def page_footer(canvas, document):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#607D8B"))
        canvas.drawString(document.leftMargin, 0.32 * inch, "Single Image Dehazing Report")
        canvas.drawRightString(A4[0] - document.rightMargin, 0.32 * inch, f"Page {document.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=page_footer, onLaterPages=page_footer)
    print(f"saved {PDF_PATH}")
    return PDF_PATH


if __name__ == "__main__":
    build()
