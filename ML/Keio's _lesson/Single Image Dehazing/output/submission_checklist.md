# Submission Checklist

- Baseline DCP implemented: yes, see `src/dcp_dehaze.py`.
- Baseline executed: yes, see `output/images/*_dcp.png`.
- Multiple input categories tested: yes, landscape, cityscape, indoor, and bright-white failure case.
- Before-vs-after comparisons included: yes, see `output/comparisons/*_comparison.png` and the PDF report.
- Technical failure discussion included: yes, PDF section 4 explains sky, snow, white objects, indoor lighting, and prior/model violations.
- Modern approach discussion included: yes, PDF section 5 compares DCP with learning-based / recent methods at literature level.
- Full PDF report generated: yes, `output/pdf/single_image_dehazing_report.pdf`.
- PDF rendered and visually checked: yes, rendered pages are in `tmp/pdfs/dehaze_report-*.png`.

Note: the optional SOTA model was not executed locally because this environment has no PyTorch package, pretrained dehazing weights, or GPU. The required baseline and comparative failure analysis are fully implemented and reproducible.
