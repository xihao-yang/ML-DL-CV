# Single Image Dehazing Exercise

This workspace contains a reproducible submission package for the computational photography exercise.

## Contents

- `src/dcp_dehaze.py` - Dark Channel Prior implementation with guided-filter transmission refinement.
- `src/generate_samples.py` - Creates four controlled scene categories: landscape, cityscape, indoor, and bright-white failure case.
- `src/run_experiment.py` - Runs DCP, saves outputs, creates before/after comparison panels, and computes PSNR/MAE.
- `src/build_report.py` - Builds the PDF report required for submission.
- `output/pdf/single_image_dehazing_report.pdf` - Final report after running the pipeline.

## Run

Use the bundled Python executable in this Codex environment:

```powershell
& 'C:\Users\xih_y\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' src/generate_samples.py
& 'C:\Users\xih_y\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' src/run_experiment.py
& 'C:\Users\xih_y\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' src/build_report.py
```

To run the baseline on a single image:

```powershell
& 'C:\Users\xih_y\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' src/dcp_dehaze.py data/input/landscape_hazy.png output/images/landscape_manual_dcp.png
```

## Notes

The workspace initially contained no input photos, so deterministic synthetic scenes are used. They preserve known clean references and depth maps, which makes the evaluation measurable and repeatable. Replace or extend `data/input` and `data/scene_manifest.json` if real images are provided later.
