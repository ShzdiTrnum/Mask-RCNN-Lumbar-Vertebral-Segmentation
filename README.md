# Mask-RCNN-Vertebral

> Standard Mask R-CNN (ResNet-101-FPN) baseline for automated lumbar vertebral body instance segmentation in sagittal T2 MRI images. This repository is the **ablation baseline** for the BMask R-CNN paper — identical hyperparameters, same SPIDER dataset, same training schedule — with the **sole difference** being the absence of the boundary supervision head. Achieves AP50 = 98.07% and AP75 = 98.07% on the curated SPIDER dataset.

> ⚠️ **Associated manuscript under review** — please do not redistribute or cite without permission from the authors.

---

## Relationship to BMask R-CNN

This repository implements the **standard Mask R-CNN baseline** used as the ablation comparison in:

> *Boundary-Aware Instance Segmentation of Lumbar Vertebrae in MRI for Automated Deformity Measurement*
> Turrnum Shahzadi, Norio Tagawa, Shuhei Tarashima, Muhammad Usman Ali, Rizwan Akram, Zeeshan Talib
> Tokyo Metropolitan University et al. *(Manuscript under review)*

| | Mask R-CNN (This repo) | BMask R-CNN |
|--|--|--|
| **Repository** | `Mask-RCNN-Vertebral` | `BMask-RCNN-Vertebral` |
| **Backbone** | ResNet-101 + FPN | ResNet-101 + FPN |
| **Mask head** | 6-conv standard | 6-conv standard |
| **Boundary head** | ❌ None | ✅ 4-conv, loss weight 1.8 |
| **Config source** | detectron2 model zoo | BMask R-CNN YAML |
| **AP50** | 98.07% | **98.41%** |
| **AP75** | 98.07% | **98.41% (+0.34 pp)** |

The +0.34 pp AP75 improvement from BMask R-CNN over this baseline confirms that explicit boundary supervision provides quantifiable improvement in contour precision — the metric most sensitive to boundary accuracy and the one most directly relevant to LLA/LSA angle estimation.

---

## Table of Contents

- [Key Results](#key-results)
- [Repository Structure](#repository-structure)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Configuration Parameters](#configuration-parameters)
- [Architecture Difference](#architecture-difference)
- [Outputs](#outputs)
- [Known Limitations](#known-limitations)
- [Related Repositories](#related-repositories)
- [References](#references)
- [License](#license)

---

## Key Results

| Metric | Value |
|--------|-------|
| Segmentation AP50 | 98.07% |
| Segmentation AP75 | 98.07% |
| AP75 vs BMask R-CNN | −0.34 pp |
| Dataset | SPIDER (505 sagittal T2 MRI images, 6 classes: L1–S1) |
| Backbone | ResNet-101 + FPN |
| Boundary head | **None** |

---

## Repository Structure

```
mask-rcnn-vertebral/
│
├── src/
│   ├── __init__.py
│   ├── setup_environment.py   # Install detectron2 (no BMask R-CNN needed)
│   ├── dataset.py             # COCO registration, class balance, oversampling
│   ├── trainer.py             # MaskRCNNTrainer, BestCheckpointer, build_cfg()
│   ├── train.py               # Main training entry point (importable + CLI)
│   ├── metrics.py             # Loss and AP plots from metrics.json
│   └── inference.py           # Predictor, COCO eval, mask saving, angles, ICC
│
├── configs/
│   └── notes.md               # Config parameter notes and ablation rationale
│
├── data/
│   └── README.md              # Dataset format instructions
│
├── docs/
│   └── ablation_notes.md      # Detailed comparison with BMask R-CNN
│
├── outputs/
│   └── README.md              # Output file descriptions
│
├── assets/                    # Sample result images
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| Python | ≥ 3.8 | Runtime |
| PyTorch | ≥ 1.10 + CUDA | Training and inference |
| detectron2 | latest (GitHub) | Base framework — standard install only, no BMask R-CNN |
| opencv-python | ≥ 4.5 | Image I/O, contour processing |
| numpy | ≥ 1.21 | Array operations |
| matplotlib | ≥ 3.4 | Metric plots |
| scipy | ≥ 1.7 | ICC(2,1) F-distribution |
| Pillow | < 10.0 | Detectron2 compatibility |

**Note:** This repo does NOT require BMask R-CNN or its custom YAML config. Only the standard `detectron2` package is needed.

---

## Installation

### Google Colab (recommended — GPU required)

```python
from google.colab import drive
drive.mount('/content/drive')
!python src/setup_environment.py
```

```python
import sys
sys.path.insert(0, "/content")   # so 'from src.xxx import' works
```

### Local (Linux, GPU required)

```bash
git clone https://github.com/ShzdiTrnum/Mask-RCNN-Vertebral.git
cd Mask-RCNN-Vertebral
pip install -r requirements.txt
pip install 'git+https://github.com/facebookresearch/detectron2.git'
wget https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-101.pkl
```

---

## Dataset

Same curated SPIDER dataset as the BMask R-CNN repository:

- **Source:** SPIDER — Lumbar Spine Segmentation in MR Images (van der Graaf et al., 2023)
- **Images:** 505 sagittal T2 MRI images
- **Classes:** L1 (V1), L2 (V2), L3 (V3), L4 (V4), L5 (V5), S1
- **Curation:** S1 annotations added by skilled clinicians (originally missing from SPIDER)
- **Format:** COCO instance segmentation JSON

See `data/README.md` for folder structure and filename conventions.

---

## Usage

### Full pipeline (Colab)

```python
# 1. Training
from src.train import run_training
cfg = run_training(
    base_path  = "/content/drive/MyDrive/newDataset_Aug",
    output_dir = "/content/drive/MyDrive/MaskRCNN_Baseline",
)

# 2. Load model and evaluate
from src.inference import load_predictor, evaluate_test_set
predictor, infer_cfg = load_predictor(cfg)
evaluate_test_set(predictor, infer_cfg)

# 3. Save predictions
from src.inference import save_predictions
save_predictions(predictor, infer_cfg)

# 4. Compute angles
from src.inference import run_angle_measurements
auto_LLA, auto_LSA, filenames = run_angle_measurements(predictor, infer_cfg)

# 5. ICC analysis (supply surgeon lists from BMask R-CNN statistics module)
from src.inference import run_icc_analysis
run_icc_analysis(auto_LLA, auto_LSA, infer_cfg.OUTPUT_DIR,
                 paired_s1_LLA=..., paired_s2_LLA=...,
                 paired_s1_LSA=..., paired_s2_LSA=...)
```

### Training only (CLI)

```bash
python src/train.py \
    --base_path  /content/drive/MyDrive/newDataset_Aug \
    --output_dir /content/drive/MyDrive/MaskRCNN_Baseline
```

---

## Configuration Parameters

All parameters are **identical to the BMask R-CNN configuration** to ensure a fair ablation:

| Parameter | Value | Shared with BMask R-CNN? |
|-----------|-------|--------------------------|
| Backbone | ResNet-101 + FPN | ✅ Same |
| `NUM_CLASSES` | 7 | ✅ Same |
| `ROI_MASK_HEAD.NUM_CONV` | 6 | ✅ Same |
| `ROI_MASK_HEAD.LOSS_WEIGHT` | 1.0 | ✅ Same |
| `BOUNDARY_MASK_HEAD` | **Not present** | ❌ Key difference |
| `BASE_LR` | 0.001 | ✅ Same |
| `MAX_ITER` | 30,000 | ✅ Same |
| `SOLVER.STEPS` | (22000, 25000) | ✅ Same |
| `WARMUP_ITERS` | 1,000 | ✅ Same |
| `LR_SCHEDULER_NAME` | WarmupCosineLR | ✅ Same |
| `IMS_PER_BATCH` | 2 | ✅ Same |
| `INPUT.MIN_SIZE_TRAIN` | (800, 896, 1024) | ✅ Same |
| Config source | detectron2 model zoo YAML | ❌ BMask R-CNN uses custom YAML |

---

## Architecture Difference

The **only** architectural difference between this baseline and BMask R-CNN is the absence of the boundary supervision branch:

```
Standard Mask R-CNN (this repo):
  Input → Backbone (R-101+FPN) → RPN → ROI Align → Classification
                                                  → BBox regression
                                                  → Mask head (6 conv)
                                                         ↓
                                                    mask_loss

BMask R-CNN (companion repo):
  Input → Backbone (R-101+FPN) → RPN → ROI Align → Classification
                                                  → BBox regression
                                                  → Mask head (6 conv)
                                                         ↓
                                                    mask_loss
                                                  → Boundary head (4 conv)
                                                         ↓
                                                  boundary_loss × 1.8
```

The boundary head receives the same ROI features and is trained on edge-derived ground truth (binary boundary maps from vertebral contours). The combined loss `mask_loss + 1.8 × boundary_loss` pushes the model to produce sharper endplate edges, improving AP75 by +0.34 pp.

---

## Outputs

All outputs saved to `cfg.OUTPUT_DIR`:

| File / Folder | Description |
|---------------|-------------|
| `model_best.pth` | Best checkpoint by validation segm/AP |
| `model_final.pth` | Final checkpoint |
| `metrics.json` | Per-iteration training metrics |
| `loss_curves.png` | Loss components (no boundary loss curve) |
| `segm_ap.png` | Segmentation AP / AP50 / AP75 |
| `bbox_ap.png` | BBox AP / AP50 / AP75 |
| `ap50_vs_ap75.png` | AP comparison |
| `test_results/maskrcnn_results.json` | COCO evaluation |
| `test_predictions/overlay/` | Colour overlays |
| `test_predictions/binary_masks/` | Binary + instance masks |
| `test_predictions/per_class_masks/` | Per-vertebra masks |
| `maskrcnn_angle_measurements.json` | LLA + LSA per image |
| `maskrcnn_ICC_results.json` | ICC(2,1) results |

---

## Known Limitations

- Paths are configured for Google Colab/Drive — update for local use
- Does not include Bland-Altman plots — use the BMask R-CNN `statistics.py` module with this model's angle outputs for full comparison
- No boundary supervision means AP75 is lower than BMask R-CNN, particularly for S1 (most oblique vertebra)

---

## Related Repositories

| Repo | Role |
|------|------|
| [Fasciculation-Simulation](https://github.com/ShzdiTrnum/Fasciculation-Simulation) | Synthetic video generation |
| [Optical-Flow](https://github.com/ShzdiTrnum/Optical-Flow) | Optical flow rotation estimation |
| [Fasciculation-LK-Synthetic](https://github.com/ShzdiTrnum/Fasciculation-LK-Synthetic) | LK affine motion detection |
| [Fasciculation-SVD-Optical-Flow](https://github.com/ShzdiTrnum/Fasciculation-SVD-Optical-Flow) | SVD + optical flow detection |
| [Fasciculation-Detection-Synthetic](https://github.com/ShzdiTrnum/Fasciculation-Detection-Synthetic) | BBVI on synthetic data |
| [Fasciculation-Detection-ALS](https://github.com/ShzdiTrnum/Fasciculation-Detection-ALS) | BBVI on real ALS data |
| [**BMask-RCNN-Vertebral**](https://github.com/ShzdiTrnum/BMask-RCNN-Vertebral) | ← Boundary-preserving model (main contribution) |
| **This repo** | ← Standard Mask R-CNN baseline |

---

## References

1. He, K., Gkioxari, G., Dollár, P., Girshick, R. (2017). *Mask R-CNN*. ICCV.
2. Cheng, T., Wang, X., Huang, L., Liu, W. (2020). *Boundary-preserving Mask R-CNN*. ECCV. [GitHub](https://github.com/hustvl/BMaskR-CNN)
3. Wu, Y. et al. (2019). *Detectron2*. [GitHub](https://github.com/facebookresearch/detectron2)
4. van der Graaf, J. et al. (2023). *SPIDER — Lumbar Spine Segmentation in MR Images*. Zenodo. DOI: 10.5281/zenodo.10159290
5. Kim, Y.-T. et al. (2023). *Automatic spine segmentation and parameter measurement*. Journal of Digital Imaging, 36(4), 1447–1459.

---

## License

MIT License. See `LICENSE` for details.

The associated manuscript is under review — please do not redistribute or cite without permission:
- Turrnum Shahzadi: turrnumshahzadi@gmail.com
- Norio Tagawa: tagawa@tmu.ac.jp
