"""
inference.py
============
Model evaluation, inference, angle measurement, and ICC(2,1) analysis
for the Mask R-CNN vertebral segmentation baseline.

Functions:
  - load_predictor()           : load best or final checkpoint
  - evaluate_test_set()        : COCO evaluation on test split
  - save_predictions()         : binary masks, instance maps, colour overlays
  - run_angle_measurements()   : batch LLA / LSA computation for all test images
  - compute_icc21()            : ICC(2,1) with 95% confidence intervals
  - run_icc_analysis()         : full ICC table vs surgeon measurements

Usage:
    from src.inference import load_predictor, evaluate_test_set, run_icc_analysis
    predictor, infer_cfg = load_predictor(cfg)
    evaluate_test_set(predictor, infer_cfg)
    auto_LLA, auto_LSA, filenames = run_angle_measurements(predictor, infer_cfg)
    run_icc_analysis(auto_LLA, auto_LSA, infer_cfg.OUTPUT_DIR)
"""

import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy import stats as sp_stats

from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import inference_on_dataset
from detectron2.utils.visualizer import ColorMode, Visualizer

from src.trainer import build_coco_evaluator


# ─────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────

def load_predictor(cfg, score_thresh: float = 0.3):
    """
    Load model_best.pth if available, else model_final.pth.

    Returns:
        (DefaultPredictor, infer_cfg)
    """
    best_w  = os.path.join(cfg.OUTPUT_DIR, "model_best.pth")
    final_w = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

    if os.path.exists(best_w):
        weights = best_w
        print("Loading: model_best.pth")
    elif os.path.exists(final_w):
        weights = final_w
        print("Loading: model_final.pth  (model_best.pth not found)")
    else:
        raise FileNotFoundError(
            f"No weights found in {cfg.OUTPUT_DIR}. "
            "Complete training first.")

    infer_cfg = cfg.clone()
    infer_cfg.defrost()
    infer_cfg.MODEL.WEIGHTS                     = weights
    infer_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    infer_cfg.freeze()

    predictor = DefaultPredictor(infer_cfg)
    meta       = MetadataCatalog.get("my_dataset_test_8")
    print(f"Predictor ready. Classes: {meta.thing_classes}")
    return predictor, infer_cfg


# ─────────────────────────────────────────────────────────────────────
# COCO evaluation
# ─────────────────────────────────────────────────────────────────────

def evaluate_test_set(predictor, infer_cfg):
    """
    Run COCO evaluation on the test set and print AP metrics.

    Returns:
        Dict of evaluation results.
    """
    test_eval_dir = os.path.join(infer_cfg.OUTPUT_DIR, "test_results")
    evaluator     = build_coco_evaluator(
        "my_dataset_test_8", infer_cfg, test_eval_dir)
    loader = build_detection_test_loader(infer_cfg, "my_dataset_test_8")

    with torch.no_grad():
        results = inference_on_dataset(predictor.model, loader, evaluator)

    print("\n── Mask R-CNN Test Set Results ───────────────────────")
    for task in ["bbox", "segm"]:
        if task in results:
            r = results[task]
            print(f"  {task.upper()}")
            print(f"    AP  (0.50:0.95) : {r.get('AP',   0):.2f}%")
            print(f"    AP50            : {r.get('AP50', 0):.2f}%")
            print(f"    AP75 [primary]  : {r.get('AP75', 0):.2f}%")
            print(f"    AR              : {r.get('AR',   0):.2f}%")
    print("\n  Paper baseline: AP50 = 98.07%  AP75 = 98.07%")
    print("  (BMask R-CNN:   AP50 = 98.41%  AP75 = 98.41%  Δ = +0.34 pp)")

    out_path = os.path.join(test_eval_dir, "maskrcnn_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")
    return results


# ─────────────────────────────────────────────────────────────────────
# Save predictions
# ─────────────────────────────────────────────────────────────────────

def save_predictions(predictor, infer_cfg):
    """
    For each test image: save combined binary mask, instance-label map,
    per-class masks, and colour overlay.
    """
    metadata     = MetadataCatalog.get("my_dataset_test_8")
    class_names  = metadata.thing_classes
    score_thresh = infer_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST

    base          = os.path.join(infer_cfg.OUTPUT_DIR, "test_predictions")
    overlay_dir   = os.path.join(base, "overlay")
    binary_dir    = os.path.join(base, "binary_masks")
    per_class_dir = os.path.join(base, "per_class_masks")
    for d in [overlay_dir, binary_dir, per_class_dir]:
        os.makedirs(d, exist_ok=True)

    stats = {"processed": 0, "skipped": 0, "no_detections": 0}

    for d in DatasetCatalog.get("my_dataset_test_8"):
        im = cv2.imread(d["file_name"])
        if im is None:
            stats["skipped"] += 1; continue

        with torch.no_grad():
            outputs = predictor(im)
        instances = outputs["instances"].to("cpu")
        stem      = Path(d["file_name"]).stem
        h, w      = im.shape[:2]

        if len(instances) == 0:
            stats["no_detections"] += 1
            cv2.imwrite(os.path.join(binary_dir, f"{stem}_combined.png"),
                        np.zeros((h, w), dtype=np.uint8))
            continue

        combined    = np.zeros((h, w), dtype=np.uint8)
        inst_map    = np.zeros((h, w), dtype=np.uint8)
        class_masks = {}

        for idx, (mask, cls_id, score) in enumerate(zip(
                instances.pred_masks.numpy(),
                instances.pred_classes.numpy(),
                instances.scores.numpy())):
            if score < score_thresh:
                continue
            binary   = (mask > 0).astype(np.uint8) * 255
            combined = cv2.bitwise_or(combined, binary)
            inst_map[mask > 0] = min(int((idx + 1) * 35), 255)
            cls_name = (class_names[cls_id]
                        if cls_id < len(class_names) else f"cls{cls_id}")
            if cls_name not in class_masks:
                class_masks[cls_name] = np.zeros((h, w), dtype=np.uint8)
            class_masks[cls_name] = cv2.bitwise_or(
                class_masks[cls_name], binary)

        cv2.imwrite(os.path.join(binary_dir, f"{stem}_combined.png"),  combined)
        cv2.imwrite(os.path.join(binary_dir, f"{stem}_instances.png"), inst_map)

        cls_out = os.path.join(per_class_dir, stem)
        os.makedirs(cls_out, exist_ok=True)
        for cls_name, cls_mask in class_masks.items():
            cv2.imwrite(os.path.join(cls_out, f"{cls_name}.png"), cls_mask)

        v   = Visualizer(im[:, :, ::-1], metadata=metadata,
                         scale=1.0, instance_mode=ColorMode.IMAGE_BW)
        out = v.draw_instance_predictions(instances)
        cv2.imwrite(
            os.path.join(overlay_dir, os.path.basename(d["file_name"])),
            out.get_image()[:, :, ::-1])

        stats["processed"] += 1

    print("\n── Inference complete ────────────────────────────────")
    print(f"  Processed        : {stats['processed']}")
    print(f"  No detections    : {stats['no_detections']}")
    print(f"  Skipped (unread) : {stats['skipped']}")


# ─────────────────────────────────────────────────────────────────────
# Endplate extraction helpers
# ─────────────────────────────────────────────────────────────────────

def _four_corners_adaptive(mask):
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    for eps in np.arange(0.02, 0.20, 0.005):
        approx = cv2.approxPolyDP(contour,
                                  eps * cv2.arcLength(contour, True), True)
        if len(approx) == 4:
            return [tuple(p[0]) for p in approx]
    hull = cv2.convexHull(contour)
    for eps in np.arange(0.02, 0.35, 0.005):
        approx = cv2.approxPolyDP(hull,
                                  eps * cv2.arcLength(hull, True), True)
        if len(approx) == 4:
            return [tuple(p[0]) for p in approx]
    return None


def _adjacent_edge(pts, top=True):
    best_pair, best_y = None, float('inf') if top else float('-inf')
    n = len(pts)
    for i in range(n):
        p1, p2   = pts[i], pts[(i+1) % n]
        mean_y   = (p1[1] + p2[1]) / 2.0
        if (top and mean_y < best_y) or (not top and mean_y > best_y):
            best_y, best_pair = mean_y, (p1, p2)
    return best_pair


def superior_endplate_lumbar(mask):
    pts = _four_corners_adaptive(mask)
    return _adjacent_edge(pts, top=True) if pts else None


def inferior_endplate_lumbar(mask):
    pts = _four_corners_adaptive(mask)
    return _adjacent_edge(pts, top=False) if pts else None


def superior_endplate_s1(mask):
    """Kim et al. (2023) p.1451 — leftmost + topmost raw contour points."""
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    pts        = max(contours, key=cv2.contourArea).reshape(-1, 2)
    left_point = tuple(pts[np.argmin(pts[:, 0])])
    top_point  = tuple(pts[np.argmin(pts[:, 1])])
    return None if left_point == top_point else (left_point, top_point)


def compute_cobb_angle(line1, line2):
    """Cobb angle per Kim et al. (2023) Eq.(2)."""
    def slope(p1, p2):
        dx = float(p2[0] - p1[0])
        return float(p2[1] - p1[1]) / dx if abs(dx) > 1e-6 else None
    m1, m2 = slope(*line1), slope(*line2)
    if m1 is None and m2 is None:
        return 0.0
    if m1 is None:
        return float(np.degrees(np.arctan(abs(1.0/m2)))) if abs(m2) > 1e-6 else 90.0
    if m2 is None:
        return float(np.degrees(np.arctan(abs(1.0/m1)))) if abs(m1) > 1e-6 else 90.0
    denom = 1.0 + m1 * m2
    if abs(denom) < 1e-6:
        return 90.0
    return float(np.degrees(np.arctan(abs((m1 - m2) / denom))))


# ─────────────────────────────────────────────────────────────────────
# Batch angle measurement
# ─────────────────────────────────────────────────────────────────────

def run_angle_measurements(predictor, infer_cfg):
    """
    Compute LLA and LSA for all test images.

    LLA: V1 superior endplate ↔ S1 superior endplate
    LSA: V5 inferior endplate ↔ S1 superior endplate

    Returns:
        auto_LLA, auto_LSA, filenames
    """
    metadata = MetadataCatalog.get("my_dataset_test_8")
    classes  = metadata.thing_classes
    dicts    = DatasetCatalog.get("my_dataset_test_8")

    auto_LLA, auto_LSA, filenames = [], [], []
    n_lla_fail = n_lsa_fail = 0

    for d in dicts:
        im = cv2.imread(d["file_name"])
        if im is None:
            continue

        with torch.no_grad():
            outputs = predictor(im)
        instances = outputs["instances"].to("cpu")

        v1_line = v5_line = s1_line = None
        for i in range(len(instances)):
            cls_name = classes[instances.pred_classes[i].item()].lower()
            mask     = instances.pred_masks[i].numpy().astype(np.uint8)
            if cls_name == "v1":
                v1_line = superior_endplate_lumbar(mask)
            elif cls_name == "v5":
                v5_line = inferior_endplate_lumbar(mask)
            elif cls_name == "s1":
                s1_line = superior_endplate_s1(mask)

        lla = compute_cobb_angle(v1_line, s1_line) if v1_line and s1_line else None
        lsa = compute_cobb_angle(v5_line, s1_line) if v5_line and s1_line else None

        if lla is None: n_lla_fail += 1
        if lsa is None: n_lsa_fail += 1
        auto_LLA.append(lla)
        auto_LSA.append(lsa)
        filenames.append(os.path.basename(d["file_name"]))

    print(f"\nAngle measurements ({len(filenames)} images)")
    print(f"  LLA computed : {sum(v is not None for v in auto_LLA)}  (failed: {n_lla_fail})")
    print(f"  LSA computed : {sum(v is not None for v in auto_LSA)}  (failed: {n_lsa_fail})")

    out = {
        "filenames": filenames,
        "auto_LLA":  [v if v is not None else "null" for v in auto_LLA],
        "auto_LSA":  [v if v is not None else "null" for v in auto_LSA],
    }
    path = os.path.join(infer_cfg.OUTPUT_DIR,
                        "maskrcnn_angle_measurements.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved: {path}")
    return auto_LLA, auto_LSA, filenames


# ─────────────────────────────────────────────────────────────────────
# ICC(2,1)
# ─────────────────────────────────────────────────────────────────────

def compute_icc21(auto, surgeon):
    """
    ICC(2,1) — two-way mixed effects, absolute agreement, single rater.
    Returns dict with icc, ci_lo, ci_hi, mae, bias, sd, n.
    """
    pairs = [(a, s) for a, s in zip(auto, surgeon)
             if a is not None and s is not None]
    if len(pairs) < 3:
        return None
    a = np.array([p[0] for p in pairs])
    s = np.array([p[1] for p in pairs])
    n, k = len(a), 2
    data  = np.column_stack([a, s])
    gm    = data.mean()
    sm    = data.mean(axis=1)
    rm    = data.mean(axis=0)
    msr   = k * np.sum((sm - gm)**2) / (n - 1)
    msc   = n * np.sum((rm - gm)**2) / (k - 1)
    ssw   = np.sum((data - sm[:, None])**2)
    mse   = (ssw - n * np.sum((rm - gm)**2)) / ((n-1)*(k-1))
    icc   = (msr - mse) / (msr + (k-1)*mse + k*(msc - mse)/n)
    F     = msr / mse
    df1, df2 = n - 1, (n-1)*(k-1)
    Flo   = F / sp_stats.f.ppf(0.975, df1, df2)
    Fhi   = F * sp_stats.f.ppf(0.975, df2, df1)
    diffs = a - s
    return {
        "icc":   round(float(icc),   3),
        "ci_lo": round(float(max(0, (Flo-1)/(Flo+k-1))), 3),
        "ci_hi": round(float(min(1, (Fhi-1)/(Fhi+k-1))), 3),
        "mae":   round(float(np.mean(np.abs(diffs))), 2),
        "bias":  round(float(np.mean(diffs)),  2),
        "sd":    round(float(np.std(diffs, ddof=1)), 2),
        "n":     len(pairs),
    }


def run_icc_analysis(auto_LLA, auto_LSA, output_dir,
                     paired_s1_LLA=None, paired_s2_LLA=None,
                     paired_s1_LSA=None, paired_s2_LSA=None):
    """
    Print and save ICC(2,1) table for Mask R-CNN angle measurements.

    Surgeon measurement lists can be imported from src.statistics
    (same data used for BMask R-CNN) for direct comparison.
    """
    print("\n── Mask R-CNN ICC(2,1) ───────────────────────────────────────")
    print(f"  {'Comparison':<30} {'ICC':>6}  {'95% CI':>16}  "
          f"{'MAE':>7}  {'Bias':>7}  {'SD':>7}  {'n':>4}")
    print("  " + "─" * 78)

    comparisons = []
    if paired_s1_LLA:
        comparisons.append(("LLA  auto vs Surgeon 1", auto_LLA, paired_s1_LLA))
    if paired_s2_LLA:
        comparisons.append(("LLA  auto vs Surgeon 2", auto_LLA, paired_s2_LLA))
    if paired_s1_LSA:
        comparisons.append(("LSA  auto vs Surgeon 1", auto_LSA, paired_s1_LSA))
    if paired_s2_LSA:
        comparisons.append(("LSA  auto vs Surgeon 2", auto_LSA, paired_s2_LSA))
    if paired_s1_LLA and paired_s2_LLA:
        comparisons.append(("LLA  S1 vs S2 (inter)",  paired_s1_LLA, paired_s2_LLA))
    if paired_s1_LSA and paired_s2_LSA:
        comparisons.append(("LSA  S1 vs S2 (inter)",  paired_s1_LSA, paired_s2_LSA))

    icc_results = {}
    for label, auto, surg in comparisons:
        r = compute_icc21(auto, surg)
        icc_results[label] = r
        if r:
            print(f"  {label:<30} {r['icc']:>6.3f}  "
                  f"[{r['ci_lo']:.3f} – {r['ci_hi']:.3f}]  "
                  f"{r['mae']:>6.2f}°  {r['bias']:>+6.2f}°  "
                  f"{r['sd']:>6.2f}°  {r['n']:>4}")
        else:
            print(f"  {label:<30}  insufficient data")

    print("\n  ICC: ≥0.90 excellent | 0.70–0.89 good | 0.50–0.69 fair")

    path = os.path.join(output_dir, "maskrcnn_ICC_results.json")
    with open(path, "w") as f:
        json.dump(icc_results, f, indent=2)
    print(f"\nSaved: {path}")
    return icc_results
