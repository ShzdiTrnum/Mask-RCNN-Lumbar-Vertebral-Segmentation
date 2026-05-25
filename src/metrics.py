"""
metrics.py
==========
Training metric visualization from detectron2's metrics.json log.

Generates:
  - loss_curves.png     : total loss + component losses
  - segm_ap.png         : segmentation AP / AP50 / AP75
  - bbox_ap.png         : bounding-box AP / AP50 / AP75
  - ap50_vs_ap75.png    : combined AP50 vs AP75 comparison

Usage:
    from src.metrics import plot_training_metrics
    plot_training_metrics("/content/drive/MyDrive/MaskRCNN_Baseline")
"""

import json
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


def _smooth(values, window: int = 50):
    v = np.array(values)
    if len(v) < window:
        return v
    return np.convolve(v, np.ones(window) / window, "valid")


def _smooth_x(x, window: int = 50):
    if len(x) < window:
        return np.array(x)
    return np.array(x)[window - 1:]


def plot_training_metrics(output_dir: str, smooth_window: int = 50):
    """
    Read metrics.json and produce four diagnostic plots.

    Args:
        output_dir:    directory containing metrics.json.
        smooth_window: moving-average window for loss curves.
    """
    metrics_file = os.path.join(output_dir, "metrics.json")
    if not os.path.exists(metrics_file):
        print(f"metrics.json not found at {metrics_file}")
        return

    with open(metrics_file) as f:
        records = [json.loads(line) for line in f if line.strip()]

    train    = {k: [] for k in ["iter", "total_loss", "loss_mask",
                                 "loss_cls", "loss_box_reg", "loss_rpn_cls",
                                 "loss_rpn_loc"]}
    bbox_val = {"iter": [], "AP": [], "AP50": [], "AP75": []}
    segm_val = {"iter": [], "AP": [], "AP50": [], "AP75": []}

    for m in records:
        if "iteration" not in m:
            continue
        it = m["iteration"]
        if "total_loss" in m:
            train["iter"].append(it)
            for k in ["total_loss", "loss_mask", "loss_cls",
                      "loss_box_reg", "loss_rpn_cls", "loss_rpn_loc"]:
                train[k].append(m.get(k, np.nan))
        if "bbox/AP" in m:
            bbox_val["iter"].append(it)
            for k in ["AP", "AP50", "AP75"]:
                bbox_val[k].append(m.get(f"bbox/{k}", np.nan))
        if "segm/AP" in m:
            segm_val["iter"].append(it)
            for k in ["AP", "AP50", "AP75"]:
                segm_val[k].append(m.get(f"segm/{k}", np.nan))

    iters = np.array(train["iter"])
    W = smooth_window

    # ── Loss curves ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, key, color, lw in [
        ("Total loss",   "total_loss",   "#2C3E50", 2.0),
        ("Mask loss",    "loss_mask",    "#2980B9", 1.5),
        ("Cls loss",     "loss_cls",     "#27AE60", 1.2),
        ("BBox reg",     "loss_box_reg", "#8E44AD", 1.0),
        ("RPN cls",      "loss_rpn_cls", "#E74C3C", 1.0),
        ("RPN loc",      "loss_rpn_loc", "#F39C12", 1.0),
    ]:
        v = np.array(train[key])
        if len(v) > 0 and not np.all(np.isnan(v)):
            ax.plot(_smooth_x(iters, W), _smooth(v, W),
                    label=label, color=color, lw=lw, alpha=0.9)
    ax.set(xlabel="Iteration", ylabel="Loss",
           title="Mask R-CNN Training Losses\n(No boundary loss — standard mask head only)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "loss_curves.png")
    plt.savefig(path, dpi=300); plt.close(fig)
    print(f"Saved: {path}")

    # ── Segmentation AP ───────────────────────────────────────────────
    if segm_val["iter"]:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(segm_val["iter"], segm_val["AP"],
                "o-",  ms=5, lw=1.8, color="#2C3E50",
                label="Segm AP (0.50:0.95)")
        ax.plot(segm_val["iter"], segm_val["AP50"],
                "s--", ms=5, lw=1.5, color="#16A085",
                label="Segm AP50")
        ax.plot(segm_val["iter"], segm_val["AP75"],
                "^:",  ms=5, lw=2.0, color="#8E44AD",
                label="Segm AP75 ← primary")
        ax.set(xlabel="Iteration", ylabel="AP (%)",
               title="Mask R-CNN — Validation Segmentation AP", ylim=(0, 105))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(output_dir, "segm_ap.png")
        plt.savefig(path, dpi=300); plt.close(fig)
        print(f"Saved: {path}")

    # ── BBox AP ───────────────────────────────────────────────────────
    if bbox_val["iter"]:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(bbox_val["iter"], bbox_val["AP"],
                "o-",  ms=5, lw=1.8, color="#2980B9",
                label="BBox AP (0.50:0.95)")
        ax.plot(bbox_val["iter"], bbox_val["AP50"],
                "s--", ms=5, lw=1.5, color="#27AE60",
                label="BBox AP50")
        ax.plot(bbox_val["iter"], bbox_val["AP75"],
                "^:",  ms=5, lw=1.5, color="#E74C3C",
                label="BBox AP75")
        ax.set(xlabel="Iteration", ylabel="AP (%)",
               title="Mask R-CNN — Validation BBox AP", ylim=(0, 105))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(output_dir, "bbox_ap.png")
        plt.savefig(path, dpi=300); plt.close(fig)
        print(f"Saved: {path}")

    # ── AP50 vs AP75 ──────────────────────────────────────────────────
    if segm_val["iter"] and bbox_val["iter"]:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(bbox_val["iter"], bbox_val["AP50"],
                "s--", ms=4, lw=1.4, color="#27AE60", label="BBox AP50")
        ax.plot(bbox_val["iter"], bbox_val["AP75"],
                "^--", ms=4, lw=1.4, color="#E74C3C", label="BBox AP75")
        ax.plot(segm_val["iter"], segm_val["AP50"],
                "s-",  ms=4, lw=1.8, color="#2980B9", label="Segm AP50")
        ax.plot(segm_val["iter"], segm_val["AP75"],
                "^-",  ms=4, lw=2.0, color="#8E44AD", label="Segm AP75")
        ax.set(xlabel="Iteration", ylabel="AP (%)",
               title="Mask R-CNN — AP50 vs AP75", ylim=(0, 105))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(output_dir, "ap50_vs_ap75.png")
        plt.savefig(path, dpi=300); plt.close(fig)
        print(f"Saved: {path}")

    print("\n── Final validation metrics ──────────────────────────")
    if segm_val["AP"]:
        print(f"  Segm AP           : {segm_val['AP'][-1]:.2f}%")
        print(f"  Segm AP50         : {segm_val['AP50'][-1]:.2f}%")
        print(f"  Segm AP75[primary]: {segm_val['AP75'][-1]:.2f}%")
    if bbox_val["AP"]:
        print(f"  BBox AP50         : {bbox_val['AP50'][-1]:.2f}%")
        print(f"  BBox AP75         : {bbox_val['AP75'][-1]:.2f}%")
