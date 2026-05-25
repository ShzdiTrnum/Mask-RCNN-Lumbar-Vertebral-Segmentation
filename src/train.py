"""
train.py
========
Main training entry point for Mask R-CNN vertebral segmentation baseline.

Pipeline:
  1. Register COCO datasets (train / valid / test)
  2. Diagnose class balance
  3. Build balanced training set
  4. Register balanced dataset
  5. Build configuration
  6. Train with MaskRCNNTrainer (BestCheckpointer included)
  7. Plot training metrics

Usage (Colab):
    from src.train import run_training
    run_training(
        base_path  = "/content/drive/MyDrive/newDataset_Aug",
        output_dir = "/content/drive/MyDrive/MaskRCNN_Baseline",
    )

CLI:
    python src/train.py \
        --base_path  /content/drive/MyDrive/newDataset_Aug \
        --output_dir /content/drive/MyDrive/MaskRCNN_Baseline
"""

import argparse
import os

from src.dataset import (
    register_datasets,
    diagnose_class_balance,
    build_balanced_dataset,
    register_balanced_dataset,
)
from src.trainer import build_cfg, MaskRCNNTrainer
from src.metrics import plot_training_metrics


def run_training(base_path: str,
                 output_dir: str,
                 oversample_factor: float = 1.0,
                 backbone_weights: str = "/content/R-101.pkl"):
    """
    Full Mask R-CNN training pipeline.

    Args:
        base_path:         root of the COCO dataset (train/valid/test subfolders).
        output_dir:        where to save checkpoints and results.
        oversample_factor: class-balancing multiplier (default 1.0).
        backbone_weights:  path to R-101.pkl pre-trained backbone.
    """
    print("=" * 60)
    print("  Mask R-CNN — Vertebral Segmentation Baseline")
    print("  (Comparison baseline for BMask R-CNN ablation)")
    print("=" * 60)

    print("\n[1/5] Registering datasets ...")
    register_datasets(base_path)

    print("\n[2/5] Class balance diagnostics ...")
    diagnose_class_balance("my_dataset_train_8")
    diagnose_class_balance("my_dataset_valid_8")
    diagnose_class_balance("my_dataset_test_8")

    print("\n[3/5] Building balanced training set ...")
    balanced = build_balanced_dataset(
        "my_dataset_train_8", oversample_factor=oversample_factor)

    print("\n[4/5] Registering balanced dataset ...")
    balanced_name = register_balanced_dataset(balanced)

    print("\n[5/5] Building configuration ...")
    cfg = build_cfg(
        balanced_dataset_name=balanced_name,
        output_dir=output_dir,
        backbone_weights=backbone_weights,
    )

    last_ckpt = os.path.join(output_dir, "last_checkpoint")
    resume    = os.path.exists(last_ckpt)
    if resume:
        with open(last_ckpt) as f:
            print(f"\nResuming from: {f.read().strip()}")
    else:
        print("\nStarting fresh Mask R-CNN training.")

    trainer = MaskRCNNTrainer(cfg)
    trainer.resume_or_load(resume=resume)
    trainer.train()

    print("\n✅ Training complete.")
    print(f"   Checkpoints → {output_dir}")

    print("\nGenerating training metric plots ...")
    plot_training_metrics(output_dir)

    return cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Mask R-CNN baseline for vertebral segmentation")
    parser.add_argument("--base_path",  required=True,
                        help="Root dataset directory (train/valid/test)")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for checkpoints and results")
    parser.add_argument("--oversample", type=float, default=1.0,
                        help="Oversample factor for class balancing (default 1.0)")
    parser.add_argument("--backbone",
                        default="/content/R-101.pkl",
                        help="Path to R-101.pkl backbone weights")
    args = parser.parse_args()

    run_training(
        base_path         = args.base_path,
        output_dir        = args.output_dir,
        oversample_factor = args.oversample,
        backbone_weights  = args.backbone,
    )
