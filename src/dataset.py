"""
dataset.py
==========
Dataset registration, class-balance diagnostics, and balanced
dataset builder for the Mask R-CNN vertebral segmentation baseline.

Usage:
    from src.dataset import register_datasets, build_balanced_dataset, register_balanced_dataset
    register_datasets("/content/drive/MyDrive/newDataset_Aug")
    balanced = build_balanced_dataset("my_dataset_train_8")
    register_balanced_dataset(balanced)
"""

import math
import os
import random
from collections import Counter

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances

# Balanced dataset name — distinct from BMask R-CNN's to allow
# both models to coexist in the same Colab session without conflicts
BALANCED_DATASET_NAME = "my_dataset_train_8_balanced_maskrcnn"


# ─────────────────────────────────────────────────────────────────────
# Dataset registration
# ─────────────────────────────────────────────────────────────────────

def register_datasets(base_path: str) -> dict:
    """
    Register train / valid / test COCO-format splits.

    Args:
        base_path: root directory containing train/, valid/, test/ subfolders,
                   each with an _annotations.coco.json file.

    Returns:
        Dict mapping split name → (json_path, img_dir).
    """
    splits = {
        "my_dataset_train_8": (
            f"{base_path}/train/_annotations.coco.json",
            f"{base_path}/train"),
        "my_dataset_valid_8": (
            f"{base_path}/valid/_annotations.coco.json",
            f"{base_path}/valid"),
        "my_dataset_test_8": (
            f"{base_path}/test/_annotations.coco.json",
            f"{base_path}/test"),
    }

    for name, (json_path, img_dir) in splits.items():
        for catalog in (DatasetCatalog, MetadataCatalog):
            try:
                catalog.get(name)
                catalog.remove(name)
            except KeyError:
                pass
        register_coco_instances(name, {}, json_path, img_dir)

    for name in splits:
        dicts = DatasetCatalog.get(name)
        meta  = MetadataCatalog.get(name)
        print(f"  {name}: {len(dicts)} images | classes: {meta.thing_classes}")

    return splits


# ─────────────────────────────────────────────────────────────────────
# Class balance diagnostic
# ─────────────────────────────────────────────────────────────────────

def diagnose_class_balance(split_name: str) -> Counter:
    """
    Print per-class instance counts and flag under/over-represented classes.

    Args:
        split_name: registered Detectron2 dataset name.

    Returns:
        Counter mapping class name → instance count.
    """
    dicts   = DatasetCatalog.get(split_name)
    meta    = MetadataCatalog.get(split_name)
    classes = meta.thing_classes
    counter = Counter()
    missing = []

    for d in dicts:
        anns    = d.get("annotations", [])
        present = {classes[a["category_id"]] for a in anns}
        for a in anns:
            counter[classes[a["category_id"]]] += 1
        absent = [c for c in classes if c not in present]
        if absent:
            missing.append((os.path.basename(d["file_name"]), absent))

    total = sum(counter.values())
    ideal = total / max(len(classes), 1)

    print(f"\n{'─'*55}")
    print(f"{split_name}: {len(dicts)} images | {total} instances")
    print(f"{'Class':<10} {'Count':>7} {'%':>7}  {'vs ideal':>10}  Status")
    print(f"{'─'*55}")
    for cls in classes:
        n    = counter.get(cls, 0)
        pct  = 100 * n / total if total else 0
        diff = (n - ideal) / ideal * 100
        flag = "OK" if abs(diff) < 20 else ("LOW ← fix" if diff < 0 else "HIGH")
        print(f"  {cls:<10} {n:>7} {pct:>6.1f}%  {diff:>+9.1f}%  {flag}")

    if missing:
        print(f"\n  Images missing at least one class: {len(missing)}")
        for fname, absent in missing[:8]:
            print(f"    {fname}  missing: {absent}")
        if len(missing) > 8:
            print(f"    ... and {len(missing)-8} more")
    else:
        print(f"\n  All images have all classes present.")

    return counter


# ─────────────────────────────────────────────────────────────────────
# Balanced dataset builder
# ─────────────────────────────────────────────────────────────────────

def build_balanced_dataset(dataset_name: str,
                            oversample_factor: float = 1.0) -> list:
    """
    Return a balanced list of dataset dicts by oversampling images
    that contain underrepresented vertebral classes.

    Args:
        dataset_name:      registered Detectron2 dataset name.
        oversample_factor: multiplier on top of the balance target.
                           1.0 = balance to the most frequent class.

    Returns:
        Shuffled list of dataset dicts.
    """
    dicts   = DatasetCatalog.get(dataset_name)
    meta    = MetadataCatalog.get(dataset_name)
    classes = meta.thing_classes

    class_counts = Counter()
    for d in dicts:
        for a in d.get("annotations", []):
            class_counts[classes[a["category_id"]]] += 1

    max_count = max(class_counts.values()) if class_counts else 1
    target    = int(max_count * oversample_factor)

    print(f"\nBalancing '{dataset_name}'")
    print(f"  Target count per class : {target}")

    class_to_dicts = {cls: [] for cls in classes}
    for d in dicts:
        for a in d.get("annotations", []):
            cls = classes[a["category_id"]]
            if d not in class_to_dicts[cls]:
                class_to_dicts[cls].append(d)

    balanced = list(dicts)

    for cls in classes:
        current = class_counts.get(cls, 0)
        if current == 0:
            print(f"  WARNING: class '{cls}' has 0 instances — cannot oversample")
            continue
        if current >= target:
            continue
        needed   = target - current
        pool     = class_to_dicts[cls]
        if not pool:
            continue
        n_copies = math.ceil(needed / len(pool))
        extras   = (pool * n_copies)[:needed]
        balanced.extend(extras)
        print(f"  {cls:<10} added {len(extras):>5} copies  "
              f"({current} → {current + len(extras)})")

    random.shuffle(balanced)
    print(f"\n  Total after balancing : {len(balanced)}  (was {len(dicts)})")
    return balanced


# ─────────────────────────────────────────────────────────────────────
# Register balanced dataset
# ─────────────────────────────────────────────────────────────────────

def register_balanced_dataset(balanced_dicts: list,
                               source_name: str = "my_dataset_train_8") -> str:
    """
    Register the balanced dict list as a new Detectron2 dataset,
    copying metadata from the source split.

    Returns:
        The registered dataset name (BALANCED_DATASET_NAME).
    """
    for catalog in (DatasetCatalog, MetadataCatalog):
        try:
            catalog.get(BALANCED_DATASET_NAME)
            catalog.remove(BALANCED_DATASET_NAME)
        except KeyError:
            pass

    DatasetCatalog.register(BALANCED_DATASET_NAME, lambda: balanced_dicts)

    orig_meta = MetadataCatalog.get(source_name)
    MetadataCatalog.get(BALANCED_DATASET_NAME).set(
        thing_classes=orig_meta.thing_classes,
        thing_dataset_id_to_contiguous_id=orig_meta.thing_dataset_id_to_contiguous_id,
        json_file=orig_meta.json_file,
        image_root=orig_meta.image_root,
    )

    n = len(DatasetCatalog.get(BALANCED_DATASET_NAME))
    print(f"\nRegistered '{BALANCED_DATASET_NAME}': {n} images")
    print(f"  Classes: {MetadataCatalog.get(BALANCED_DATASET_NAME).thing_classes}")
    return BALANCED_DATASET_NAME
