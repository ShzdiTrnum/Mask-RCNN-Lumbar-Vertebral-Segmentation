"""
trainer.py
==========
MaskRCNNTrainer with:
  - Balanced dataset loader
  - BestCheckpointer hook (saves model_best.pth on highest segm/AP)
  - Compatible COCOEvaluator builder
  - Standard Mask R-CNN R-101-FPN configuration — identical hyperparameters
    to the BMask R-CNN config EXCEPT no BOUNDARY_MASK_HEAD settings.
    This ensures the only architectural variable in the comparison is the
    presence or absence of the boundary supervision branch.

Usage:
    from src.trainer import build_cfg, MaskRCNNTrainer
    cfg     = build_cfg(balanced_dataset_name, output_dir)
    trainer = MaskRCNNTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
"""

import inspect
import logging
import os

import torch
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, build_detection_train_loader
from detectron2.engine import DefaultTrainer, hooks
from detectron2.engine.hooks import HookBase
from detectron2.evaluation import COCOEvaluator
from detectron2.model_zoo import model_zoo


# ─────────────────────────────────────────────────────────────────────
# BestCheckpointer hook
# ─────────────────────────────────────────────────────────────────────

class BestCheckpointer(HookBase):
    """
    Saves a checkpoint whenever the monitored validation metric improves.
    Output: model_best.pth in cfg.OUTPUT_DIR.
    """
    def __init__(self, eval_period, checkpointer,
                 val_metric="segm/AP", mode="max",
                 file_prefix="model_best"):
        self._eval_period  = eval_period
        self._checkpointer = checkpointer
        self._val_metric   = val_metric
        self._file_prefix  = file_prefix
        assert mode in ("max", "min")
        self._mode = mode
        self._best = None

    def _is_better(self, v):
        if self._best is None:
            return True
        return v > self._best if self._mode == "max" else v < self._best

    def _try_save(self, iteration):
        try:
            val, _ = self.trainer.storage.latest().get(
                self._val_metric, (None, None))
        except Exception:
            val = None
        if val is None:
            return
        if self._is_better(val):
            self._best = val
            self._checkpointer.save(self._file_prefix)
            logging.getLogger("detectron2").info(
                f"BestCheckpointer: new best {self._val_metric}="
                f"{val:.4f} at iter {iteration} → '{self._file_prefix}.pth'")

    def after_step(self):
        nxt = self.trainer.iter + 1
        if nxt % self._eval_period == 0 and nxt < self.trainer.max_iter:
            self._try_save(nxt)

    def after_train(self):
        self._try_save(self.trainer.max_iter)


# Inject into detectron2 hooks module
import detectron2.engine.hooks as _hooks
_hooks.BestCheckpointer = BestCheckpointer


# ─────────────────────────────────────────────────────────────────────
# Compatible COCOEvaluator factory
# ─────────────────────────────────────────────────────────────────────

_coco_sig = inspect.signature(COCOEvaluator.__init__).parameters


def build_coco_evaluator(dataset_name: str, cfg, output_dir: str):
    """Build COCOEvaluator compatible with multiple detectron2 API versions."""
    os.makedirs(output_dir, exist_ok=True)
    if "tasks" in _coco_sig:
        return COCOEvaluator(
            dataset_name,
            tasks=("bbox", "segm"),
            distributed=False,
            output_dir=output_dir,
        )
    return COCOEvaluator(dataset_name, cfg, False, output_dir)


# ─────────────────────────────────────────────────────────────────────
# Custom trainer
# ─────────────────────────────────────────────────────────────────────

class MaskRCNNTrainer(DefaultTrainer):
    """
    Standard Mask R-CNN trainer — baseline for comparison with BMask R-CNN.
    Identical to BMask R-CNN's CustomTrainer except uses the standard
    Mask R-CNN config (no boundary head).
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "eval", dataset_name)
        return build_coco_evaluator(dataset_name, cfg, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        """Uses cfg.DATASETS.TRAIN → balanced dataset name set in build_cfg()."""
        return build_detection_train_loader(cfg)

    def build_hooks(self):
        hook_list = super().build_hooks()
        hook_list.insert(
            -1,
            hooks.BestCheckpointer(
                eval_period=self.cfg.TEST.EVAL_PERIOD,
                checkpointer=self.checkpointer,
                val_metric="segm/AP",
                mode="max",
                file_prefix="model_best",
            ),
        )
        return hook_list


# ─────────────────────────────────────────────────────────────────────
# Configuration builder
# ─────────────────────────────────────────────────────────────────────

def build_cfg(balanced_dataset_name: str,
              output_dir: str,
              backbone_weights: str = "/content/R-101.pkl"):
    """
    Build and freeze the Mask R-CNN training configuration.

    All hyperparameters are IDENTICAL to the BMask R-CNN configuration
    EXCEPT there is no BOUNDARY_MASK_HEAD — this is the only architectural
    difference, ensuring a fair ablation comparison.

    Args:
        balanced_dataset_name: name of the balanced training dataset.
        output_dir:            directory for checkpoints and evaluation.
        backbone_weights:      path to the pre-trained R-101 backbone (.pkl).

    Returns:
        Frozen CfgNode ready for MaskRCNNTrainer.
    """
    cfg = get_cfg()

    # Standard Mask R-CNN R-101 FPN from detectron2 model zoo
    # NOTE: does NOT use the BMask R-CNN YAML — this is the key difference
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))

    # ── Datasets ──────────────────────────────────────────────────────
    cfg.DATASETS.TRAIN = (balanced_dataset_name,)
    cfg.DATASETS.TEST  = ("my_dataset_valid_8",)

    # ── Output ────────────────────────────────────────────────────────
    cfg.OUTPUT_DIR = output_dir
    os.makedirs(output_dir, exist_ok=True)

    # ── Model ─────────────────────────────────────────────────────────
    # Same R-101 backbone as BMask R-CNN — isolates boundary head effect
    cfg.MODEL.WEIGHTS            = backbone_weights
    cfg.MODEL.BACKBONE.FREEZE_AT = 0

    # ── ROI heads — identical to BMask R-CNN ──────────────────────────
    cfg.MODEL.ROI_HEADS.NUM_CLASSES          = 7
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST    = 0.3
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION    = 0.5
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST      = 0.5

    # ── Mask head — same NUM_CONV as BMask R-CNN ──────────────────────
    # NOTE: No BOUNDARY_MASK_HEAD — this is the sole architectural difference
    cfg.MODEL.ROI_MASK_HEAD.NUM_CONV    = 6
    cfg.MODEL.ROI_MASK_HEAD.LOSS_WEIGHT = 1.0

    # ── RPN — identical to BMask R-CNN ────────────────────────────────
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN   = 3000
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN  = 2000
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST   = 1000
    cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.RPN.NMS_THRESH           = 0.7

    # ── Input — identical to BMask R-CNN ──────────────────────────────
    cfg.INPUT.MIN_SIZE_TRAIN = (800, 896, 1024)
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TEST  = 1024
    cfg.INPUT.MAX_SIZE_TEST  = 1333
    cfg.INPUT.RANDOM_FLIP    = "horizontal"

    # ── Solver — identical to BMask R-CNN proven config ───────────────
    cfg.SOLVER.IMS_PER_BATCH       = 2
    cfg.SOLVER.BASE_LR             = 0.001
    cfg.SOLVER.MAX_ITER            = 30000
    cfg.SOLVER.STEPS               = (22000, 25000)
    cfg.SOLVER.WARMUP_ITERS        = 1000
    cfg.SOLVER.LR_SCHEDULER_NAME   = "WarmupCosineLR"
    cfg.SOLVER.WEIGHT_DECAY        = 0.0001
    cfg.SOLVER.MOMENTUM            = 0.9
    cfg.SOLVER.CHECKPOINT_PERIOD   = 2000

    cfg.SOLVER.CLIP_GRADIENTS.ENABLED    = True
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE  = "norm"
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
    cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE  = 2.0

    # ── Evaluation ────────────────────────────────────────────────────
    cfg.TEST.EVAL_PERIOD          = 2000
    cfg.TEST.DETECTIONS_PER_IMAGE = 100

    cfg.freeze()

    print("\n── Mask R-CNN Configuration ─────────────────────────")
    print(f"  Backbone    : ResNet-101 + FPN")
    print(f"  Mask convs  : {cfg.MODEL.ROI_MASK_HEAD.NUM_CONV}")
    print(f"  Boundary    : NONE  ← key difference from BMask R-CNN")
    print(f"  Train set   : {cfg.DATASETS.TRAIN[0]}")
    print(f"  Train images: {len(DatasetCatalog.get(balanced_dataset_name))}")
    print(f"  LR          : {cfg.SOLVER.BASE_LR}")
    print(f"  Max iters   : {cfg.SOLVER.MAX_ITER}")
    print(f"  Output dir  : {cfg.OUTPUT_DIR}")

    return cfg
