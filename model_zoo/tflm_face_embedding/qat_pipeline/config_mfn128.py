"""
MFN-128D training config — MobileFaceNet scale=1, 128D output.
Matches esp-dl MFN architecture (~1.2M params) with reduced embedding dimension.

Usage:
    torchrun --nproc_per_node=N train_v2.py configs/mfn128_ms1mv3.py
"""
import os
from easydict import EasyDict as edict

config = edict()
config.margin_list = (1.0, 0.5, 0.0)
config.network = "mfn128"
config.resume = False
config.output = os.environ.get("TRAIN_OUTPUT", "/workspace/output/mfn128")
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 1e-4
# Batch size auto-scaled by GPU count (tiny model, can use large batches)
_NUM_GPUS = int(os.environ.get("NUM_GPUS", "1"))
config.batch_size = {1: 512, 2: 384, 4: 256, 8: 160}.get(_NUM_GPUS, 256)
config.lr = 0.1
config.verbose = 2000
config.dali = False
config.seed = 42
config.save_all_states = True

# MS1MV3 dataset (aligned 112×112)
config.rec = os.environ.get("DATA_DIR", "/data/ms1m-retinaface-t1")
config.num_classes = 93431
config.num_image = 5179510
config.num_epoch = 60
config.warmup_epoch = 2
config.val_targets = ['lfw', 'cfp_fp', "agedb_30"]
