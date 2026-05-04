# Face Embedding Models for Grove Vision AI V2

Face detection (SCRFD) + Face embedding (GhostFaceNet/MobileFaceNet) for Ethos-U55 NPU.

## Final Models (Ready to Flash)

| Model | File | Size | Flash Address |
|-------|------|------|---------------|
| SCRFD-500M-KPS | `scrfd/models/scrfd_500m_kps_int8_vela.tflite` | 701 KB | 0x200000 |
| GhostFaceNet-0.5 | `ghostfacenet/models/ghostfacenet_fixed_int8_vela.tflite` | 849 KB | 0x400000 |

## Model Specifications

### SCRFD-500M-KPS (Face Detection)
- Input: [1, 160, 160, 3] INT8
- Output: Multi-scale detection (bounding boxes + 5-point landmarks)
- SRAM: ~201 KB
- Inference: ~4.4ms @ 500MHz

### GhostFaceNet-0.5 (Face Embedding)
- Input: [1, 112, 112, 3] INT8
- Output: [1, 512] INT8 (512-dimensional embedding)
- SRAM: ~245 KB
- Inference: ~17ms @ 500MHz

## Directory Structure

```
tflm_face_embedding/
├── scrfd/                           # SCRFD Face Detection
│   ├── models/                      # Model files
│   │   ├── scrfd_500m_kps_int8_vela.tflite  # Final model (flash to 0x200000)
│   │   ├── scrfd_500m_kps_int8.tflite       # INT8 quantized
│   │   ├── scrfd_500m_kps.onnx              # Original ONNX
│   │   └── scrfd_500m_kps.pth               # PyTorch weights
│   ├── scripts/                     # Conversion scripts
│   │   ├── convert_scrfd.py         # PTQ conversion
│   │   ├── convert_scrfd_enhanced.py # Enhanced conversion
│   │   └── scrfd_model.py           # Model definition
│   └── quantization/                # QAT training
│       ├── qat_scrfd_enhanced.py    # Main QAT training script
│       ├── run_qat_enhanced.sh      # Training runner
│       ├── export_from_checkpoint.py # Export trained model
│       ├── validate_quantization.py # Validation script
│       └── README.md                # QAT documentation
│
├── ghostfacenet/                    # GhostFaceNet Face Embedding
│   ├── models/                      # Model files
│   │   ├── ghostfacenet_fixed_int8_vela.tflite  # Final model (flash to 0x400000)
│   │   ├── ghostfacenet_fixed_int8.tflite       # INT8 quantized
│   │   ├── ghostfacenet_float32.onnx            # Float ONNX
│   │   └── GN_W0.5_S2_ArcFace_epoch16.h5        # Original H5
│   └── scripts/                     # Conversion scripts
│       ├── convert_ghostfacenet.py  # Main conversion script
│       ├── qat_ghostfacenet.py      # QAT training
│       └── fix_ghostfacenet_overlap.py  # Memory overlap fix
│
├── calibration_data/                # Calibration data for INT8 quantization
│   ├── fd_160/                      # Face detection calibration (160x160)
│   ├── emb_112/                     # Embedding calibration (112x112)
│   └── lfw/                         # LFW dataset for validation
│
├── foamliu_mobilefacenet_128d/      # MobileFaceNet alternative
│
├── prepare_calibration_data.py      # Generate calibration images
├── download_datasets.py             # Download training datasets
├── analyze_*.py                     # Analysis scripts
├── SCRFD_DECODING.md               # SCRFD output format documentation
└── pyproject.toml                   # Python dependencies
```

## Quick Start

```bash
# Setup environment
cd model_zoo/tflm_face_embedding
uv sync

# Convert GhostFaceNet (PTQ)
cd ghostfacenet/scripts
uv run python convert_ghostfacenet.py

# Convert SCRFD (PTQ)
cd ../../scrfd/scripts
uv run python convert_scrfd.py

# SCRFD QAT Training (for better accuracy)
cd ../quantization
./run_qat_enhanced.sh
```

## Evaluation Scripts

Use these from `model_zoo/tflm_face_embedding` with `uv run python ...`.

| Script | Purpose |
|--------|---------|
| `_device_compare.py` | Simulates firmware-side preprocessing and embedding extraction. Use this when checking whether PC results match device behavior, including the `pixel - 129` INT8 input path. |
| `run_full_comparison.py` | Full LFW + CFP-FP model comparison. Reports same-person vs different-person separation, best threshold accuracy, same mean, and diff mean across BASELINE/TTA/CENTER variants. |
| `run_cfp_only.py` | CFP-FP-only stress test for frontal/profile face pairs. Use this after LFW or when checking harder pose variation. |
| `evaluate_w600k_compression.py` | PC-side 512D to 128D post-embedding compression check for `official_mobilefacenet/w600k_mbf_int8.tflite`. Compares baseline 512D, truncation, random projection, and PCA projection. This does not reduce Ethos-U tensor arena SRAM by itself. |
| `train_w600k_projection_128d.py` | Trains a 512D to 128D projection from cached w600k teacher embeddings. Produces `outputs/w600k_projection_128d.npz` with float32 and int8 projection weights for later integration experiments. |
| `train_mfn_student_distill.py` | Trains a compact MobileFaceNet-style 128D student from w600k teacher embeddings. Use this for SRAM-reduction experiments; it supports width scaling, checkpoint continuation, weighted pairwise loss, and hard-negative margin loss. |
| `train_mfn_student_pair_finetune.py` | Fine-tunes an S2 student with LFW DevTrain matched/mismatched pairs while retaining a configurable w600k projection distillation loss. It can optionally add CFP-FP train splits with `--cfp-splits`; reserve split 01 for evaluation. |
| `download_glint360k_subset.py` | Downloads a bounded aligned Glint360K WebDataset subset from Hugging Face into `datasets/glint360k_subset_112/<identity>/*.jpg`. Use for internal/research training unless dataset licensing is cleared for product use. |
| `run_s2_w1_pairft_conservative_remote.sh` | WSL2 remote sweep for conservative S2 pair fine-tuning. It reuses aligned/teacher caches, runs two higher-distillation 8-epoch variants, and is intended to test CFP retention before downloading larger face datasets. |
| `run_s2_w1_pairft_score_sweep_remote.sh` | WSL2 remote sweep that continues from the balanced checkpoint and tries higher-distillation settings against the balanced single-threshold metric. |
| `run_s2_w1_pairft_cfp_score_remote.sh` | WSL2 remote sweep that adds CFP-FP splits 02-10 as training pairs while leaving split 01 for evaluation. Use this only as a controlled cross-pose experiment. |
| `run_s2_w1_pairft_cfp_fallback_remote.sh` | WSL2 remote sweep after enabling cropped-face fallback. Trains with CFP-FP splits 02-10 and reuses the generated fallback aligned/teacher caches. |
| `run_s2_w1_pairft_threshold_remote.sh` | WSL2 remote sweep that continues from `cfp_fb_b` and adds threshold-aware hinge loss around the deployment threshold. |
| `run_s2_w1_pairft_teacher_hn_remote.sh` | WSL2 remote sweep that continues from `thr_a`, adds teacher 512D pair-similarity loss, and mines high-similarity different-identity hard negatives. |
| `run_s2_w1_pairft_arcface_remote.sh` | WSL2 remote sweep that continues from `tpair_hn_a` and adds a training-only ArcFace identity head. The exported TFLite still contains only the compact embedding model. |
| `run_s2_w1_pairft_glint_arcface_remote.sh` | WSL2 remote run that downloads a Glint360K aligned subset and adds external identity images to ArcFace training. |
| `evaluate_embedding_models.py` | Compares multiple pre-Vela embedding models on the same local LFW/CFP pairs. Use this before considering a lower-SRAM model swap. |
| `compute_embedding.py` | Shared PC-side SCRFD + alignment + embedding pipeline used by the evaluation scripts. Also useful for one-off image pair checks. |

Important notes:
- The `models` dictionaries in these scripts may need to be updated before each experiment; older entries point to `mobilefacenet_no_bn_*` or `mobilefacenet_qat_*`.
- Vela models with the `ethos-u` custom op cannot run directly in PC TFLite. Evaluate the pre-Vela INT8 model for accuracy, then use Vela summary/output for device memory and NPU coverage.
- The current best-discriminating baseline found during recent testing was the original w600k quantized model: `official_mobilefacenet/w600k_mbf_int8.tflite`.
- To evaluate a trained projection, pass it to `evaluate_w600k_compression.py`, for example: `uv run python evaluate_w600k_compression.py --projection outputs/w600k_projection_128d.npz`.
- `evaluate_embedding_models.py` also prints a balanced single-threshold table. Use that table for model selection because firmware normally needs one recognition threshold across scenes. The table reports the shared threshold, LFW/CFP-FP accuracy at that threshold, `Floor=min(LFW, CFP-FP)`, `Gap=abs(LFW-CFP-FP)`, and `Score=harmonic_mean(LFW, CFP-FP) - 0.25 * Gap`.
- `evaluate_embedding_models.py` also prints production-oriented verification metrics: EER, TAR@FAR10%, TAR@FAR5%, TAR@FAR1%, and FAR/FRR at the balanced shared threshold. Use these for production feasibility and false-accept risk; the balanced score alone is not a production safety metric.
- `compute_embedding.py` keeps firmware-equivalent SCRFD alignment by default. `evaluate_embedding_models.py` and `train_mfn_student_pair_finetune.py` enable an opt-in center-crop fallback for already-cropped benchmark faces when SCRFD detects no face. This makes CFP-FP profile evaluation/training complete instead of dropping hard profile crops.

Recent 128D projection result:
- Trained on WSL2 `wsl2-local` from 1196 valid w600k teacher embeddings with `train_w600k_projection_128d.py --num-train 1200 --steps 1000`.
- Output: `outputs/w600k_projection_128d.npz`, including float32 projection weights and int8 projection weights.
- Matrix size: 256 KiB float32, 64 KiB int8.
- Local evaluation command: `uv run python evaluate_w600k_compression.py --max-pairs 120 --num-calib 800 --projection outputs/w600k_projection_128d.npz`.
- LFW: 512D baseline `sep=0.5560 acc=97.8%`; 128D trained projection `sep=0.6008 acc=97.8%`.
- CFP-FP: 512D baseline `sep=0.1110 acc=78.0%`; 128D trained projection `sep=0.1593 acc=79.7%`.
- This projection is suitable for 128D matching experiments after w600k inference, but it does not reduce the w600k Ethos-U tensor arena peak SRAM.

Recent S2 student compression result:
- Training ran on WSL2 `wsl2-local` with `/home/harve/.local/bin/uv`; TensorFlow used RTX 3060 GPU.
- `official_mobilefacenet/student_distill_w1_margin/mfn_w1_distill_128d.int8.tflite`: width `1.0`, 128D, 60 epochs, weighted pairwise + hard-negative distillation. Vela: `599.84 KiB` SRAM, `1057.14 KiB` flash, `CPU ops=0`, `NPU=100%`.
- `official_mobilefacenet/student_distill_w1_hardneg/mfn_w1_distill_128d.int8.tflite`: 24 more epochs from the width `1.0` checkpoint with lower LR and stricter negative margin. Vela: `599.84 KiB` SRAM, `1056.83 KiB` flash, `CPU ops=0`, `NPU=100%`.
- Best hard-negative local evaluation command: `uv run python evaluate_embedding_models.py --max-pairs 120 --model w600k-512d=official_mobilefacenet/w600k_mbf_int8.tflite --model s2-w1-hardneg-i8=official_mobilefacenet/student_distill_w1_hardneg/mfn_w1_distill_128d.int8.tflite`.
- Hard-negative S2 result: LFW `sep=0.1462 acc=73.9%`; CFP-FP `sep=0.1104 acc=78.0%`.
- Conclusion: the S2 architecture meets the SRAM target, and CFP-FP is close to w600k, but LFW is still far below the w600k baseline (`sep=0.5560 acc=97.8%`). Do not replace w600k with this student yet. The next useful path is supervised identity or pair-based fine-tuning from aligned LFW/CFP or a larger labeled face dataset, not S3.

Recent S2 pair fine-tune result:
- `official_mobilefacenet/student_distill_w1_pairft/mfn_w1_pairft_128d.int8.tflite`: LFW improved to `sep=0.3231 acc=81.1%`, but CFP-FP regressed to `sep=0.0754 acc=74.6%`.
- `official_mobilefacenet/student_distill_w1_pairft_balanced/mfn_w1_pairft_128d.int8.tflite`: balanced loss (`distill=1.0`, lower positive weight, stronger negative weight) gave LFW `sep=0.2510 acc=81.7%` and CFP-FP `sep=0.0813 acc=76.3%`.
- Vela for the balanced pair fine-tune: `599.84 KiB` SRAM, `1056.28 KiB` flash, `CPU ops=0`, `NPU=100%`.
- Conclusion: supervised pair loss improves LFW without increasing SRAM, but current local LFW-only supervision hurts cross-pose CFP-FP versus the hard-negative distillation model. Width `1.0` is not the immediate bottleneck; widening should wait until there is broader labeled training data or a better validation split.

Recent conservative S2 pair fine-tune sweep:
- Remote command: `bash /home/harve/gv2_face_train/tflm_face_embedding/run_s2_w1_pairft_conservative_remote.sh` on WSL2 `wsl2-local` with `/home/harve/.local/bin/uv`; TensorFlow created `GPU:0` on RTX 3060.
- `official_mobilefacenet/student_distill_w1_pairft_distill2/mfn_w1_pairft_128d.int8.tflite`: 8 epochs from hard-negative weights, `distill=2.0`, `positive=0.4`, `negative=8.0`, `margin=0.03`. LFW `sep=0.1667 acc=77.8%`; CFP-FP `sep=0.1038 acc=79.7%`. Vela: `599.84 KiB` SRAM, `1057.09 KiB` flash, `CPU ops=0`, `NPU=100%`.
- `official_mobilefacenet/student_distill_w1_pairft_distill3/mfn_w1_pairft_128d.int8.tflite`: 8 epochs from hard-negative weights, `distill=3.0`, `positive=0.25`, `negative=8.0`, `margin=0.02`. LFW `sep=0.1642 acc=78.3%`; CFP-FP `sep=0.0977 acc=81.4%`. Vela: `599.84 KiB` SRAM, `1057.22 KiB` flash, `CPU ops=0`, `NPU=100%`.
- Conclusion: higher teacher retention avoids the CFP-FP regression seen in LFW-heavy pair fine-tuning, but it cannot recover w600k-like LFW discrimination. Do not download more pair-only validation data first. If more data is needed, prioritize identity-labeled, multi-pose face training data and an ArcFace-style identity objective, then distill into the S2 architecture. The current local LFW pair supervision is useful for diagnosis but too narrow to be the main training signal.

Balanced single-threshold S2 selection:
- Command: `uv run python evaluate_embedding_models.py --max-pairs 120 --model hardneg=official_mobilefacenet/student_distill_w1_hardneg/mfn_w1_distill_128d.int8.tflite --model balanced=official_mobilefacenet/student_distill_w1_pairft_balanced/mfn_w1_pairft_128d.int8.tflite --model distill2=official_mobilefacenet/student_distill_w1_pairft_distill2/mfn_w1_pairft_128d.int8.tflite --model distill3=official_mobilefacenet/student_distill_w1_pairft_distill3/mfn_w1_pairft_128d.int8.tflite`.
- Current ranking by `Score=harmonic_mean(LFW@Thr, CFP@Thr) - 0.25 * Gap`: `balanced` score `0.762`, shared threshold `0.1589`, LFW `76.1%`, CFP-FP `76.3%`; `distill2` score `0.753`; `distill3` score `0.751`; `hardneg` score `0.713`.
- Use the balanced single-threshold table for deployment candidate selection. Use the per-dataset best-threshold tables only for diagnosis, because they hide threshold-transfer risk.

Score-directed S2 training attempts:
- Continuing from `balanced` with higher distillation (`score_a/b/c`) improved LFW best-threshold accuracy to `82.8%`, but lowered the single-threshold score to `0.720-0.724`. This shifts the similarity distribution and is worse for deployment.
- Adding CFP-FP train splits 02-10 (`cfp_score_a`) improved best-threshold LFW/CFP to `82.8%/78.0%`, but the shared-threshold score dropped to `0.719` with threshold `0.0138`. Current SCRFD alignment also rejects many CFP profile images, so cross-pose training is partly bottlenecked by detection/alignment.
- Conclusion: keep `student_distill_w1_pairft_balanced` as the current S2 deployment candidate. Further improvement should optimize the single-threshold objective directly and/or fix profile-face alignment before more pair fine-tuning.

Profile fallback S2 result:
- Enabling cropped-face fallback changed CFP-FP evaluation from partial `44s/15d` to full `120s/60d`; each evaluated model used `72` fallback images and had `0` failures.
- New full CFP-FP baseline: `w600k` score `0.705` (`LFW@Thr=81.1%`, `CFP@Thr=67.8%`, gap `13.3%`); `student_distill_w1_pairft_balanced` score `0.690` (`68.9%/69.4%`, gap `0.6%`).
- `official_mobilefacenet/student_distill_w1_pairft_cfp_fb_b/mfn_w1_pairft_128d.int8.tflite`: trained from `balanced` with fallback-aligned LFW + CFP-FP splits 02-10, `distill=0.8`, `positive=1.0`, `negative=12.0`, `margin=0.03`. Full evaluation: LFW `sep=0.2621 acc=81.7%`; CFP-FP `sep=0.0973 acc=73.3%`; single-threshold score `0.700`, threshold `0.0138`, LFW@Thr `69.4%`, CFP@Thr `71.7%`, gap `2.2%`.
- Vela for `cfp_fb_b`: `599.83 KiB` SRAM, `1056.80 KiB` flash, `CPU ops=0`, `NPU=100%`.
- Conclusion: `cfp_fb_b` is the best current 128D/SRAM candidate under the complete CFP-FP fallback evaluation. It is slightly below w600k score but much more balanced across LFW/CFP and remains well under 1 MiB SRAM.

Threshold-aware S2 result:
- `train_mfn_student_pair_finetune.py` supports `--threshold-weight`, `--threshold`, and `--threshold-margin`. This adds a hinge loss that pushes positive pairs above `threshold + margin` and negative pairs below `threshold - margin`.
- `official_mobilefacenet/student_distill_w1_pairft_thr_a/mfn_w1_pairft_128d.int8.tflite`: continued from `cfp_fb_b` for 6 epochs with fallback-aligned LFW + CFP-FP splits 02-10, `threshold_weight=0.5`, `threshold=0.02`, `threshold_margin=0.04`, `distill=0.8`, `positive=1.0`, `negative=12.0`, `margin=0.03`. Full evaluation: LFW `sep=0.2821 acc=83.3%`; CFP-FP `sep=0.0965 acc=72.8%`; single-threshold score `0.714`, threshold `0.0463`, LFW@Thr `71.1%`, CFP@Thr `72.2%`, gap `1.1%`.
- Vela for `thr_a`: `599.83 KiB` SRAM, `1056.88 KiB` flash, `CPU ops=0`, `NPU=100%`.
- Current ranking under complete fallback evaluation: `thr_a` score `0.714`; w600k score `0.705`; `cfp_fb_b` score `0.700`. `thr_a` is the best current 128D candidate.

Teacher-pair + hard-negative S2 result:
- `train_mfn_student_pair_finetune.py` supports `--teacher-pair-weight` to match w600k 512D pair similarities and `--mine-hard-negatives` to append high-similarity different-identity pairs mined from the initial student. In this run, mining selected `1200` negatives with student similarity from `0.8238` down to `0.7003`.
- `official_mobilefacenet/student_distill_w1_pairft_tpair_hn_a/mfn_w1_pairft_128d.int8.tflite`: continued from `thr_a` for 6 epochs with `teacher_pair_weight=0.5`, `mine_hard_negatives=1200`, threshold-aware loss unchanged. Full evaluation: LFW `sep=0.2066 acc=81.1%`; CFP-FP `sep=0.0831 acc=71.7%`; single-threshold score `0.725`, threshold `0.1064`, LFW@Thr `73.3%`, CFP@Thr `72.2%`, gap `1.1%`.
- Vela for `tpair_hn_a`: `599.84 KiB` SRAM, `1056.86 KiB` flash, `CPU ops=0`, `NPU=100%`.
- Current ranking under complete fallback evaluation: `tpair_hn_a` score `0.725`; `thr_a` score `0.714`; w600k score `0.705`. `tpair_hn_a` is the best current 128D candidate.

ArcFace fine-tuning path:
- `train_mfn_student_pair_finetune.py` supports a training-only ArcFace identity head via `--arcface-weight`, `--arcface-scale`, `--arcface-margin`, and `--arcface-min-images`. Identities are inferred from LFW/CFP paths; identities with fewer than `--arcface-min-images` aligned images are ignored by the ArcFace loss.
- The ArcFace classifier weights are not part of the exported model. They are used only to shape the student embedding space during training, so exported SRAM/flash should remain governed by the same compact student backbone.
- `run_s2_w1_pairft_arcface_remote.sh` continues from `tpair_hn_a` and tries conservative ArcFace weights.
- Result: ArcFace trained correctly but did not improve this local benchmark. `arc_a` used `arcface_weight=0.05`, `margin=0.25`; `arc_b` used `arcface_weight=0.10`, `margin=0.35`. Both used `1046` ArcFace classes and `3636/5103` aligned images. Losses decreased across 8 epochs for both runs.
- Full evaluation command: `uv run python evaluate_embedding_models.py --max-pairs 120 --model w600k=official_mobilefacenet/w600k_mbf_int8.tflite --model tpair_hn=official_mobilefacenet/student_distill_w1_pairft_tpair_hn_a/mfn_w1_pairft_128d.int8.tflite --model arc_a=official_mobilefacenet/student_distill_w1_pairft_arc_a/mfn_w1_pairft_128d.int8.tflite --model arc_b=official_mobilefacenet/student_distill_w1_pairft_arc_b/mfn_w1_pairft_128d.int8.tflite`.
- `arc_a`: LFW `sep=0.2034 acc=80.6%`; CFP-FP `sep=0.0735 acc=70.6%`; single-threshold score `0.681`; LFW EER `23.3%`; CFP-FP EER `41.7%`.
- `arc_b`: LFW `sep=0.2061 acc=80.0%`; CFP-FP `sep=0.0728 acc=72.2%`; single-threshold score `0.684`; LFW EER `23.3%`; CFP-FP EER `40.0%`.
- Vela: both ArcFace exports remain `599.84 KiB` SRAM, `CPU ops=0`, `NPU=100%`; flash is `1056.25 KiB` for `arc_a` and `1055.92 KiB` for `arc_b`.
- Conclusion: keep `tpair_hn_a` as the current best 128D candidate. ArcFace is wired into the pipeline, but with the current small/narrow LFW+CFP identity set it shifts the similarity distribution in the wrong direction. Use ArcFace again only with broader identity-labeled training data or a stronger teacher-generated identity/pseudo-label set.

Glint360K subset experiment:
- Download command used by `run_s2_w1_pairft_glint_arcface_remote.sh`: `uv run python download_glint360k_subset.py --output-dir datasets/glint360k_subset_112 --start-shard 0 --num-shards 8 --max-images 50000 --max-images-per-id 20`. This produced `50000` aligned images across `39574` identities on WSL2.
- The first 50k-image training attempt generated a `2.0 GiB` aligned cache and `108 MiB` teacher cache, then exited around GPU initialization. The current `train_mfn_student_pair_finetune.py` keeps the full image tensor as a TensorFlow constant; larger identity runs need a streaming dataset implementation before using 50k/100k images safely.
- `official_mobilefacenet/student_distill_w1_pairft_glint_arc_c/mfn_w1_pairft_128d.int8.tflite`: trained from `tpair_hn_a` with `10000` Glint images, plus LFW + CFP-FP train splits 02-10. It used `15103` total images, `1562` ArcFace classes, and `4710/15103` images eligible for ArcFace (`min_images=2`). Training loss decreased from `3.81251` to `2.61406`; identity ArcFace loss decreased from `18.83940` to `16.91416`.
- Evaluation command: `uv run python evaluate_embedding_models.py --max-pairs 120 --model w600k=official_mobilefacenet/w600k_mbf_int8.tflite --model tpair_hn=official_mobilefacenet/student_distill_w1_pairft_tpair_hn_a/mfn_w1_pairft_128d.int8.tflite --model glint_arc_c=official_mobilefacenet/student_distill_w1_pairft_glint_arc_c/mfn_w1_pairft_128d.int8.tflite`.
- `glint_arc_c` local metrics: LFW `sep=0.2215 acc=77.2%`; CFP-FP `sep=0.0719 acc=70.6%`; single-threshold score `0.712` versus `tpair_hn_a` score `0.725`.
- Low-FAR changes versus `tpair_hn_a`: LFW TAR@FAR5 improved `52.5% -> 57.5%`, LFW TAR@FAR1 improved `29.2% -> 44.2%`; CFP-FP TAR@FAR5 improved `12.5% -> 17.5%`, CFP-FP TAR@FAR1 improved `3.3% -> 4.2%`.
- Vela for `glint_arc_c`: `599.84 KiB` SRAM, `1055.95 KiB` flash, `CPU ops=0`, `NPU=100%`.
- Conclusion: `glint_arc_c` is not the new best balanced candidate, but it is the first run where extra identity data improves low-FAR behavior. Keep `tpair_hn_a` as the current deployment/experience candidate; continue with Glint data after changing training to stream identity batches and tuning the ArcFace/threshold loss balance.

Production feasibility metrics:
- Command: `uv run python evaluate_embedding_models.py --max-pairs 120 --model w600k=official_mobilefacenet/w600k_mbf_int8.tflite --model thr_a=official_mobilefacenet/student_distill_w1_pairft_thr_a/mfn_w1_pairft_128d.int8.tflite --model tpair_hn=official_mobilefacenet/student_distill_w1_pairft_tpair_hn_a/mfn_w1_pairft_128d.int8.tflite`.
- LFW verification: w600k EER `3.3%`, TAR@FAR5 `96.7%`; `thr_a` EER `20.0%`, TAR@FAR5 `67.5%`; `tpair_hn_a` EER `22.9%`, TAR@FAR5 `52.5%`.
- CFP-FP verification: w600k EER `36.7%`, TAR@FAR5 `26.7%`; `thr_a` EER `38.3%`, TAR@FAR5 `24.2%`; `tpair_hn_a` EER `35.0%`, TAR@FAR5 `12.5%`.
- At the balanced shared threshold, false-accept rates are high for all tested models: `tpair_hn_a` LFW FAR `75.0%`, CFP-FP FAR `55.0%`; w600k LFW FAR `56.7%`, CFP-FP FAR `63.3%`.
- Feasibility conclusion: `tpair_hn_a` is useful as a low-SRAM balanced-experience candidate, but it is not production-ready for low-FAR/security-sensitive recognition. The next production gate needs a real business validation set and explicit FAR targets; the current local LFW/CFP subsets show that balanced accuracy can improve while false-accept risk remains too high.

## SCRFD QAT Training

For improved SCRFD detection accuracy, use Quantization-Aware Training:

```bash
cd scrfd/quantization

# Quick test (10 min)
NUM_IMAGES=5000 EPOCHS=5 ./run_qat_enhanced.sh

# Standard training (45 min, recommended)
NUM_IMAGES=30000 EPOCHS=10 ./run_qat_enhanced.sh

# Full training (2 hours)
NUM_IMAGES=50000 EPOCHS=15 ./run_qat_enhanced.sh
```

See `scrfd/quantization/README.md` for detailed QAT documentation.

## Flashing Models

```bash
# Flash with firmware
./build_and_flash.sh

# Or manually:
python3 xmodem/xmodem_send.py \
  --port=/dev/tty.usbmodem* \
  --baudrate=921600 \
  --file=we2_image_gen_local/output_case1_sec_wlcsp/output.img \
  --model="model_zoo/tflm_face_embedding/scrfd/models/scrfd_500m_kps_int8_vela.tflite 0x200000 0x0" \
  --model="model_zoo/tflm_face_embedding/ghostfacenet/models/ghostfacenet_fixed_int8_vela.tflite 0x400000 0x0"
```

## Troubleshooting

See `scrfd/quantization/README.md` for common issues:
- Q1: Calibration data format requirements
- Q2: ONNX opset version compatibility
- Q3: Input value range normalization
- Q4-Q10: Various conversion and quantization issues
