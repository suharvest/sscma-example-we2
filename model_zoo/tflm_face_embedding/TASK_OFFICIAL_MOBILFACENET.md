# Task: Replace foamliu MobileFaceNet with InsightFace Official MobileFaceNet

## Background

Current firmware uses **foamliu/MobileFaceNet 128D** (community implementation from https://github.com/foamliu/MobileFaceNet). On-device accuracy is insufficient — different people are incorrectly matched as the same person.

### Root Cause

The foamliu model has poor embedding discriminability. Its MegaFace score is **82.55%** vs official InsightFace's **92.59%** — a 10 percentage point gap on cross-domain face verification. LFW numbers (99.48%) are misleading because LFW is too easy.

Key differences from official:
- Training data: MS-Celeb-1M (3.8M) vs MS1MV2 (5.8M)
- Loss: softmax/focal (no angle margin) vs **ArcFace** (additive angular margin)
- ArcFace explicitly enforces inter-class separation, which directly addresses the false-positive problem

### Evidence

Tested on LFW images with full pipeline (SCRFD detect + align + embed):
- Float32 same-person similarity: 0.55–0.83 (unstable)
- Float32 different-person: often > 0.6 (too high, should be < 0.3)
- INT8 quantization only adds ~2% degradation (not the main cause)

### Previous attempts

GhostFaceNet was tried but abandoned:
- Quantization accuracy dropped too much (QAT didn't help)
- Vela-compiled model caused AllocateTensors hang — tensor graph incompatible with firmware (commit `3bdc43e`)

## Goal

Convert InsightFace official MobileFaceNet to INT8 Vela TFLite, verify quantization accuracy is ≥ 98%, then swap into firmware.

## Prerequisites

Working directory: `model_zoo/tflm_face_embedding/`

```bash
cd model_zoo/tflm_face_embedding
uv sync  # install from pyproject.toml
```

Models (all .tflite files are excluded from git via .gitignore):
- Place under `model_zoo/tflm_face_embedding/official_mobilefacenet/`

## Step 1: Download Official InsightFace MobileFaceNet

**Target:** Get the float32 ONNX model from InsightFace's model zoo.

The official model is at:
- https://github.com/deepinsight/insightface
- Model zoo: https://github.com/deepinsight/insightface/wiki/Model-Zoo
- File: `insightface/models/buffalo_l/w600k_r50.onnx` is the R50 model, but for **MobileFaceNet**:
  - The original is at: insightface/models/ — check for `mobilefacenet.onnx` or search "InsightFace MobileFaceNet ONNX"

**Alternative:** Use ONNX export from the PyTorch checkpoint at:
- https://github.com/deepinsight/insightface (the `recognition/arcface_torch` directory)
- The MobileFaceNet architecture is defined in the paper and can be exported from a trained checkpoint

**Deliverable:** A float32 ONNX file `official_mobilefacenet/mobilefacenet.onnx` with:
- Input: [1, 3, 112, 112] (NCHW) or [1, 112, 112, 3] (NHWC)
- Output: [1, 128]
- Input range: [-1, 1] (normalized)

> Note: Official MobileFaceNet natively outputs **128D** (last conv 1×1 maps to 128). No PCA/dimension reduction needed. The 512D intermediate is just the backbone before the embedding head.

## Step 2: Convert to TFLite Float32

Reference: `foamliu_mobilefacenet_128d/convert.py` (the `fix_onnx_shapes()` and `convert_to_tflite_with_fusion()` functions).

1. Fix dynamic batch to static [1, ...]
2. Convert via `onnx2tf` (handles BatchNorm fusion)
3. Verify all ops are supported (no unfused BatchNorm)

**Deliverable:** `official_mobilefacenet/mobilefacenet_float32.tflite`

## Step 3: INT8 Quantization

Reference: `foamliu_mobilefacenet_128d/convert.py` (the `quantize_to_int8()` function).

Key parameters:
- Representative dataset: `calibration_data/qat_112/` (13233 aligned 112×112 face crops)
- Full integer quantization: `tf.lite.OpsSet.TFLITE_BUILTINS_INT8`
- Input/output both INT8
- Per-channel quantization enabled

**Deliverable:** `official_mobilefacenet/mobilefacenet_qat_int8.tflite`

## Step 4: Validate Quantization Accuracy

Compare float32 vs INT8 embeddings on calibration images:

```python
# For each image in calibration set:
#   1. Run float32 model → embedding_f32
#   2. Run INT8 model → dequantize → embedding_int8
#   3. Compute cosine similarity(embedding_f32, embedding_int8)
#
# Mean cosine similarity MUST be ≥ 0.98 (98%) to PASS.
```

Also test discriminability:
- Same person (different images): cosine should be ≥ 0.6
- Different person: cosine should be ≤ 0.3
- Separation (same_mean - diff_mean) should be ≥ 0.3

**Deliverable:** Validation report with similarity numbers.

## Step 5: Vela Compilation

```bash
vela official_mobilefacenet/mobilefacenet_qat_int8.tflite \
  --accelerator-config ethos-u55-64 \
  --optimise Performance \
  --output-dir official_mobilefacenet/
```

Verify:
- [ ] CPU operators = 0 (100% NPU)
- [ ] Tensor arena size ≤ 700 KB (firmware's MOBILEFACENET_ARENA_SIZE)
- [ ] No Vela errors or warnings

**Deliverable:** `official_mobilefacenet/mobilefacenet_qat_int8_vela.tflite`

## Step 6: Firmware Integration

Files to modify in `EPII_CM55M_APP_S/app/scenario_app/sscma_face/`:

1. **Verify model input dimensions match:**
   - Input shape: [1, 112, 112, 3], INT8
   - Output shape: [1, 128], INT8
   - If quantization params differ from foamliu, update preprocessing in `cvapp_face_embedding.cpp`

2. **`common_config.h`** — update if arena size or model address changes:
   ```c
   #define MOBILEFACENET_MODEL_FLASH_ADDR  (BASE_ADDR_FLASH1_R_ALIAS + 0x510000)
   // MOBILEFACENET_ARENA_SIZE stays at 700 KB unless Vela reports different
   ```

3. **`build_and_flash.sh`** — update model path:
   ```bash
   EMBEDDING_MODEL="${PROJECT_ROOT}/model_zoo/tflm_face_embedding/official_mobilefacenet/mobilefacenet_qat_int8_vela.tflite"
   ```

4. **Fix quantization preprocessing** if input quant params differ from foamliu (discovered during investigation):

   Current firmware hardcodes `pixel - 128` for INT8 input conversion. If the official model has different `scale`/`zp`, this MUST be fixed:

   ```c
   // In cvapp_face_embedding.cpp, near line 770-780
   // Replace hardcoded:
   //   dst[i] = (int8_t)((int)src[i] - 128);
   // With model-aware quantization:
   float scale = emb_input->params.scale;
   int32_t zp = emb_input->params.zero_point;
   for (int i = 0; i < aligned_face_buffer_size; i++) {
       int32_t val = (int32_t)round((float)((int)src[i] - 128) / (127.5f * scale)) + zp;
       if (val < -128) val = -128;
       if (val > 127) val = 127;
       dst[i] = (int8_t)val;
   }
   ```

5. **`cvapp_face_embedding.h`** — update comment referencing foamliu → InsightFace

## Success Criteria

- [ ] INT8 model loads and runs without errors on PC (TFLite interpreter)
- [ ] Cosine similarity(float32, INT8) ≥ 0.98 on calibration set
- [ ] Same-person vs different-person separation ≥ 0.3
- [ ] Vela compilation: 100% NPU, no CPU ops
- [ ] Tensor arena fits in 700 KB
- [ ] Firmware builds successfully with new model
- [ ] `build_and_flash.sh` flashes correct model at correct address

## Reference Files

| File | Purpose |
|------|---------|
| `foamliu_mobilefacenet_128d/convert.py` | Reference for conversion pipeline |
| `foamliu_mobilefacenet_128d/README.md` | foamliu model doc, architecture details |
| `compute_embedding.py` | PC-side pipeline for validation testing |
| `calibration_data/qat_112/` | 13233 aligned face crops for INT8 calibration |
| `datasets/lfw/` | LFW images for discriminability validation |
| `../../../EPII_CM55M_APP_S/app/scenario_app/sscma_face/common_config.h` | Firmware model config |
| `../../../EPII_CM55M_APP_S/app/scenario_app/sscma_face/cvapp_face_embedding.cpp` | Firmware embedding pipeline |
| `../../../build_and_flash.sh` | Flash script with model paths |

## Environment

```bash
# Python venv is already set up
cd model_zoo/tflm_face_embedding
source .venv/bin/activate
# or: uv run python <script>.py
```

Key dependencies (already installed):
- tensorflow==2.19.0
- onnx==1.16.0
- onnx2tf==1.25.0
- onnxruntime==1.20.1
- opencv-python==4.9.0.80

Vela:
```bash
pip install ethos-u-vela
```
