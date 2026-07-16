# QAT distill_v2 ReLU6 128D — deployed face embedding model

This is the current production face embedding model for `sscma_face` (Grove Vision
AI V2 / SenseCap Watcher). It replaces the previous `qat_distilled_128d`
(w600k-line PCA) model, which had weak discrimination and caused the customer's
"认错人" (stranger false-accept) complaint.

## Files
| File | Purpose |
|---|---|
| `model_128d.int8_vela.tflite` | **Deployed** — flashed to `0x510000`, runs on Ethos-U55 NPU |
| `model_128d.int8.tflite` | Pre-Vela INT8 — for PC-side evaluation (Vela version has the ethos-u custom op and cannot run on PC TFLite) |
| `model_128d.onnx` | FP32 export (pre-quantization) |
| `model_qat_relu6_best.pt` | QAT checkpoint (reproducibility) |
| `model_128d.int8_summary_*.csv` | Vela compilation summary |

## Specs
- Input: **112×112×3 RGB int8**, quant `(scale=0.007843137, zp=-1)` — identical to
  the old model, so firmware `pixel-129` preprocessing is unchanged. **Drop-in.**
- Output: **128-D int8**, quant `(scale=0.012660, zp=18)` — different scale from the
  old model, but firmware reads scale/zp from the tensor and dequantizes, so this is
  handled automatically.
- Vela (ethos-u55-64): **100% NPU (0 CPU ops)**, SRAM **596.84 KiB** (≤620 budget),
  Flash ~1008 KiB.

## Provenance
- Teacher: InsightFace glint360k_r100 (via distill_v2 512D float student).
- Student: MobileFaceNet scale=1 blocks=(1,4,6,2), **activations swapped
  LeakyReLU → ReLU6** (bounded activations quantize far better on Ethos-U55's
  per-tensor int8; also fused into conv, fewer NPU ops).
- QAT fine-tune from distill_v2 weights, then INT8 export with the QAT-learned
  ranges hard-clamped so the deployable INT8 keeps the QAT discrimination (the
  earlier PTQ export collapsed here).
- Production scripts: `../qat_pipeline/` (pulled from the spark training run).

## Accuracy (INT8, measured)
| | LFW acc | genuine mean | impostor mean | office4 impostor | CFP acc |
|---|---|---|---|---|---|
| **this model** | 99.33% | 0.608 | **0.005** | **0.045 (on-device Vela)** | 94.26% |
| old (qat_distilled_128d) | 98.93% | 0.699 | 0.237 | 0.25 (on-device) | 91.54% |

Impostor (stranger similarity) drops ~50× on LFW and ~5.6× on-device. On-device
Vela reproduces the PC embedding (cos 0.99), so the on-device numbers are faithful.

## ⚠️ Match threshold — change 0.4 → ~0.30
The old model's embedding scale put the useful match threshold near **0.4**. This
model centers impostors at ~0 and genuine at ~0.6, so the operating point moves:
- genuine p5 = 0.385, impostor p99 = 0.231 → a threshold of **~0.30** cleanly
  separates them (accept most genuine, reject ~all strangers).
- **The match threshold lives on the ESP32 / host side, not in this firmware repo.**
  When deploying this model, set the host-side cosine match threshold to ~0.30
  (was 0.4). Leaving it at 0.4 will falsely reject some genuine matches.

Re-enroll on-device (device camera), not from uploaded photos: the device-camera
imaging domain shifts embeddings away from clean photos, so photo-enrolled
templates match poorly (genuine ~0.25). Device-enroll + device-recognize works.
