# SCRFD INT8 Evaluation Report

Date: 2026-05-04

## Scope

Evaluated the existing SCRFD INT8 simulation / accuracy toolchain around:

- Production PTQ INT8 model: `scrfd/models/scrfd_500m_kps_int8.tflite`
- QAT v5 sigmoid INT8 model: `scrfd/quantization/qat_tflite_output/scrfd_qat_v5_full_integer_quant.tflite`
- QAT v5 nosigmoid INT8 model: `scrfd/quantization/scrfd_qat_v5_nosigmoid_int8.tflite`
- Float32 / INT8 consistency for the available QAT float32 and full-integer TFLite pairs
- Firmware-equivalent SCRFD decoding in `compute_embedding.py` and `scrfd_postprocessing.cc`

Dataset used for the local proxy evaluation:

- `calibration_data/qat_160`
- First 1000 images for detection-rate proxy
- First 200 images for output distribution and float-vs-int8 correlation

The dataset is made of face crops, so this is a firmware-pipeline sanity check and quantization consistency evaluation, not a full WIDER FACE mAP benchmark.

## Findings

1. The production PTQ INT8 SCRFD model does not show a general low-confidence problem in PC simulation.

   At threshold `0.70`, the production model detects a valid best face in `97.0%` of the 1000 sampled crops. Best-face confidence mean is `0.873`; p10 is `0.840`.

2. The dominant production detections come from stride 16.

   Production model mean max score by stride:

   | Model | s8 | s16 | s32 |
   |---|---:|---:|---:|
   | PTQ INT8 production | 0.141 | 0.870 | 0.039 |

   This means low s8 / s32 confidence is expected for these crop-like inputs. A global log line such as `max_score: s8=... s16=... s32=...` should be judged primarily by the best stride, not by all strides being high.

3. QAT v5 score heads have a real stride-32 issue.

   Both QAT v5 sigmoid and nosigmoid INT8 models produced stride-32 max score `0.000` over the tested samples. Tensor metadata confirms the nosigmoid INT8 stride-32 score output has an unusably tiny scale:

   `shape=[50,1], dtype=int8, quant=(7.84313680668447e-09, 0)`

   This branch is effectively collapsed. It may not hurt face-crop tests, but it will hurt large-face / low-resolution detection cases where stride 32 matters.

4. Existing PC analysis script had a decode mismatch.

   `analyze_scrfd_quantization.py` decoded bboxes using `(x + 0.5) * stride`, while `SCRFD_DECODING.md`, `compute_embedding.py`, and firmware `scrfd_postprocessing.cc` use anchor corner `x * stride`. This makes PC visualization / accuracy analysis disagree with device behavior. The script has been fixed to use anchor corner.

5. The available QAT float32-vs-INT8 pairs are highly correlated.

   Output correlation on 200 sampled images:

   | Pair | Scores s8 | Scores s16 | BBox range | KPS range |
   |---|---:|---:|---:|---:|
   | QAT v5 nosigmoid float vs int8 | 0.9958 | 0.9990 | 0.9968-0.99997 | 0.9961-0.99975 |
   | QAT v5 sigmoid float vs int8 | 0.9972 | 0.9993 | 0.9982-0.99996 | 0.9981-0.99977 |

   Stride-32 score correlation is undefined because both float and int8 outputs are constant zero.

## Detection Proxy Results

Valid best face means NMS result has best face size >= `MIN_FACE_SIZE` (`40px`).

| Model | Threshold | Detect Rate | Avg Dets | Avg Best Score | p10 Best Score |
|---|---:|---:|---:|---:|---:|
| PTQ INT8 production | 0.30 | 96.9% | 1.18 | 0.871 | 0.840 |
| PTQ INT8 production | 0.50 | 96.9% | 1.12 | 0.873 | 0.840 |
| PTQ INT8 production | 0.70 | 97.0% | 1.05 | 0.873 | 0.840 |
| PTQ INT8 production | 0.85 | 81.4% | 0.82 | 0.881 | 0.863 |
| QAT v5 sigmoid INT8 | 0.70 | 91.9% | 1.06 | 0.874 | 0.757 |
| QAT v5 nosigmoid INT8 | 0.70 | 95.9% | 1.10 | 0.989 | 0.980 |

## Recommendations

1. Keep using `scrfd/models/scrfd_500m_kps_int8.tflite` / Vela equivalent for production unless there is a WIDER FACE mAP result showing QAT is better.

2. Do not ship the current QAT v5 models as a detection-quality improvement. The stride-32 score branch is collapsed and should be fixed before deployment.

3. If device logs show low confidence with the production model, inspect these first:

   - Confirm the flashed model is the PTQ production Vela model, not a QAT nosigmoid experiment.
   - Print `fd_input->params.scale` and `zero_point`; production should be close to `scale=1/255`, `zp=-128`.
   - Dump per-stride output tensor scales/zero points on-device; production scores should be `scale=0.00390625`, `zp=-128`.
   - Confirm preprocessing remains direct resize RGB and `dst = src + zero_point`; this only works cleanly for `[0,1]` input quantization.
   - Judge `max_score` by the best stride. On the tested crop set, stride 16 is the expected high-confidence branch.

4. For a real accuracy claim, add a labeled detector benchmark:

   - WIDER FACE validation subset resized through the exact firmware preprocessing path.
   - Report AP / recall at thresholds `0.3`, `0.5`, `0.7`.
   - Compare float32 TFLite, PTQ INT8 TFLite, and Vela-on-device outputs where possible.

