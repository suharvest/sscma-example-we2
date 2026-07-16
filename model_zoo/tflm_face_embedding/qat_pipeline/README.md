# QAT Pipeline — distill_v2 ReLU6 128D face embedding

Production training pipeline for the currently-deployed `sscma_face` embedding model.
Scripts pulled from the training run at `spark:~/project/qat-mfn/training/`.
The output of this pipeline is `../qat_distill_v2_relu6_128d/` (the deployed model).

## Environment
- Runs on **spark** (GB10 box) with `uv` + **torch / cu130**.
- Data: ImageFolder of aligned 112×112 face JPGs (glint360k subset).
- Calibration: `../calibration_data/qat_112/` (representative dataset for INT8).

## What this pipeline produces
A MobileFaceNet (scale=1, blocks=(1,4,6,2)) student whose activations were swapped
**LeakyReLU → ReLU6** (bounded activations quantize far better on Ethos-U55 per-tensor
int8), QAT-fine-tuned from the distill_v2 float weights, exported to INT8 TFLite via a
clamp path, and Vela-compiled to 100% NPU. Deployed as `model_128d.int8_vela.tflite`.

## Flow (high level)
```
distill_mfn.py            ResNet100 teacher -> MobileFaceNet 512D student (ArcFace + MSE distill)
      │                   (distill_v2 float weights)
      ▼
train_128d.py             512D student -> 128D head
      │
      ▼
qat_finetune.py           QAT fine-tune (self-distillation, per-tensor int8 FakeQuantize
qat_finetune_if.py        after every block). *_if = ImageFolder variant for spark.
      │                   LeakyReLU -> ReLU6 swap lives here; ReLU6 clamps the range so
      │                   TFLite's representative-dataset min/max stays tight (no PTQ collapse).
      ▼
export_qat_128d_clamp.py  QAT .pt -> clean model w/ hard torch.clamp at learned FQ ranges
      │                   -> ONNX  (robust path; *_fq.py is the alternate QDQ/FakeQuant export)
      ▼
quantize_and_vela.py      ONNX --onnx2tf--> TF SavedModel --tf.lite INT8--> vela (ethos-u55-64)
                          -> ../qat_distill_v2_relu6_128d/model_128d.int8_vela.tflite
```

## Entry scripts
- **QAT fine-tune (main entry on spark):** `qat_finetune_if.py`
  (wraps `qat_finetune.py`, which holds the model def + FakeQuantize insertion).
- **Distill from teacher (if regenerating float weights):** `distill_mfn.py`
- **128D head:** `train_128d.py`
- **Export to INT8:** `export_qat_128d_clamp.py` (clamp path, recommended);
  `export_qat_128d_fq.py` (FakeQuant/QDQ alternate); `export_128d.py` (plain float export).
- **Quantize + Vela:** `quantize_and_vela.py`

## Config / model definition
- `backbones.py`, `mfn_cfg.py`, `config_mfn128.py` — MobileFaceNet architecture + config.

## Evaluation
- `eval_tflite_full.py` — full LFW / CFP-FP eval of the INT8 TFLite.
- `office4_eval.py` — office/customer-collected face set eval.
- `torch_eval_qat_lfw.py` — LFW eval of the QAT `.pt` (float/int8-simulated) before export.

## Notes
- Vela `*_vela.tflite` contains the `ethos-u` custom op and cannot run on PC TFLite;
  evaluate the pre-Vela INT8 for accuracy, then check the Vela summary CSV for SRAM/NPU.
- Deployed model specs and provenance: see `../qat_distill_v2_relu6_128d/README.md`.
