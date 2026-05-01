import sys, os
BASE = '/Users/harvest/project/grove_vision_2/sscma-example-we2/model_zoo/tflm_face_embedding'
os.chdir(BASE)
sys.path.insert(0, BASE)
from compute_embedding import (FaceEmbeddingPipeline, cosine_similarity, l2_normalize,
                                image_to_bgr_planar, direct_resize_bgr_planar_to_rgb,
                                quantize_uint8_to_int8, scrfd_decode_and_nms, get_best_face,
                                compute_face_alignment, apply_face_alignment)
import numpy as np
from pathlib import Path
from PIL import Image
import random, tensorflow as tf

random.seed(42)

# ===== Load LFW pairs =====
lfw = Path('datasets/lfw')
people = sorted([d for d in lfw.iterdir() if d.is_dir() and len(list(d.glob('*.jpg'))) >= 2])
tp = random.sample(people, min(30, len(people)))

same_pairs, diff_pairs = [], []
for p in tp:
    imgs = sorted(p.glob('*.jpg'))[:2]
    if len(imgs)==2: same_pairs.append((str(imgs[0]), str(imgs[1])))
for i in range(0, len(tp)-1, 2):
    a, b = sorted(tp[i].glob('*.jpg')), sorted(tp[i+1].glob('*.jpg'))
    if a and b: diff_pairs.append((str(a[0]), str(b[0])))

print(f"LFW: {len(same_pairs)} same, {len(diff_pairs)} diff pairs\n")

# ===== Test 3 QAT model variants =====
models = {
    'QAT-TFLite-F32':  'official_mobilefacenet/mobilefacenet_qat_float32.tflite',
    'QAT-TFLite-INT8': 'official_mobilefacenet/mobilefacenet_qat_int8.tflite',
    'QAT-Vela':        'official_mobilefacenet/mobilefacenet_qat_int8_vela.tflite',
}

# Test each model
for mname, mpath in models.items():
    scrfd = 'scrfd/models/scrfd_500m_kps_int8.tflite'
    
    if 'Vela' in mname:
        # Vela model can't run on PC TFLite (has Ethos-U custom op)
        print(f"{mname}: SKIP (Vela model can't run on PC, NPU-only)")
        continue
    
    try:
        pipeline = FaceEmbeddingPipeline(scrfd, mpath, backend='tflite')
    except Exception as e:
        print(f"{mname}: LOAD ERROR: {e}")
        continue
    
    same_s, diff_s = [], []
    fail_s, fail_d = 0, 0
    
    for a, b in same_pairs:
        try:
            r1 = pipeline.compute(Image.open(a), debug=False)
            r2 = pipeline.compute(Image.open(b), debug=False)
            same_s.append(cosine_similarity(r1['embedding'], r2['embedding']))
        except: fail_s += 1
    
    for a, b in diff_pairs:
        try:
            r1 = pipeline.compute(Image.open(a), debug=False)
            r2 = pipeline.compute(Image.open(b), debug=False)
            diff_s.append(cosine_similarity(r1['embedding'], r2['embedding']))
        except: fail_d += 1
    
    if same_s and diff_s:
        sm, dm = np.mean(same_s), np.mean(diff_s)
        sep = sm - dm
        best = max((np.sum(np.array(same_s)>=t) + np.sum(np.array(diff_s)<t)) / (len(same_s)+len(diff_s))
                  for t in np.linspace(-1, 1, 200))
        print(f"{mname:<20}: sep={sep:.4f}  acc={best*100:.1f}%  "
              f"same={sm:.4f}  diff={dm:.4f}  "
              f"s_min={np.min(same_s):.4f}  d_max={np.max(diff_s):.4f}  "
              f"N={len(same_s)}s/{len(diff_s)}d (fail {fail_s}/{fail_d})")

# ===== Device simulation: what the firmware ACTUALLY does =====
print(f"\n--- Device firmware preprocessing simulation ---")

# Load the QAT INT8 model directly (bypass FaceEmbeddingPipeline,
# simulate exact firmware preprocessing)
i8 = tf.lite.Interpreter(model_path='official_mobilefacenet/mobilefacenet_qat_int8.tflite')
i8.allocate_tensors()
emb_inp = i8.get_input_details()[0]
emb_out = i8.get_output_details()[0]
emb_in_q = emb_inp.get('quantization_parameters', {})
emb_in_zp = int(np.asarray(emb_in_q.get('zero_points', [-128])).flat[0])
emb_in_scale = float(np.asarray(emb_in_q.get('scales', [1.0])).flat[0])
emb_out_q = emb_out.get('quantization_parameters', {})
emb_out_scale = float(np.asarray(emb_out_q.get('scales', [1.0])).flat[0])
emb_out_zp = int(np.asarray(emb_out_q.get('zero_points', [0])).flat[0])

print(f"Model input:  zp={emb_in_zp}, scale={emb_in_scale:.8f}")
print(f"Model output: zp={emb_out_zp}, scale={emb_out_scale:.8f}")

# Simulate exact firmware path:
# 1. SCRFD detects + aligns → aligned_face [0,255] uint8 RGB
# 2. Firmware: dst[i] = src[i] - 129  (for zp=-1)
#    - pixel=0: int8=-129→clamped to -128
#    - pixel=128: int8=-1  
#    - pixel=255: int8=126
# This is ~pixel-128 but offset by 1 for zp=-1

def firmware_preprocess(aligned_face_uint8):
    """Exact firmware preprocessing: pixel - (128 - zp)"""
    # For zp=-1: int8 = pixel - 128 - 1 = pixel - 129
    # Actually: float = (pixel - 127.5)/127.5
    # int8 = round(float/scale + zp)
    # For scale=1/127.5, zp=-1:
    # int8 = round((pixel-127.5) + (-1)) = round(pixel - 128.5) = pixel - 129 (for pixel>0)
    # But firmware does: dst[i] = src[i] - 129 (for zp=-1)
    val = aligned_face_uint8.astype(np.int32) - 129  # matches zp=-1
    return np.clip(val, -128, 127).astype(np.int8)

def firmware_dequantize(int8_emb):
    """Firmware dequantization: (int8 - zp) * scale"""
    return (int8_emb.astype(np.float32) - emb_out_zp) * emb_out_scale

# Test with pipeline
pipeline = FaceEmbeddingPipeline('scrfd/models/scrfd_500m_kps_int8.tflite',
                                  'official_mobilefacenet/mobilefacenet_qat_int8.tflite',
                                  backend='tflite')

same_s_dev, diff_s_dev = [], []
same_s_pc, diff_s_pc = [], []

for a, b in same_pairs:
    try:
        # PC pipeline path (correct TFLite quantization)
        r1 = pipeline.compute(Image.open(a), debug=False)
        r2 = pipeline.compute(Image.open(b), debug=False)
        same_s_pc.append(cosine_similarity(r1['embedding'], r2['embedding']))
        
        # Device firmware path (pixel-129 quantization)
        aligned1 = r1['aligned_face']
        aligned2 = r2['aligned_face']
        dev_in1 = firmware_preprocess(aligned1)
        dev_in2 = firmware_preprocess(aligned2)
        
        i8.set_tensor(emb_inp['index'], np.expand_dims(dev_in1, 0))
        i8.invoke()
        dev_emb1 = i8.get_tensor(emb_out['index'])[0]
        dev_emb1 = firmware_dequantize(dev_emb1)
        dev_emb1 = l2_normalize(dev_emb1.flatten()[:128])
        
        i8.set_tensor(emb_inp['index'], np.expand_dims(dev_in2, 0))
        i8.invoke()
        dev_emb2 = i8.get_tensor(emb_out['index'])[0]
        dev_emb2 = firmware_dequantize(dev_emb2)
        dev_emb2 = l2_normalize(dev_emb2.flatten()[:128])
        
        same_s_dev.append(cosine_similarity(dev_emb1, dev_emb2))
    except: pass

for a, b in diff_pairs:
    try:
        r1 = pipeline.compute(Image.open(a), debug=False)
        r2 = pipeline.compute(Image.open(b), debug=False)
        diff_s_pc.append(cosine_similarity(r1['embedding'], r2['embedding']))
        
        dev_in1 = firmware_preprocess(r1['aligned_face'])
        dev_in2 = firmware_preprocess(r2['aligned_face'])
        i8.set_tensor(emb_inp['index'], np.expand_dims(dev_in1, 0)); i8.invoke()
        dev_emb1 = l2_normalize(firmware_dequantize(i8.get_tensor(emb_out['index'])[0]).flatten()[:128])
        i8.set_tensor(emb_inp['index'], np.expand_dims(dev_in2, 0)); i8.invoke()
        dev_emb2 = l2_normalize(firmware_dequantize(i8.get_tensor(emb_out['index'])[0]).flatten()[:128])
        diff_s_dev.append(cosine_similarity(dev_emb1, dev_emb2))
    except: pass

print(f"\n{'Method':<25} {'Sep':>8} {'Same':>8} {'Diff':>8} {'S_min':>8} {'D_max':>8}")
print("-"*70)

for label, ss, ds in [('PC Pipeline (correct)', same_s_pc, diff_s_pc),
                        ('Device firmware (pixel-129)', same_s_dev, diff_s_dev)]:
    if ss and ds:
        sm, dm = np.mean(ss), np.mean(ds)
        print(f"{label:<25} {sm-dm:>8.4f} {sm:>8.4f} {dm:>8.4f} {np.min(ss):>8.4f} {np.max(ds):>8.4f}")

# Compare individual embedding differences
print(f"\n--- PC vs Device embedding comparison ---")
# Take first 5 same pairs, compare embeddings side by side
for i, (a, b) in enumerate(same_pairs[:3]):
    try:
        r1 = pipeline.compute(Image.open(a), debug=False)
        r2 = pipeline.compute(Image.open(b), debug=False)
        
        # PC path
        pc_emb1 = r1['embedding']
        
        # Device path  
        dev_in1 = firmware_preprocess(r1['aligned_face'])
        i8.set_tensor(emb_inp['index'], np.expand_dims(dev_in1, 0)); i8.invoke()
        dev_emb1 = l2_normalize(firmware_dequantize(i8.get_tensor(emb_out['index'])[0]).flatten()[:128])
        
        cos_pc_dev = cosine_similarity(pc_emb1, dev_emb1)
        print(f"Pair {i} img1: PC vs Device cosine = {cos_pc_dev:.4f}")
    except Exception as e:
        print(f"Pair {i}: {e}")

print("\nDone!")
