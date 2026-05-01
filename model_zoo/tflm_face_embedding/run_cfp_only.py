#!/usr/bin/env python3
"""CFP-FP only embedding model comparison (LFW already completed)."""
import sys
sys.path.insert(0, '.')
import compute_embedding
# CFP images are face-cropped (face ~80%+ of image).
# Default MAX_FACE_RATIO=0.6 filters these as "oversized false positives".
# Relax to 1.0 so face-cropped images pass the size filter.
compute_embedding.MAX_FACE_RATIO = 1.0
from compute_embedding import (FaceEmbeddingPipeline, cosine_similarity,
                                l2_normalize)
import numpy as np
from pathlib import Path
from PIL import Image
import time

# ============================================================
# CFP-FP Dataset loader (FIXED path resolution)
# ============================================================

def load_cfp_pairs(max_pairs=80):
    CFP = Path('datasets/cfp/cfp-dataset')

    def load_list(path):
        mapping = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 2:
                        mapping[int(parts[0])] = parts[1]
        return mapping

    frontal_map = load_list(CFP / 'Protocol/Pair_list_F.txt')
    profile_map = load_list(CFP / 'Protocol/Pair_list_P.txt')

    def load_pairs(path):
        pairs = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and ',' in line:
                    f_idx, p_idx = line.split(',')
                    pairs.append((int(f_idx), int(p_idx)))
        return pairs

    split_dir = CFP / 'Protocol/Split/FP'
    same_pairs_raw = load_pairs(split_dir / '01/same.txt')
    diff_pairs_raw = load_pairs(split_dir / '01/diff.txt')

    def idx_to_path(idx, mapping):
        img_path = mapping.get(idx, '')
        if not img_path:
            return None
        # Pair list paths are relative to Protocol/ (e.g., '../Data/Images/001/frontal/01.jpg')
        # Resolve from Protocol/ directory
        full_path = (CFP / 'Protocol' / img_path).resolve()
        return str(full_path) if full_path.exists() else None

    same_pairs = []
    for f_idx, p_idx in same_pairs_raw[:max_pairs]:
        f_path = idx_to_path(f_idx, frontal_map)
        p_path = idx_to_path(p_idx, profile_map)
        if f_path and p_path:
            same_pairs.append((f_path, p_path))

    diff_pairs = []
    for f_idx, p_idx in diff_pairs_raw[:max_pairs]:
        f_path = idx_to_path(f_idx, frontal_map)
        p_path = idx_to_path(p_idx, profile_map)
        if f_path and p_path:
            diff_pairs.append((f_path, p_path))

    return same_pairs[:max_pairs], diff_pairs[:max_pairs//2]

# ============================================================
# Trick implementations
# ============================================================

def flip_aligned_face(aligned_face):
    """Horizontally flip the aligned face (112x112x3 uint8 RGB)."""
    return np.fliplr(aligned_face).copy()

def compute_embedding_with_pipeline(pipeline, image_path, use_tta=False):
    """Run pipeline, optionally with TTA flip augmentation."""
    img = Image.open(image_path)

    result = pipeline.compute(img, debug=False)
    emb_base = result['embedding'].copy()
    aligned = result['aligned_face'].copy()

    if use_tta:
        aligned_flipped = flip_aligned_face(aligned)

        if pipeline.emb_input_dtype == np.int8:
            emb_in_q = pipeline.emb_interp.get_input_details()[0].get("quantization_parameters", {})
            emb_in_zp = int(np.asarray(emb_in_q.get("zero_points", [-128])).flat[0])
            if emb_in_zp == -128:
                emb_input_f = (aligned_flipped.astype(np.int32) - 128).clip(-128, 127).astype(np.int8)
            else:
                emb_in_scale = float(np.asarray(emb_in_q.get("scales", [1.0])).flat[0])
                emb_float = aligned_flipped.astype(np.float32) / 255.0
                emb_input_f = np.clip(np.round(emb_float / emb_in_scale + emb_in_zp), -128, 127).astype(np.int8)
        else:
            emb_input_f = (aligned_flipped.astype(np.float32) / 127.5) - 1.0

        emb_input_f = emb_input_f[np.newaxis, ...]
        pipeline.emb_interp.set_tensor(pipeline.emb_input_idx, emb_input_f)
        pipeline.emb_interp.invoke()
        emb_output_f = pipeline.emb_interp.get_tensor(pipeline.emb_output_idx)

        emb_out_dtype = pipeline.emb_interp.get_output_details()[0]["dtype"]
        if emb_out_dtype in (np.int8, np.uint8):
            emb_flip = (emb_output_f.astype(np.float32) - pipeline.emb_out_zp) * pipeline.emb_out_scale
        else:
            emb_flip = emb_output_f.astype(np.float32)
        emb_flip = emb_flip.flatten()[:128]
        emb_flip = l2_normalize(emb_flip)

        # Average and re-normalize
        emb_avg = l2_normalize(emb_base + emb_flip)
        result['embedding'] = emb_avg

    return result

# ============================================================
# Main evaluation
# ============================================================

def compute_mean_embedding(pipeline, num_samples=200):
    calib = Path('calibration_data/qat_112')
    embs = []
    for p in sorted(calib.glob('*.jpg'))[:num_samples]:
        try:
            result = pipeline.compute(Image.open(str(p)), debug=False)
            embs.append(result['embedding'])
        except Exception:
            pass
    if embs:
        return np.mean(embs, axis=0)
    return np.zeros(128, dtype=np.float32)

models = {
    'Official-F32': ('scrfd/models/scrfd_500m_kps_int8.tflite',
                      'official_mobilefacenet/mobilefacenet_no_bn_float32.tflite'),
    'Official-I8': ('scrfd/models/scrfd_500m_kps_int8.tflite',
                     'official_mobilefacenet/mobilefacenet_no_bn_int8.tflite'),
    'Foamliu-I8': ('scrfd/models/scrfd_500m_kps_int8.tflite',
                    'foamliu_mobilefacenet_128d/foamliu_mobilefacenet_128d_qat_int8.tflite'),
}

tricks = ['BASELINE', 'TTA', 'CENTER', 'TTA+CENTER']
all_results = {}

t_start = time.time()

print(f"\n{'#'*70}")
print(f"# DATASET: CFP-FP (FIXED loader)")
print(f"{'#'*70}")

same_pairs, diff_pairs = load_cfp_pairs(80)
print(f"Same pairs: {len(same_pairs)}, Diff pairs: {len(diff_pairs)}")

if len(same_pairs) == 0:
    print("ERROR: No same pairs found! Debugging...")
    # Debug
    CFP = Path('datasets/cfp/cfp-dataset')
    mapping = {}
    with open(CFP / 'Protocol/Pair_list_F.txt') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 2:
                    mapping[int(parts[0])] = parts[1]
    print(f"  mapping[5] = {mapping.get(5)}")
    resolved = (CFP / 'Protocol' / mapping.get(5, '')).resolve()
    print(f"  resolved = {resolved}")
    print(f"  exists = {resolved.exists()}")
    # Test first split entry
    with open(CFP / 'Protocol/Split/FP/01/same.txt') as f:
        first = f.readline().strip()
    print(f"  first same pair: {first}")
    sys.exit(1)

for model_name, (scrfd_path, emb_path) in models.items():
    print(f"\n  --- {model_name} ---")
    t_model_start = time.time()

    try:
        pipeline = FaceEmbeddingPipeline(scrfd_path, emb_path, backend='tflite')
    except Exception as e:
        print(f"    LOAD ERROR: {e}")
        continue

    mean_emb = compute_mean_embedding(pipeline, 200)
    print(f"    Mean embedding computed from 200 calib images")

    for trick in tricks:
        use_tta = 'TTA' in trick
        use_center = 'CENTER' in trick

        same_sims = []
        fail_same = 0
        for a, b in same_pairs:
            try:
                r1 = compute_embedding_with_pipeline(pipeline, a, use_tta)
                r2 = compute_embedding_with_pipeline(pipeline, b, use_tta)
                e1, e2 = r1['embedding'], r2['embedding']
                if use_center:
                    e1 = l2_normalize(e1 - mean_emb)
                    e2 = l2_normalize(e2 - mean_emb)
                same_sims.append(cosine_similarity(e1, e2))
            except Exception:
                fail_same += 1

        diff_sims = []
        fail_diff = 0
        for a, b in diff_pairs:
            try:
                r1 = compute_embedding_with_pipeline(pipeline, a, use_tta)
                r2 = compute_embedding_with_pipeline(pipeline, b, use_tta)
                e1, e2 = r1['embedding'], r2['embedding']
                if use_center:
                    e1 = l2_normalize(e1 - mean_emb)
                    e2 = l2_normalize(e2 - mean_emb)
                diff_sims.append(cosine_similarity(e1, e2))
            except Exception:
                fail_diff += 1

        if same_sims and diff_sims:
            same_mean = np.mean(same_sims)
            diff_mean = np.mean(diff_sims)
            separation = same_mean - diff_mean

            all_sims_arr = np.array(same_sims + diff_sims)
            labels = np.array([1]*len(same_sims) + [0]*len(diff_sims))
            thresholds = np.linspace(-1, 1, 200)
            best_acc = 0
            for t in thresholds:
                acc = (np.sum(np.array(same_sims) >= t) + np.sum(np.array(diff_sims) < t)) / len(labels)
                if acc > best_acc:
                    best_acc = acc

            key = f"CFP-FP|{model_name}|{trick}"
            all_results[key] = {
                'same_mean': float(same_mean), 'diff_mean': float(diff_mean),
                'separation': float(separation), 'best_acc': float(best_acc),
                'same_N': len(same_sims), 'diff_N': len(diff_sims),
                'same_min': float(np.min(same_sims)), 'diff_max': float(np.max(diff_sims)),
            }
            print(f"    {trick:15s}: sep={separation:.4f}, acc={best_acc*100:.1f}%, "
                  f"same={same_mean:.4f}, diff={diff_mean:.4f} "
                  f"[{len(same_sims)}s/{len(diff_sims)}d, fail={fail_same}/{fail_diff}]")
        else:
            print(f"    {trick:15s}: FAILED (no valid results)")

    t_model_elapsed = time.time() - t_model_start
    print(f"    Model time: {t_model_elapsed:.0f}s")

# ============================================================
# CFP-FP Report
# ============================================================
print(f"\n{'='*80}")
print("CFP-FP RESULTS TABLE")
print(f"{'='*80}")
print(f"\n{'Model':<15} {'Trick':<15} {'Separation':>10} {'Accuracy':>10} {'Same':>8} {'Diff':>8} {'S_min':>8} {'D_max':>8}")
print("-" * 90)

for key in sorted(all_results.keys(), key=lambda k: -all_results[k]['separation']):
    _, model, trick = key.split('|')
    r = all_results[key]
    print(f"{model:<15} {trick:<15} {r['separation']:>10.4f} {r['best_acc']:>9.1%} {r['same_mean']:>8.4f} {r['diff_mean']:>8.4f} {r['same_min']:>8.4f} {r['diff_max']:>8.4f}")

print(f"\n{'='*80}")
print("BEST PER MODEL")
print(f"{'='*80}")
for model_name in models:
    best_trick = None
    best_sep = -1
    for trick in tricks:
        key = f"CFP-FP|{model_name}|{trick}"
        if key in all_results and all_results[key]['separation'] > best_sep:
            best_sep = all_results[key]['separation']
            best_trick = trick
    if best_trick:
        r = all_results[f"CFP-FP|{model_name}|{best_trick}"]
        print(f"  {model_name:<15} -> {best_trick:<15} sep={r['separation']:.4f} acc={r['best_acc']*100:.1f}%")

# ============================================================
# COMBINED REPORT (LFW + CFP-FP)
# ============================================================
# LFW results from previous run (hardcoded)
lfw_results = {
    'LFW|Official-F32|BASELINE':    {'separation': 0.4722, 'best_acc': 0.964, 'same_mean': 0.6681, 'diff_mean': 0.1959},
    'LFW|Official-F32|TTA':         {'separation': 0.4762, 'best_acc': 0.964, 'same_mean': 0.6768, 'diff_mean': 0.2006},
    'LFW|Official-F32|CENTER':      {'separation': 0.5606, 'best_acc': 0.964, 'same_mean': 0.6029, 'diff_mean': 0.0423},
    'LFW|Official-F32|TTA+CENTER':  {'separation': 0.5668, 'best_acc': 0.964, 'same_mean': 0.6121, 'diff_mean': 0.0453},
    'LFW|Official-I8|BASELINE':     {'separation': 0.4134, 'best_acc': 0.964, 'same_mean': 0.6387, 'diff_mean': 0.2252},
    'LFW|Official-I8|TTA':          {'separation': 0.4332, 'best_acc': 0.964, 'same_mean': 0.6586, 'diff_mean': 0.2254},
    'LFW|Official-I8|CENTER':       {'separation': 0.5149, 'best_acc': 0.964, 'same_mean': 0.5486, 'diff_mean': 0.0337},
    'LFW|Official-I8|TTA+CENTER':   {'separation': 0.5415, 'best_acc': 0.964, 'same_mean': 0.5688, 'diff_mean': 0.0273},
    'LFW|Foamliu-I8|BASELINE':      {'separation': 0.5494, 'best_acc': 0.952, 'same_mean': 0.5863, 'diff_mean': 0.0369},
    'LFW|Foamliu-I8|TTA':           {'separation': 0.5676, 'best_acc': 0.964, 'same_mean': 0.6121, 'diff_mean': 0.0445},
    'LFW|Foamliu-I8|CENTER':        {'separation': 0.5572, 'best_acc': 0.952, 'same_mean': 0.5789, 'diff_mean': 0.0218},
    'LFW|Foamliu-I8|TTA+CENTER':    {'separation': 0.5766, 'best_acc': 0.952, 'same_mean': 0.6041, 'diff_mean': 0.0275},
}

print(f"\n\n{'='*80}")
print("FULL COMPARISON: LFW + CFP-FP")
print(f"{'='*80}")
print(f"\n{'Dataset':<10} {'Model':<15} {'Trick':<15} {'Separation':>10} {'Accuracy':>10} {'Same':>8} {'Diff':>8}")
print("-" * 78)

all_combined = {**lfw_results, **all_results}
for key in sorted(all_combined.keys(), key=lambda k: (k.split('|')[0], -all_combined[k]['separation'])):
    ds, model, trick = key.split('|')
    r = all_combined[key]
    print(f"{ds:<10} {model:<15} {trick:<15} {r['separation']:>10.4f} {r['best_acc']:>9.1%} {r['same_mean']:>8.4f} {r['diff_mean']:>8.4f}")

print(f"\n{'='*80}")
print("BEST CONFIGURATION PER MODEL (LFW + CFP-FP)")
print(f"{'='*80}")
for ds_name in ['LFW', 'CFP-FP']:
    print(f"\n{ds_name}:")
    for model_name in ['Official-F32', 'Official-I8', 'Foamliu-I8']:
        best_trick = None
        best_sep = -1
        for trick in tricks:
            key = f"{ds_name}|{model_name}|{trick}"
            if key in all_combined and all_combined[key]['separation'] > best_sep:
                best_sep = all_combined[key]['separation']
                best_trick = trick
        if best_trick:
            r = all_combined[f"{ds_name}|{model_name}|{best_trick}"]
            print(f"  {model_name:<15} -> {best_trick:<15} sep={r['separation']:.4f} acc={r['best_acc']*100:.1f}%")

t_total = time.time() - t_start
print(f"\nTotal CFP-FP time: {t_total:.0f}s ({t_total/60:.1f} min)")
print("Done!")
