# SSCMA Face - Face Recognition Extension for SSCMA

This scenario app extends SSCMA with face recognition capabilities for the SenseCAP Watcher.

## Features

- Standard SSCMA AT command interface for object detection
- `AT+FACE=1` command to enable face recognition mode
- Face detection using SCRFD_500M_KPS (160x160)
- 128D face embedding using QAT distill_v2 ReLU6 MobileFaceNet (112x112)
- Face alignment using 5-point landmarks
- JPEG image output for remote preview
- Compatible with ESP32 face database for recognition

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      SSCMA Face App                          │
├──────────────────────────────────────────────────────────────┤
│  AT+FACE=0                     │  AT+FACE=1                  │
│  (Object Detection Mode)       │  (Face Recognition Mode)    │
│                                │                             │
│  ┌─────────────────┐           │  ┌─────────────────────┐    │
│  │ Standard SSCMA  │           │  │ SCRFD Face Detect   │    │
│  │ Model Pipeline  │           │  │ (160x160 → boxes)   │    │
│  └────────┬────────┘           │  └──────────┬──────────┘    │
│           │                    │             │               │
│           v                    │             v               │
│  ┌─────────────────┐           │  ┌─────────────────────┐    │
│  │ INVOKE Output   │           │  │ Face Alignment      │    │
│  │ (boxes/classes) │           │  │ (similarity xform)  │    │
│  └─────────────────┘           │  └──────────┬──────────┘    │
│                                │             │               │
│                                │             v               │
│                                │  ┌─────────────────────┐    │
│                                │  │ MobileFaceNet       │    │
│                                │  │ (112x112 → 128D)    │    │
│                                │  └──────────┬──────────┘    │
│                                │             │               │
│                                │             v               │
│                                │  ┌─────────────────────┐    │
│                                │  │ INVOKE Output       │    │
│                                │  │ (image + faces +    │    │
│                                │  │  embedding)         │    │
│                                │  └─────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

## AT Commands

### Standard SSCMA Commands
- `AT+ID?` - Get device ID
- `AT+NAME?` - Get device name
- `AT+MODEL` - Set model
- `AT+INVOKE` - Run inference
- ... (all standard SSCMA commands)

### Face Mode Commands
- `AT+FACE=1` - Enable face recognition mode
- `AT+FACE=0` - Disable face mode (return to object detection)
- `AT+FACE?` - Query current face mode status

## Output Format

### Object Detection Mode (AT+FACE=0)

Standard SSCMA INVOKE format with `boxes[]`:

```json
{
  "type": 1,
  "name": "INVOKE",
  "code": 0,
  "data": {
    "count": 1,
    "image": "<base64 JPEG>",
    "boxes": [[100, 50, 80, 100, 95, 0]],
    "resolution": [640, 480]
  }
}
```

### Face Recognition Mode (AT+FACE=1)

INVOKE format with `faces[]` (includes embedding, landmarks, quality):

```json
{
  "type": 1,
  "name": "INVOKE",
  "code": 0,
  "data": {
    "mode": "face",
    "count": 1,
    "image": "<base64 JPEG>",
    "resolution": [640, 480],
    "faces": [
      {
        "box": [100, 50, 80, 100],
        "score": 95,
        "quality": 0.85,
        "landmarks": [[120, 70], [160, 70], [140, 95], [125, 120], [155, 120]],
        "embedding": [0.0123, -0.0456, ..., 0.0789]
      }
    ]
  }
}
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `mode` | string | Always `"face"` in face mode |
| `image` | string | Base64 encoded JPEG from DP pipeline |
| `resolution` | [int, int] | Image resolution [width, height] |
| `faces` | array | Detected faces (empty `[]` if no face) |
| `faces[].box` | [int, int, int, int] | Bounding box [x, y, w, h] |
| `faces[].score` | int | Detection confidence (0-100) |
| `faces[].quality` | float | Face quality score |
| `faces[].landmarks` | [[int, int], ...] | 5-point landmarks (left eye, right eye, nose, left mouth, right mouth) |
| `faces[].embedding` | [float, ...] | 128D face embedding vector (L2 normalized) |

## Model Flash Layout

| Address | Model | Size |
|---------|-------|------|
| 0x400000 | SCRFD_500M_KPS | ~700 KB |
| 0x510000 | QAT distill_v2 ReLU6 MobileFaceNet 128D | ~1008 KB |

The 128D embedding model is a drop-in replacement (112×112×3 int8 input, 128D int8
output), so the firmware needs no code change to adopt it — just reflash the
`0x510000` slot. See
`model_zoo/tflm_face_embedding/qat_distill_v2_relu6_128d/README.md` for provenance,
accuracy, and the host-side match threshold.

## Memory Usage

- SCRFD arena: 220 KB
- Embedding-model arena: 620 KB reserved (Vela reports 596.84 KiB)
- Total: ~840 KB (allocated from EL_ALLOC region at runtime)

## Face Matching and Enrollment

The Himax firmware only produces embeddings; the cosine match/enrollment decision
happens on the host (ESP32 / SBC) face database.

- **Host-side cosine match threshold: ~0.30.** The deployed QAT distill_v2 model
  centers stranger (impostor) similarity near 0 and genuine similarity near 0.6, so
  the operating point moved down from the old model's ~0.4. Leaving the host
  threshold at 0.4 falsely rejects some genuine matches.
- **Enroll on-device.** Register faces from the device camera rather than uploaded
  phone photos: the device-camera imaging domain differs from clean photos. The new
  model narrows this cross-domain gap (genuine ~0.6), so photo enrollment is
  becoming viable, but device-camera enrollment remains the recommended path —
  validate with multiple people before relying on photo enrollment.

## Building

```bash
cd EPII_CM55M_APP_S
gmake clean && gmake -j8 TARGET=SENSECAP_WATCHER
```

## Integration with SBC

The SenseCAP Watcher uses a dual-serial architecture for face recognition:

```
┌──────────────┐     USB Port A (Console CRUD)    ┌──────────────┐
│  SBC App     │  <-----------------------------> │  ESP32-S3    │
│  (e.g. RPi)  │  face_list / face_add /          │  FaceDatabase│
│              │  face_delete / face_rename        │  (NVS)       │
│              │                                   └──────────────┘
│              │     USB Port B (SSCMA AT)         ┌──────────────┐
│              │  <-----------------------------> │  Himax WE2   │
│              │  AT+FACE=1 → INVOKE with          │  SCRFD +     │
│              │  image + faces + embedding         │  MobileFaceNet│
└──────────────┘                                   └──────────────┘
```

**Flow:**
1. SBC sends `AT+FACE=1` to Himax via Port B to enable face mode
2. SBC sends `AT+INVOKE=-1,1` to start continuous inference
3. Himax streams INVOKE messages with `image`, `faces[].embedding`, etc.
4. SBC compares embedding against ESP32's face database (via Port A CRUD)
5. SBC sends `face_add <name> <embedding>` to register new faces

## License

Apache 2.0
