/*
 * common_config.h
 *
 * Configuration for Face Embedding using SCRFD + MobileFaceNet.
 *
 * Models:
 *   - SCRFD_500M_KPS: Face detection with 5-point landmarks
 *   - MobileFaceNet: QAT distill_v2 ReLU6 (glint360k teacher, 128D output)
 *
 *  Created on: Dec 11, 2025
 *      Author: Face Embedding App
 */


#ifndef SCENARIO_TFLM_FACE_EMBEDDING_COMMON_CONFIG_H_
#define SCENARIO_TFLM_FACE_EMBEDDING_COMMON_CONFIG_H_

/* Debug and system configuration */
#define FRAME_CHECK_DEBUG               1
#define EN_ALGO                         1
#define SPI_SEN_PIC_CLK                 (12000000)
#define WATCHDOG_VERSION
#define DBG_APP_LOG                     0

/*
 * Model flash addresses (4KB aligned)
 *
 * Memory layout (sscma_micro scans 0x400000-0xE00000):
 *   0x00000000 - 0x00200000: Firmware (2 MB)
 *   0x00400000 - 0x004B4000: SCRFD model (717 KB)         -> ID=1
 *   0x00510000 - 0x0064C000: MobileFaceNet QAT distill_v2 ReLU6 128D -> ID=2
 *   0x00700000 - 0x0089B000: Swift YOLO / test input area
 */
#define SCRFD_MODEL_FLASH_ADDR          (BASE_ADDR_FLASH1_R_ALIAS + 0x400000)
#define MOBILEFACENET_MODEL_FLASH_ADDR  (BASE_ADDR_FLASH1_R_ALIAS + 0x510000)
#define FACE_EMB_TEST_INPUT_FLASH_OFFSET (0x700000)
#define FACE_EMB_TEST_INPUT_FLASH_ADDR  (BASE_ADDR_FLASH1_R_ALIAS + FACE_EMB_TEST_INPUT_FLASH_OFFSET)

/* Legacy defines for backward compatibility */
#define FACE_DETECT_FLASH_ADDR          SCRFD_MODEL_FLASH_ADDR
#define FACE_EMBEDDING_FLASH_ADDR       MOBILEFACENET_MODEL_FLASH_ADDR

/*
 * SCRFD Face Detection Model Configuration
 *
 * SCRFD_500M_KPS specifications:
 *   - Input: 160x160 RGB
 *   - Output: Multi-scale detection (strides 8, 16, 32)
 *   - Features: Bounding boxes + 5-point landmarks
 */
#define FD_INPUT_TENSOR_WIDTH           160
#define FD_INPUT_TENSOR_HEIGHT          160
#define FD_INPUT_TENSOR_CHANNEL         3

/* SCRFD-specific configuration */
#define SCRFD_NUM_STRIDES               3       /* Detection at 3 scales */
#define SCRFD_NUM_ANCHORS_PER_LOC       2       /* Anchors per grid location */
#define SCRFD_NUM_LANDMARKS             5       /* 5-point landmarks */

/*
 * MobileFaceNet Embedding Model Configuration
 *
 * QAT distill_v2 ReLU6 128D MobileFaceNet
 * (model_zoo/tflm_face_embedding/qat_distill_v2_relu6_128d/):
 *   - Teacher: InsightFace glint360k_r100; student QAT-fine-tuned, LeakyReLU
 *     swapped to ReLU6 (bounded activations keep discrimination through
 *     Ethos-U55 per-tensor int8), 128D projection trained end-to-end.
 *   - Input: 112x112 RGB (aligned face, normalized to [-1,1])
 *   - Output: 128-dimensional embedding (L2 normalized), int8 zp=18 scale~0.01266
 *   - Accuracy (int8): LFW 99.33% / CFP-FP 94.26%; LFW impostor mean 0.005.
 *     On-device stranger impostor ~0.045 (was ~0.25 with the old w600k model).
 *   - Vela: 596.84 KiB SRAM, ~1008 KiB flash, 100% NPU (0 CPU ops)
 *   - Host-side cosine match threshold ~0.30 (this model centers impostors at
 *     ~0; the old model needed ~0.4). Enroll on-device, not from photos.
 */
#define EMBEDDING_INPUT_WIDTH           112
#define EMBEDDING_INPUT_HEIGHT          112
#define EMBEDDING_INPUT_CHANNEL         3
#define EMBEDDING_OUTPUT_DIM            128     /* QAT distill_v2 ReLU6 128D embedding */

/*
 * Memory Configuration
 *
 * Tensor arena allocation for dual-model inference.
 *
 * Memory usage (from Vela 3.9.0 compilation):
 *   - SCRFD: 201 KB (Vela reports 200.81 KB)
 *   - MobileFaceNet: 620 KB reserved (Vela reports 599 KiB)
 *   - Total: ~840 KB
 *
 * Note: 620 KB = 599 KiB Vela peak + ~3.5% TFLM/alignment margin. The smaller
 * footprint widens the SenseCap Watcher stream-on headroom (see
 * FACE_SAFE_RUNTIME_ARENA_BUDGET below). If AllocateTensors fails on device,
 * bump only MOBILEFACENET_ARENA_SIZE first before changing the flashed model.
 */
#define SCRFD_ARENA_SIZE                (220 * 1024)    /* 220 KB for face detection (Vela: 201 KB) */
#define MOBILEFACENET_ARENA_SIZE        (620 * 1024)     /* QAT distill_v2 ReLU6 128D (Vela: 597 KiB) */

/* Legacy define for total reference */
#define TENSOR_ARENA_SIZE               (SCRFD_ARENA_SIZE + MOBILEFACENET_ARENA_SIZE)

/* Aligned face buffer size (112x112 RGB) */
#define ALIGNED_FACE_BUFFER_SIZE        (112 * 112 * 3)

/*
 * 640x480 camera YUV422 DMA occupies the tail of SRAM1 from 0x3416A000.
 * Face inference allocations start at 0x34054000, leaving 0x116000 bytes
 * before they overlap live camera DMA memory.
 *
 * If the runtime arena footprint is larger than this safe budget, stop the
 * camera stream before running SCRFD/MobileFaceNet. Keep this as a configurable
 * policy so smaller models can run with the stream left active.
 */
#define FACE_SRAM_BEFORE_YUV422_BYTES       (0x116000)
#define FACE_FIXED_IMAGE_BUFFER_SIZE        (FD_INPUT_TENSOR_WIDTH * FD_INPUT_TENSOR_HEIGHT * FD_INPUT_TENSOR_CHANNEL + ALIGNED_FACE_BUFFER_SIZE)
#define FACE_SAFE_RUNTIME_ARENA_BUDGET      (FACE_SRAM_BEFORE_YUV422_BYTES - FACE_FIXED_IMAGE_BUFFER_SIZE)
#define FACE_RUNTIME_ARENA_SIZE             (SCRFD_ARENA_SIZE + MOBILEFACENET_ARENA_SIZE)

#ifndef FACE_STOP_STREAM_BEFORE_INFERENCE
#define FACE_STOP_STREAM_BEFORE_INFERENCE   (FACE_RUNTIME_ARENA_SIZE > FACE_SAFE_RUNTIME_ARENA_BUDGET)
#endif

/*
 * Optional diagnostic path for running a pre-Vela MobileFaceNet model through
 * TFLite Micro CPU kernels on the device. Keep this disabled in production:
 * the extra kernels exceed the Watcher/Grove Vision firmware memory budget.
 */
#ifndef FACE_ENABLE_EMB_CPU_OPS
#define FACE_ENABLE_EMB_CPU_OPS             0
#endif

/*
 * Ethos-U coherency guard for embedding inference. The production default
 * cleans the CPU-written input and invalidates the NPU-written output only.
 * Do not invalidate the full tensor arena before Invoke(): TFLM may have
 * prepared dirty CPU-side arena state that the NPU still needs to read.
 * Full-arena invalidation is kept only as a diagnostic switch.
 */
#ifndef FACE_CLEAN_INVALIDATE_EMB_ARENA_BEFORE_INVOKE
#define FACE_CLEAN_INVALIDATE_EMB_ARENA_BEFORE_INVOKE 0
#endif

#ifndef FACE_INVALIDATE_EMB_ARENA_BEFORE_INVOKE
#define FACE_INVALIDATE_EMB_ARENA_BEFORE_INVOKE 0
#endif

/* Diagnostic only: print face pipeline buffer/tensor addresses at init. */
#ifndef FACE_DEBUG_MEMORY_LAYOUT
#define FACE_DEBUG_MEMORY_LAYOUT        0
#endif

/* Diagnostic only: isolate cache coherency by running MobileFaceNet with
 * D-Cache disabled around Invoke(). Keep disabled for production. */
#ifndef FACE_DISABLE_DCACHE_FOR_EMB_INVOKE
#define FACE_DISABLE_DCACHE_FOR_EMB_INVOKE 0
#endif

/* Face detection thresholds */
#define FACE_CONF_THRESHOLD             0.50f   /* SCRFD detection score gate: only high-score frontal faces pass, so the embedding handed to the matcher comes from a clean crop. (An earlier comment tied specific cosine values to score here; those were the old w600k model's cross-domain figures and no longer apply to the QAT distill_v2 model.) */
#define FACE_NMS_THRESHOLD              0.4f    /* NMS IoU threshold */
#define MIN_FACE_SIZE                   40      /* Minimum face size in pixels */

/* Face pose limits for quality filtering (more lenient with alignment) */
#define MAX_YAW_ANGLE                   45.0f   /* Left-right rotation limit */
#define MAX_PITCH_ANGLE                 45.0f   /* Up-down rotation limit */
#define MAX_ROLL_ANGLE                  45.0f   /* In-plane rotation limit */

/* Face quality gate (yaw-based, see estimate_face_quality in face_alignment.c).
 * quality = 1.0 frontal .. 0.0 at ~45deg profile. 0.3 rejects roughly >40deg yaw,
 * which the eyes-only alignment cannot straighten. Conservative until swept on
 * device with AT+FACEFRAME known-yaw frames; raise it once calibrated. */
#define MIN_FACE_QUALITY                0.3f

/* UART communication */
#define DATA_TYPE_FACE_EMBEDDING        0xA0

/* Enable face alignment (recommended for best accuracy) */
#define ENABLE_FACE_ALIGNMENT           1

#endif /* SCENARIO_TFLM_FACE_EMBEDDING_COMMON_CONFIG_H_ */
