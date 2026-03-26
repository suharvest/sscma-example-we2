/*
 * cvapp_face_embedding.cpp
 *
 * Face Embedding Implementation using SCRFD + MobileFaceNet
 *
 * Pipeline:
 *   1. Face Detection: SCRFD_500M_KPS (160x160) with 5-point landmarks
 *   2. Face Alignment: Similarity transform using eye positions
 *   3. Face Embedding: MobileFaceNet (112x112 -> 128D)
 *
 * Features:
 *   - CMSIS-NN hardware acceleration
 *   - INT8 quantization for both models
 *   - Face alignment for improved recognition accuracy
 *
 * Created for Grove Vision AI Module V2
 */

#include <cstdio>
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "WE2_device.h"
#include "board.h"
#include "cvapp_face_embedding.h"
#include "WE2_core.h"

#include "ethosu_driver.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/c/common.h"
#if TFLM2209_U55TAG2205
#include "tensorflow/lite/micro/micro_error_reporter.h"
#endif
#include "img_proc_helium.h"

/* sscma_micro's YUV422→RGB conversion (must be before send_result.h to avoid type redefinition) */
#include "el_types.h"
#include "el_cv.h"

#include "xprintf.h"
#include "spi_master_protocol.h"
#include "memory_manage.h"
#include "porting/el_misc.h"  /* el_aligned_malloc_once */
#include "common_config.h"
#include "face_embedding_protocol.h"
#include "scrfd_postprocessing.h"
#include "face_alignment.h"
#include "send_result.h"

/* DP pipeline helper (C function from cis_sensor) */
extern "C" void cisdp_get_jpginfo(uint32_t *jpeg_enc_filesize, uint32_t *jpeg_enc_addr);

/* Access the existing Ethos-U driver from sscma_micro */
namespace edgelab {
namespace porting {
extern struct ethosu_driver _ethosu_drv;
}
}

#ifdef TRUSTZONE_SEC
#define U55_BASE    BASE_ADDR_APB_U55_CTRL_ALIAS
#else
#ifndef TRUSTZONE
#define U55_BASE    BASE_ADDR_APB_U55_CTRL_ALIAS
#else
#define U55_BASE    BASE_ADDR_APB_U55_CTRL
#endif
#endif

#define TOTAL_STEP_TICK 0  /* Disable per-frame timing output */
#define CPU_CLK (0xffffff + 1)

/* DEBUG: Output all detected faces without running embedding
 * Set to 1 to debug face detection, 0 for normal operation */
#define DEBUG_DETECTION_ONLY 0

/* Debug verbosity levels:
 * 0 = Minimal (only timing summary)
 * 1 = Normal (step timing + key info)
 * 2 = Verbose (all debug info)
 */
#define DEBUG_VERBOSE 0

#if DEBUG_VERBOSE >= 2
#define DBG_VERBOSE(fmt, ...) xprintf(fmt, ##__VA_ARGS__)
#else
#define DBG_VERBOSE(fmt, ...) ((void)0)
#endif

#if DEBUG_VERBOSE >= 1
#define DBG_INFO(fmt, ...) xprintf(fmt, ##__VA_ARGS__)
#else
#define DBG_INFO(fmt, ...) ((void)0)
#endif

/* Helper macro for step timing */
#define TICK_TO_MS(ticks) ((ticks) / 24000)

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

using namespace std;

namespace {

/*
 * Memory allocation for dual-model inference
 *
 * Strategy: SEPARATE tensor arenas for each model, allocated as STATIC buffers
 * to avoid conflicts with sscma_micro's BSS memory region.
 *
 * Memory usage (Vela 3.9.0):
 *   SCRFD arena:        220 KB (Vela: 201 KB)
 *   MobileFaceNet arena: 700 KB (Vela: 600 KB)
 *   Total:              920 KB
 */
constexpr int scrfd_arena_size = SCRFD_ARENA_SIZE;
constexpr int mobilefacenet_arena_size = MOBILEFACENET_ARENA_SIZE;
constexpr int mobilefacenet_arena_head_offset = 128;  /* Safety offset to avoid arena head conflicts */
constexpr int fd_resize_image_size = FD_INPUT_TENSOR_WIDTH * FD_INPUT_TENSOR_HEIGHT * FD_INPUT_TENSOR_CHANNEL;
constexpr int aligned_face_buffer_size = ALIGNED_FACE_BUFFER_SIZE;
constexpr int rgb_frame_buffer_size = 240 * 240 * 3;  /* Full frame interleaved RGB: 240x240 from sscma_micro camera */

/*
 * Buffer pointers - allocated dynamically from sscma_micro's elHeap via
 * el_aligned_malloc_once(). This shares the 1112 KB EL_ALLOC region with
 * sscma_micro's tensor arenas and other allocations.
 *
 * Total face embedding memory: ~1.26 MB
 *   - SCRFD arena: 220 KB
 *   - MobileFaceNet arena: 700 KB
 *   - Image buffers: ~337 KB
 *
 * Note: If sscma_micro already uses significant elHeap memory (e.g., for YOLO),
 * face embedding init may fail due to insufficient memory.
 */
static uint32_t scrfd_tensor_arena = 0;
static uint32_t mobilefacenet_tensor_arena = 0;
static uint32_t fd_resized_img = 0;
static uint32_t aligned_face_img = 0;
static uint32_t rgb_frame_buffer = 0;  /* Buffer for interleaved RGB (from planar BGR) */

/* Scale factors for coordinate mapping (model space -> original image space) */
static float scale_w = 1.0f;
static float scale_h = 1.0f;

/* Letterbox parameters for SCRFD preprocessing */
static float letterbox_scale = 1.0f;  /* Uniform scale factor */
static int letterbox_pad_x = 0;       /* X padding (left side) */
static int letterbox_pad_y = 0;       /* Y padding (top side) */

/* Ethos-U NPU driver - use the existing driver from sscma_micro
 * DO NOT declare a new ethosu_drv here - we must use the one from
 * edgelab::porting::_ethosu_drv which was initialized at device startup */

/* Face Detection (SCRFD) interpreter */
tflite::MicroInterpreter *fd_int_ptr = nullptr;
TfLiteTensor *fd_input = nullptr;

/* SCRFD outputs: 9 tensors (3 strides x (score + bbox + kps)) */
TfLiteTensor *fd_score_tensors[SCRFD_NUM_STRIDES];
TfLiteTensor *fd_bbox_tensors[SCRFD_NUM_STRIDES];
TfLiteTensor *fd_kps_tensors[SCRFD_NUM_STRIDES];

/* Face Embedding (MobileFaceNet) interpreter */
tflite::MicroInterpreter *emb_int_ptr = nullptr;
TfLiteTensor *emb_input = nullptr;
TfLiteTensor *emb_output = nullptr;

/* SCRFD network configuration */
scrfd_network scrfd_net;

static uint32_t g_face_emb_init = 0;

/*
 * Operator resolver
 *
 * Vela compiles most ops to NPU:
 *   - SCRFD: 100% NPU
 *   - MobileFaceNet: NPU + L2_NORMALIZATION (CPU fallback)
 *
 * L2_NORMALIZATION is not supported by Ethos-U, requires CPU fallback.
 */
static tflite::MicroMutableOpResolver<2> op_resolver;  /* EthosU + L2Norm */

}  // namespace

/*
 * NOTE: NPU initialization functions are REMOVED from this file.
 *
 * The Ethos-U55 NPU is already initialized by sscma_micro at device startup
 * (see el_device_we2.cpp). The driver instance is edgelab::porting::_ethosu_drv.
 *
 * We must NOT:
 * - Call ethosu_init() again (causes HALTED error)
 * - Set up our own IRQ handler (would override sscma_micro's handler)
 * - Declare our own ethosu_drv variable (must use existing one)
 *
 * The TFLite Micro EthosU delegate will automatically use the NPU through
 * the driver that was initialized by sscma_micro.
 */

/* ========== PUBLIC API ========== */

int cv_face_embedding_init(bool security_enable, bool privilege_enable,
                           uint32_t fd_model_addr, uint32_t embedding_model_addr)
{
    xprintf("Face Embedding Init...\n");

    /*
     * Memory allocation - reset elHeap bump allocator to reclaim memory used by
     * sscma_micro's YOLO tensor arena (1110 KB), then allocate face buffers.
     * This is safe because face mode doesn't use sscma_micro's standard model.
     */
    el_aligned_malloc_reset();
    void* arena1 = el_aligned_malloc_once(32, scrfd_arena_size);
    void* arena2 = el_aligned_malloc_once(32, mobilefacenet_arena_size + mobilefacenet_arena_head_offset);
    void* buf1 = el_aligned_malloc_once(32, fd_resize_image_size);
    void* buf2 = el_aligned_malloc_once(32, aligned_face_buffer_size);
    void* buf3 = el_aligned_malloc_once(32, rgb_frame_buffer_size);

    if (!arena1 || !arena2 || !buf1 || !buf2 || !buf3) {
        xprintf("ERROR: Face buffer allocation failed!\n");
        xprintf("  Need: SCRFD=%d, MFN=%d, resize=%d, align=%d, rgb=%d\n",
                scrfd_arena_size, mobilefacenet_arena_size, fd_resize_image_size,
                aligned_face_buffer_size, rgb_frame_buffer_size);
        return -30;
    }

    scrfd_tensor_arena = (uint32_t)arena1;
    mobilefacenet_tensor_arena = (uint32_t)((uint8_t*)arena2 + mobilefacenet_arena_head_offset);
    fd_resized_img = (uint32_t)buf1;
    aligned_face_img = (uint32_t)buf2;
    rgb_frame_buffer = (uint32_t)buf3;

    xprintf("Face buffers allocated from elHeap\n");

    /* NOTE: NPU initialization is completely SKIPPED here because sscma_micro
     * (el_device_we2.cpp) already initializes the Ethos-U55 NPU at device startup.
     */

    /* Load models from flash */
    static const tflite::Model *fd_model = tflite::GetModel((const void *)fd_model_addr);
    static const tflite::Model *emb_model = tflite::GetModel((const void *)embedding_model_addr);

    /* Model Integrity Check - silent on success */
    uint8_t* scrfd_bytes = (uint8_t*)fd_model_addr;
    if (scrfd_bytes[4] != 'T' || scrfd_bytes[5] != 'F' ||
        scrfd_bytes[6] != 'L' || scrfd_bytes[7] != '3') {
        xprintf("ERROR: SCRFD model invalid @ 0x%08X\n", fd_model_addr);
        return -10;
    }

    uint8_t* emb_bytes = (uint8_t*)embedding_model_addr;
    if (emb_bytes[4] != 'T' || emb_bytes[5] != 'F' ||
        emb_bytes[6] != 'L' || emb_bytes[7] != '3') {
        xprintf("ERROR: MobileFaceNet model invalid @ 0x%08X\n", embedding_model_addr);
        return -11;
    }

    /* Verify model schema versions */
    if (fd_model->version() != TFLITE_SCHEMA_VERSION) {
        xprintf("ERROR: SCRFD schema mismatch\n");
        return -2;
    }
    if (emb_model->version() != TFLITE_SCHEMA_VERSION) {
        xprintf("ERROR: MobileFaceNet schema mismatch\n");
        return -3;
    }
    /* Register operators */
    if (g_face_emb_init == 0) {
        if (kTfLiteOk != op_resolver.AddEthosU()) {
            xprintf("ERROR: Failed to add Ethos-U\n");
            return -4;
        }
        if (kTfLiteOk != op_resolver.AddL2Normalization()) {
            xprintf("ERROR: Failed to add L2Normalization\n");
            return -5;
        }
    }

    /* Create interpreters */

    if (scrfd_tensor_arena == 0) {
        xprintf("ERROR: SCRFD arena allocation failed!\n");
        return -20;
    }
    if (mobilefacenet_tensor_arena == 0) {
        xprintf("ERROR: MobileFaceNet arena allocation failed!\n");
        return -21;
    }

#if TFLM2209_U55TAG2205
    static tflite::MicroErrorReporter micro_error_reporter;
    static tflite::MicroInterpreter fd_static_interpreter(
        fd_model, op_resolver,
        (uint8_t *)scrfd_tensor_arena, scrfd_arena_size,
        &micro_error_reporter);
    static tflite::MicroInterpreter emb_static_interpreter(
        emb_model, op_resolver,
        (uint8_t *)mobilefacenet_tensor_arena, mobilefacenet_arena_size,
        &micro_error_reporter);
#else
    static tflite::MicroInterpreter fd_static_interpreter(
        fd_model, op_resolver,
        (uint8_t *)scrfd_tensor_arena, scrfd_arena_size);
    static tflite::MicroInterpreter emb_static_interpreter(
        emb_model, op_resolver,
        (uint8_t *)mobilefacenet_tensor_arena, mobilefacenet_arena_size);
#endif

    /* Allocate tensors */
    if (fd_static_interpreter.AllocateTensors() != kTfLiteOk) {
        xprintf("ERROR: SCRFD tensor allocation failed\n");
        return -26;
    }
    if (emb_static_interpreter.AllocateTensors() != kTfLiteOk) {
        xprintf("ERROR: MobileFaceNet tensor allocation failed\n");
        return -27;
    }

    /* Setup SCRFD interpreter */
    fd_int_ptr = &fd_static_interpreter;
    fd_input = fd_static_interpreter.input(0);

    /* Get SCRFD output tensors */
    int num_outputs = fd_static_interpreter.outputs_size();

    /* Initialize all to nullptr first */
    for (int i = 0; i < SCRFD_NUM_STRIDES; i++) {
        fd_score_tensors[i] = nullptr;
        fd_bbox_tensors[i] = nullptr;
        fd_kps_tensors[i] = nullptr;
    }

    /* Map outputs by shape analysis - handle both 2D (Vela) and 4D (original) formats */
    for (int i = 0; i < num_outputs; i++) {
        TfLiteTensor* t = fd_static_interpreter.output(i);

        int stride_idx = -1;
        int tensor_type = -1;  /* 0=score, 1=bbox, 2=kps */

        if (t->dims->size == 2) {
            /* Vela 2D format: [num_elements, channels] */
            int num_elements = t->dims->data[0];
            int channels = t->dims->data[1];

            /* Determine stride from number of elements */
            if (num_elements == 800) stride_idx = 0;       /* 20*20*2 = stride 8 */
            else if (num_elements == 200) stride_idx = 1;  /* 10*10*2 = stride 16 */
            else if (num_elements == 50) stride_idx = 2;   /* 5*5*2 = stride 32 */

            /* Determine tensor type from channels */
            if (channels == 1) tensor_type = 0;       /* score */
            else if (channels == 4) tensor_type = 1;  /* bbox */
            else if (channels == 10) tensor_type = 2; /* kps */

        } else if (t->dims->size == 4) {
            /* Original 4D format: [1, H, W, C] */
            int h = t->dims->data[1];
            int w = t->dims->data[2];
            int c = t->dims->data[3];

            if (h == 20 && w == 20) stride_idx = 0;
            else if (h == 10 && w == 10) stride_idx = 1;
            else if (h == 5 && w == 5) stride_idx = 2;

            if (c == 2) tensor_type = 0;
            else if (c == 8) tensor_type = 1;
            else if (c == 20) tensor_type = 2;
        }

        if (stride_idx < 0 || tensor_type < 0) continue;

        /* Assign tensor to appropriate slot */
        if (tensor_type == 0) {
            fd_score_tensors[stride_idx] = t;
        } else if (tensor_type == 1) {
            fd_bbox_tensors[stride_idx] = t;
        } else if (tensor_type == 2) {
            fd_kps_tensors[stride_idx] = t;
        }
    }

    /* Debug: print ALL output tensors for diagnosis */
    xprintf("SCRFD: %d outputs, input=%dx%d %s (zp=%d, sc=%d/1e6)\n",
            num_outputs,
            fd_input->dims->data[1], fd_input->dims->data[2],
            fd_input->type == kTfLiteInt8 ? "INT8" : "UINT8",
            fd_input->params.zero_point,
            (int)(fd_input->params.scale * 1000000));
    for (int i = 0; i < num_outputs; i++) {
        TfLiteTensor* t = fd_static_interpreter.output(i);
        xprintf("  out[%d]: dims=%d [", i, t->dims->size);
        for (int d = 0; d < t->dims->size; d++) {
            xprintf("%d%s", t->dims->data[d], d < t->dims->size - 1 ? "," : "");
        }
        xprintf("] %s zp=%d sc=%d/1e6 bytes=%d\n",
                t->type == kTfLiteInt8 ? "int8" : (t->type == kTfLiteUInt8 ? "uint8" : "f32"),
                t->params.zero_point,
                (int)(t->params.scale * 1000000),
                (int)t->bytes);
    }

    /* Verify we have at least one complete branch */
    bool has_valid_branch = false;
    for (int i = 0; i < SCRFD_NUM_STRIDES; i++) {
        if (fd_score_tensors[i] && fd_bbox_tensors[i] && fd_kps_tensors[i]) {
            has_valid_branch = true;
        }
    }

    if (!has_valid_branch) {
        xprintf("ERROR: No complete SCRFD branch found\n");
        return -28;
    }

    /* Initialize SCRFD post-processing */
    scrfd_net = scrfd_init(
        fd_score_tensors, fd_bbox_tensors, fd_kps_tensors,
        FD_INPUT_TENSOR_WIDTH, FD_INPUT_TENSOR_HEIGHT,
        FACE_CONF_THRESHOLD, FACE_NMS_THRESHOLD);

    /* Setup MobileFaceNet interpreter */
    emb_int_ptr = &emb_static_interpreter;
    emb_input = emb_static_interpreter.input(0);
    emb_output = emb_static_interpreter.output(0);

    /* No camera reconfiguration needed.
     * We use sscma_micro's existing YUV422 camera pipeline directly.
     * face_invoke.hpp captures frames via camera->start_stream/get_frame/stop_stream,
     * then passes YUV422P data here. We convert to RGB888 using el_img_convert(). */

    g_face_emb_init = 1;
    xprintf("Face init OK\n");

    return 0;
}

int cv_face_embedding_run(uint8_t *frame_data, uint32_t frame_width, uint32_t frame_height,
                           struct_algoResult *alg_result, face_embedding_msg_t *embedding_msg)
{
    static int frame_count = 0;
    frame_count++;
    DBG_VERBOSE("\n[Frame %d] cv_face_embedding_run started\n", frame_count);

#if TOTAL_STEP_TICK
    uint32_t systick_1, systick_2;
    uint32_t loop_cnt_1, loop_cnt_2;
    SystemGetTick(&systick_1, &loop_cnt_1);

    /* Step timing variables */
    uint32_t tick_preprocess, tick_scrfd, tick_postproc, tick_align, tick_mobilefacenet;
    uint32_t systick_step, loop_cnt_step;
#endif

    TfLiteStatus invoke_status = kTfLiteOk;

    /* ===== STEP 0: Convert YUV422P → RGB888 using sscma_micro's el_img_convert ===== */
    /* frame_data is YUV422P from sscma_micro's camera (same as YOLO uses).
     * We use el_img_convert() to resize + convert to interleaved RGB888.
     * This is the same conversion path that YOLO and all other sscma algorithms use. */
    uint32_t img_w = frame_width;
    uint32_t img_h = frame_height;

    /* CRITICAL: Invalidate D-Cache for DMA-written camera buffer (defensive).
     * face_invoke.hpp also invalidates, but double-invalidation is harmless. */
    uint32_t yuv_frame_size = img_w * img_h * 2;  /* YUV422P: 2 bytes/pixel */
    SCB_InvalidateDCache_by_Addr((uint32_t*)frame_data, yuv_frame_size);

    /* Convert YUV422P → RGB888 at SCRFD input size (160x160) using el_img_convert.
     * el_img_convert does resize + YUV→RGB in one pass (nearest-neighbor). */
    {
        el_img_t src_img = {};
        src_img.data = frame_data;
        src_img.size = img_w * img_h * 2;  /* YUV422P: 2 bytes/pixel */
        src_img.width = (uint16_t)img_w;
        src_img.height = (uint16_t)img_h;
        src_img.format = EL_PIXEL_FORMAT_YUV422;
        src_img.rotate = EL_PIXEL_ROTATE_0;

        el_img_t dst_img = {};
        dst_img.data = (uint8_t *)fd_resized_img;
        dst_img.size = FD_INPUT_TENSOR_WIDTH * FD_INPUT_TENSOR_HEIGHT * 3;
        dst_img.width = FD_INPUT_TENSOR_WIDTH;   /* 160 */
        dst_img.height = FD_INPUT_TENSOR_HEIGHT;  /* 160 */
        dst_img.format = EL_PIXEL_FORMAT_RGB888;
        dst_img.rotate = EL_PIXEL_ROTATE_0;

        edgelab::el_img_convert(&src_img, &dst_img);
    }

    /* Also convert full-size for face alignment later */
    {
        el_img_t src_img = {};
        src_img.data = frame_data;
        src_img.size = img_w * img_h * 2;
        src_img.width = (uint16_t)img_w;
        src_img.height = (uint16_t)img_h;
        src_img.format = EL_PIXEL_FORMAT_YUV422;
        src_img.rotate = EL_PIXEL_ROTATE_0;

        el_img_t dst_img = {};
        dst_img.data = (uint8_t *)rgb_frame_buffer;
        dst_img.size = img_w * img_h * 3;
        dst_img.width = (uint16_t)img_w;
        dst_img.height = (uint16_t)img_h;
        dst_img.format = EL_PIXEL_FORMAT_RGB888;
        dst_img.rotate = EL_PIXEL_ROTATE_0;

        edgelab::el_img_convert(&src_img, &dst_img);
    }

    DBG_VERBOSE("  YUV422P %dx%d → RGB888 converted\n", img_w, img_h);

    /* DEBUG: Check converted data for first 3 frames */
    if (frame_count <= 3) {
        uint8_t *rgb = (uint8_t *)fd_resized_img;
        int total = FD_INPUT_TENSOR_WIDTH * FD_INPUT_TENSOR_HEIGHT * 3;
        uint32_t sum = 0;
        uint8_t vmin = 255, vmax = 0;
        for (int i = 0; i < total; i++) {
            sum += rgb[i];
            if (rgb[i] < vmin) vmin = rgb[i];
            if (rgb[i] > vmax) vmax = rgb[i];
        }
        DBG_INFO("[DBG] Frame %d: %dx%d YUV->RGB: min=%d max=%d avg=%d first8=[%d,%d,%d,%d,%d,%d,%d,%d]\n",
                frame_count, img_w, img_h, vmin, vmax, (int)(sum / total),
                rgb[0], rgb[1], rgb[2], rgb[3], rgb[4], rgb[5], rgb[6], rgb[7]);
    }

    /* Initialize results */
    alg_result->num_tracked_human_targets = 0;
    memset(embedding_msg, 0, sizeof(face_embedding_msg_t));

    /* ===== STEP 1: Preprocess for SCRFD ===== */
    /* el_img_convert already resized YUV422P → RGB888 at 160x160 into fd_resized_img.
     * Just need to set up coordinate mapping and copy to input tensor. */
    DBG_VERBOSE("  Step 1: Preprocessing for SCRFD...\n");

    int new_w = FD_INPUT_TENSOR_WIDTH;   // 160
    int new_h = FD_INPUT_TENSOR_HEIGHT;  // 160

    letterbox_pad_x = 0;
    letterbox_pad_y = 0;
    letterbox_scale = 1.0f;

    /* Direct resize scales: model space → original image space */
    float w_scale = (float)img_w / new_w;
    float h_scale = (float)img_h / new_h;
    scale_w = w_scale;
    scale_h = h_scale;

    /* Fill input tensor with resized image (direct copy, no letterbox padding) */
    if (fd_input->type == kTfLiteInt8) {
        int8_t *dst = fd_input->data.int8;
        uint8_t *src = (uint8_t *)fd_resized_img;
        int32_t zp = fd_input->params.zero_point;
        float in_scale = fd_input->params.scale;
        int total = FD_INPUT_TENSOR_WIDTH * FD_INPUT_TENSOR_HEIGHT * FD_INPUT_TENSOR_CHANNEL;

        /* Direct copy with quantization (uint8 -> int8 with zero point) */
        for (int i = 0; i < total; i++) {
            int32_t val = (int32_t)src[i] + zp;
            if (val < -128) val = -128;
            if (val > 127) val = 127;
            dst[i] = (int8_t)val;
        }

        /* DEBUG: Print input tensor params and data for first 3 frames */
        if (frame_count <= 3) {
            /* Resized image stats */
            uint32_t rgb_sum = 0;
            uint8_t rgb_min = 255, rgb_max = 0;
            for (int i = 0; i < total; i++) {
                rgb_sum += src[i];
                if (src[i] < rgb_min) rgb_min = src[i];
                if (src[i] > rgb_max) rgb_max = src[i];
            }
            DBG_INFO("[DBG-INPUT] type=int8, zp=%d, scale=%d/1e6, total=%d\n",
                    zp, (int)(in_scale * 1000000), total);
            DBG_INFO("[DBG-INPUT] resized_img: min=%d max=%d avg=%d, first8=[%d,%d,%d,%d,%d,%d,%d,%d]\n",
                    rgb_min, rgb_max, (int)(rgb_sum / total),
                    src[0], src[1], src[2], src[3], src[4], src[5], src[6], src[7]);
            DBG_INFO("[DBG-INPUT] tensor_int8: first8=[%d,%d,%d,%d,%d,%d,%d,%d]\n",
                    dst[0], dst[1], dst[2], dst[3], dst[4], dst[5], dst[6], dst[7]);
        }
    } else {
        uint8_t *dst = fd_input->data.uint8;
        uint8_t *src = (uint8_t *)fd_resized_img;
        int total = FD_INPUT_TENSOR_WIDTH * FD_INPUT_TENSOR_HEIGHT * FD_INPUT_TENSOR_CHANNEL;

        /* Direct copy (uint8 -> uint8) */
        memcpy(dst, src, total);

        if (frame_count <= 3) {
            DBG_INFO("[DBG-INPUT] type=uint8, zp=%d, scale=%d/1e6\n",
                    fd_input->params.zero_point, (int)(fd_input->params.scale * 1000000));
            DBG_INFO("[DBG-INPUT] tensor_uint8: first8=[%d,%d,%d,%d,%d,%d,%d,%d]\n",
                    dst[0], dst[1], dst[2], dst[3], dst[4], dst[5], dst[6], dst[7]);
        }
    }


#if TOTAL_STEP_TICK
    SystemGetTick(&systick_step, &loop_cnt_step);
    tick_preprocess = (loop_cnt_step - loop_cnt_1) * CPU_CLK + (systick_1 - systick_step);
#endif

    /* ===== STEP 2: Run SCRFD Face Detection ===== */
    DBG_VERBOSE("  Step 2: Running SCRFD inference...\n");

    /* D-Cache Coherency Fix - CRITICAL for NPU */
    if (fd_input->data.data) {
        SCB_CleanDCache_by_Addr((uint32_t*)fd_input->data.data, fd_input->bytes);
    }
    
    invoke_status = fd_int_ptr->Invoke();
    if (invoke_status != kTfLiteOk) {
        xprintf("ERROR: SCRFD invoke failed\n");
        return -1;
    }

    /* CRITICAL: Invalidate D-Cache for output tensors after NPU inference
     * NPU writes directly to memory, bypassing CPU cache. Without invalidation,
     * CPU may read stale cached data causing false detections (ghost faces).
     */
    for (int i = 0; i < SCRFD_NUM_STRIDES; i++) {
        if (fd_score_tensors[i] && fd_score_tensors[i]->data.data) {
            SCB_InvalidateDCache_by_Addr((uint32_t*)fd_score_tensors[i]->data.data,
                                         fd_score_tensors[i]->bytes);
        }
        if (fd_bbox_tensors[i] && fd_bbox_tensors[i]->data.data) {
            SCB_InvalidateDCache_by_Addr((uint32_t*)fd_bbox_tensors[i]->data.data,
                                         fd_bbox_tensors[i]->bytes);
        }
        if (fd_kps_tensors[i] && fd_kps_tensors[i]->data.data) {
            SCB_InvalidateDCache_by_Addr((uint32_t*)fd_kps_tensors[i]->data.data,
                                         fd_kps_tensors[i]->bytes);
        }
    }

    /* DEBUG: Dump raw score tensor info for first 3 frames */
    if (frame_count <= 3) {
        for (int i = 0; i < SCRFD_NUM_STRIDES; i++) {
            if (fd_score_tensors[i]) {
                int8_t *data = fd_score_tensors[i]->data.int8;
                int sz = fd_score_tensors[i]->bytes;
                float sc = fd_score_tensors[i]->params.scale;
                int zp = fd_score_tensors[i]->params.zero_point;
                /* Find min/max int8 values */
                int8_t vmin = 127, vmax = -128;
                int cnt_above_zp = 0;
                int32_t sum = 0;
                for (int j = 0; j < sz; j++) {
                    if (data[j] < vmin) vmin = data[j];
                    if (data[j] > vmax) vmax = data[j];
                    if (data[j] > zp) cnt_above_zp++;
                    sum += data[j];
                }
                /* Compute max dequantized logit and sigmoid score */
                float max_logit = (float)(vmax - zp) * sc;
                float max_sigmoid = 1.0f / (1.0f + expf(-max_logit));
                DBG_INFO("[DBG] Score[%d]: zp=%d sc=%d/1e6 range=[%d,%d] avg=%d above_zp=%d/%d logit=%d sig=%d/1000\n",
                        i, zp, (int)(sc * 1000000), (int)vmin, (int)vmax,
                        (int)(sum / sz), cnt_above_zp, sz, (int)(max_logit * 1000), (int)(max_sigmoid * 1000));
            }
        }
        /* Also dump bbox tensor stats for stride 16 to check if model is doing anything meaningful */
        if (fd_bbox_tensors[1]) {
            int8_t *data = fd_bbox_tensors[1]->data.int8;
            int sz = fd_bbox_tensors[1]->bytes;
            int8_t vmin = 127, vmax = -128;
            for (int j = 0; j < sz; j++) {
                if (data[j] < vmin) vmin = data[j];
                if (data[j] > vmax) vmax = data[j];
            }
            DBG_INFO("[DBG] Bbox[1]: zp=%d range=[%d,%d] bytes=%d\n",
                    fd_bbox_tensors[1]->params.zero_point, (int)vmin, (int)vmax, sz);
        }
    }

#if TOTAL_STEP_TICK
    uint32_t systick_scrfd, loop_cnt_scrfd;
    SystemGetTick(&systick_scrfd, &loop_cnt_scrfd);
    tick_scrfd = (loop_cnt_scrfd - loop_cnt_step) * CPU_CLK + (systick_step - systick_scrfd);
#endif

    /* ===== STEP 3: SCRFD Post-processing ===== */
    /* Set coordinate mapping parameters
     * Formula: orig = (model - pad) * scale
     * We use the explicit geometric scales calculated during resize.
     * scale_x = img_w / new_w (stored in scale_w)
     * scale_y = img_h / new_h (stored in scale_h)
     */
    scrfd_net.scale_x = scale_w;
    scrfd_net.scale_y = scale_h;
    scrfd_net.pad_x = letterbox_pad_x;
    scrfd_net.pad_y = letterbox_pad_y;

    int num_faces = 0;
    std::forward_list<scrfd_face> faces = scrfd_detect(&scrfd_net, img_w, img_h, &num_faces);

    DBG_INFO("[FACE] Frame %d: %d faces detected (img=%dx%d)\n", frame_count, num_faces, img_w, img_h);

    /* Debug: show first face details */
    if (num_faces > 0) {
        for (auto& f : faces) {
            if (f.score > 0) {
                DBG_INFO("[FACE] best_face: bbox=(%d,%d,%d,%d) score=%d/1000 stride=%d\n",
                        (int)f.bbox.x, (int)f.bbox.y, (int)f.bbox.w, (int)f.bbox.h,
                        (int)(f.score * 1000), f.stride_idx);
                break;
            }
        }
    }

    if (num_faces == 0) {
        return 0;
    }

    DBG_INFO("[Frame] Detected %d face(s)\n", num_faces);


#if DEBUG_DETECTION_ONLY
    /* DEBUG MODE: Send all detected faces without running embedding */
    {
        uint32_t jpeg_size = 0;
        uint32_t jpeg_addr = 0;
        cisdp_get_jpginfo(&jpeg_size, &jpeg_addr);

        /* Build JSON with multiple faces */
        std::string json_str = "{\"type\":1,\"name\":\"FACE_RESULT\",\"code\":0,\"data\":{";

        /* Add image */
        if (jpeg_addr && jpeg_size > 0) {
            el_img_t jpeg_img;
            jpeg_img.data = (uint8_t*)jpeg_addr;
            jpeg_img.size = jpeg_size;
            jpeg_img.width = img_w;
            jpeg_img.height = img_h;
            jpeg_img.format = EL_PIXEL_FORMAT_JPEG;
            json_str += img_2_json_str(&jpeg_img);
            json_str += ", ";
            json_str += img_res_2_json_str(&jpeg_img);
        } else {
            json_str += "\"image\": \"\", \"resolution\": [0, 0]";
        }

        /* Add all faces */
        json_str += ", \"faces\": [";
        bool first_face = true;
        for (auto& f : faces) {
            if (f.score <= 0) continue;

            if (!first_face) json_str += ", ";
            first_face = false;

            char face_buf[256];
            snprintf(face_buf, sizeof(face_buf),
                     "{\"bbox\": [%d, %d, %d, %d], \"confidence\": %d.%02d, \"landmarks\": [",
                     (int)f.bbox.x, (int)f.bbox.y, (int)f.bbox.w, (int)f.bbox.h,
                     (int)(f.score * 100) / 100, (int)(f.score * 100) % 100);
            json_str += face_buf;

            /* Add 5 landmarks */
            for (int k = 0; k < 5; k++) {
                if (k > 0) json_str += ", ";
                char lm_buf[32];
                snprintf(lm_buf, sizeof(lm_buf), "[%d, %d]",
                         (int)f.landmarks[k].x, (int)f.landmarks[k].y);
                json_str += lm_buf;
            }
            json_str += "]}";
        }
        json_str += "]}}";

        /* Send JSON */
        json_str += "\r\n";
        send_bytes(json_str.c_str(), json_str.size());
        scrfd_free_dets(faces);
        return 0;
    }
#endif

    /* Get best (highest score) face with valid size */
    scrfd_face *best_face = scrfd_get_best_face(faces, MIN_FACE_SIZE);
    if (best_face == nullptr || best_face->score <= 0) {
        DBG_INFO("[FACE] best_face=NULL (min_size=%d), faces dropped by size filter\n", MIN_FACE_SIZE);
        scrfd_free_dets(faces);
        return 0;
    }
    DBG_INFO("[FACE] best_face: bbox=(%d,%d,%d,%d) score=%d\n",
            (int)best_face->bbox.x, (int)best_face->bbox.y,
            (int)best_face->bbox.w, (int)best_face->bbox.h,
            (int)(best_face->score * 1000));

    /* Store detection result */
    alg_result->num_tracked_human_targets = 1;
    alg_result->ht[0].upper_body_score = (uint32_t)(best_face->score * 100);
    alg_result->ht[0].upper_body_bbox.x = (uint32_t)best_face->bbox.x;
    alg_result->ht[0].upper_body_bbox.y = (uint32_t)best_face->bbox.y;
    alg_result->ht[0].upper_body_bbox.width = (uint32_t)best_face->bbox.w;
    alg_result->ht[0].upper_body_bbox.height = (uint32_t)best_face->bbox.h;

    /* Quality check: minimum face size */
    if (best_face->bbox.w < MIN_FACE_SIZE || best_face->bbox.h < MIN_FACE_SIZE) {
        DBG_INFO("  Face too small: %.0fx%.0f\n", best_face->bbox.w, best_face->bbox.h);
        alg_result->num_tracked_human_targets = 0;
        scrfd_free_dets(faces);
        return 0;
    }

    /* Validate landmarks */
    if (!scrfd_validate_face(best_face, MIN_FACE_SIZE, img_w, img_h)) {
        DBG_INFO("  Face validation failed\n");
        alg_result->num_tracked_human_targets = 0;
        scrfd_free_dets(faces);
        return 0;
    }

#if TOTAL_STEP_TICK
    uint32_t systick_post, loop_cnt_post;
    SystemGetTick(&systick_post, &loop_cnt_post);
    tick_postproc = (loop_cnt_post - loop_cnt_scrfd) * CPU_CLK + (systick_scrfd - systick_post);
#endif

    DBG_INFO("  Face: %dx%d score=%d%%\n",
            (int)best_face->bbox.w, (int)best_face->bbox.h, (int)(best_face->score * 100));

#if ENABLE_FACE_ALIGNMENT
    /* ===== STEP 4: Face Alignment ===== */
    /* rgb_frame_buffer already has full-size RGB888 from el_img_convert in STEP 0.
     * apply_face_alignment expects interleaved RGB input. */
    affine_transform_t align_transform;
    compute_face_alignment(best_face->landmarks, &align_transform);

    apply_face_alignment(
        (uint8_t *)rgb_frame_buffer, img_w, img_h,
        (uint8_t *)aligned_face_img,
        &align_transform);

    /* Copy to embedding input tensor - handle INT8 vs UINT8 */
    if (emb_input->type == kTfLiteInt8) {
        uint8_t *src = (uint8_t *)aligned_face_img;
        int8_t *dst = emb_input->data.int8;
        for (int i = 0; i < aligned_face_buffer_size; i++) {
            dst[i] = (int8_t)((int)src[i] - 128);
        }
        DBG_VERBOSE("  Embedding input ready (INT8, aligned)\n");
    } else {
        memcpy(emb_input->data.uint8, (uint8_t *)aligned_face_img, aligned_face_buffer_size);
        DBG_VERBOSE("  Embedding input ready (UINT8, aligned)\n");
    }
#else
    /* Fallback: simple resize of full frame to 112x112 */
    DBG_VERBOSE("  Step 4: Resizing for embedding (no alignment)...\n");

    /* Resize YUV422P → RGB888 at embedding input size using el_img_convert */
    {
        el_img_t src_img = {};
        src_img.data = frame_data;
        src_img.size = img_w * img_h * 2;
        src_img.width = (uint16_t)img_w;
        src_img.height = (uint16_t)img_h;
        src_img.format = EL_PIXEL_FORMAT_YUV422;
        src_img.rotate = EL_PIXEL_ROTATE_0;

        el_img_t dst_img = {};
        dst_img.data = (uint8_t *)aligned_face_img;
        dst_img.size = EMBEDDING_INPUT_WIDTH * EMBEDDING_INPUT_HEIGHT * 3;
        dst_img.width = EMBEDDING_INPUT_WIDTH;
        dst_img.height = EMBEDDING_INPUT_HEIGHT;
        dst_img.format = EL_PIXEL_FORMAT_RGB888;
        dst_img.rotate = EL_PIXEL_ROTATE_0;

        edgelab::el_img_convert(&src_img, &dst_img);
    }

    /* Copy to embedding input tensor - handle INT8 vs UINT8 */
    if (emb_input->type == kTfLiteInt8) {
        uint8_t *src = (uint8_t *)aligned_face_img;
        int8_t *dst = emb_input->data.int8;
        for (int i = 0; i < aligned_face_buffer_size; i++) {
            dst[i] = (int8_t)((int)src[i] - 128);
        }
        DBG_VERBOSE("  Embedding input ready (INT8)\n");
    } else {
        memcpy(emb_input->data.uint8, (uint8_t *)aligned_face_img, aligned_face_buffer_size);
        DBG_VERBOSE("  Embedding input ready (UINT8)\n");
    }
#endif

#if TOTAL_STEP_TICK
    uint32_t systick_align, loop_cnt_align;
    SystemGetTick(&systick_align, &loop_cnt_align);
    tick_align = (loop_cnt_align - loop_cnt_post) * CPU_CLK + (systick_post - systick_align);
#endif

    /* ===== STEP 5: Run MobileFaceNet Embedding ===== */
    DBG_VERBOSE("  Step 5: Running MobileFaceNet...\n");

    /* D-Cache Coherency Fix */
    SCB_CleanDCache_by_Addr((uint32_t*)emb_input->data.data, emb_input->bytes);
    SCB_CleanDCache_by_Addr((uint32_t*)mobilefacenet_tensor_arena, mobilefacenet_arena_size);
    __DSB();
    __ISB();

    /* Run embedding inference */
    invoke_status = emb_int_ptr->Invoke();
    if (invoke_status != kTfLiteOk) {
        xprintf("ERROR: MobileFaceNet invoke failed\n");
        scrfd_free_dets(faces);
        return -2;
    }

#if TOTAL_STEP_TICK
    uint32_t systick_mfn, loop_cnt_mfn;
    SystemGetTick(&systick_mfn, &loop_cnt_mfn);
    tick_mobilefacenet = (loop_cnt_mfn - loop_cnt_align) * CPU_CLK + (systick_align - systick_mfn);
#endif

    /* ===== STEP 6: Extract Embedding ===== */
    int emb_dim = MIN(emb_output->dims->data[1], EMBEDDING_OUTPUT_DIM);
    DBG_VERBOSE("  Extracting embedding: type=%d, dim=%d\n", emb_output->type, emb_dim);

    if (emb_output->type == kTfLiteFloat32) {
        memcpy(embedding_msg->embedding, emb_output->data.f, emb_dim * sizeof(float));
    } else if (emb_output->type == kTfLiteInt8) {
        float scale = emb_output->params.scale;
        int32_t zero_point = emb_output->params.zero_point;
        int8_t *quant_data = emb_output->data.int8;
        for (int i = 0; i < emb_dim; i++) {
            embedding_msg->embedding[i] = (quant_data[i] - zero_point) * scale;
        }
    } else if (emb_output->type == kTfLiteUInt8) {
        float scale = emb_output->params.scale;
        int32_t zero_point = emb_output->params.zero_point;
        uint8_t *quant_data = emb_output->data.uint8;
        for (int i = 0; i < emb_dim; i++) {
            embedding_msg->embedding[i] = (quant_data[i] - zero_point) * scale;
        }
    }

    /* L2 Normalization */
    normalize_embedding(embedding_msg->embedding, emb_dim);

    /* ===== STEP 7: Fill Message Metadata ===== */
    embedding_msg->face_id = 0;

    uint32_t systick, loop_cnt;
    SystemGetTick(&systick, &loop_cnt);
    embedding_msg->timestamp = loop_cnt * CPU_CLK + systick;

    embedding_msg->confidence = best_face->score;
    embedding_msg->quality = estimate_face_quality(best_face->landmarks);

    embedding_msg->bbox.x = (uint16_t)best_face->bbox.x;
    embedding_msg->bbox.y = (uint16_t)best_face->bbox.y;
    embedding_msg->bbox.width = (uint16_t)best_face->bbox.w;
    embedding_msg->bbox.height = (uint16_t)best_face->bbox.h;

    estimate_face_pose(
        best_face->landmarks,
        &embedding_msg->pose.yaw,
        &embedding_msg->pose.pitch,
        &embedding_msg->pose.roll);

    /* Copy landmarks */
    for (int i = 0; i < 5; i++) {
        embedding_msg->landmarks[i].x = best_face->landmarks[i].x;
        embedding_msg->landmarks[i].y = best_face->landmarks[i].y;
    }

    /* NOTE: pack_face_embedding_msg() removed — it was for binary UART protocol
     * and its internal memset() destroyed bbox/landmarks/quality data that
     * face_invoke.hpp reads for JSON output. */
    DBG_VERBOSE("  Embedding extracted (%dD), quality=%.2f\n", emb_dim, embedding_msg->quality);

    /* JSON output is now handled by face_invoke.hpp event_reply() */

    scrfd_free_dets(faces);

#if TOTAL_STEP_TICK
    SystemGetTick(&systick_2, &loop_cnt_2);
    uint32_t algo_tick = (loop_cnt_2 - loop_cnt_1) * CPU_CLK + (systick_1 - systick_2);
    uint32_t total_ms = algo_tick / 24000;

    xprintf("[Perf] SCRFD=%lums MFN=%lums Total=%lums (%.1f FPS)\n",
            tick_scrfd / 24000,
            tick_mobilefacenet / 24000,
            total_ms,
            total_ms > 0 ? 1000.0f / total_ms : 0.0f);
#endif

    return 0;
}

int cv_face_detect_only(uint8_t *frame_data, uint32_t frame_width, uint32_t frame_height,
                         struct_algoResult *alg_result)
{
    /* Lightweight detection-only path: runs SCRFD (~4ms) without MobileFaceNet (~15ms).
     * Used for skip frames to keep bbox updated while reducing latency. */

    if (g_face_emb_init == 0) return -1;

    uint32_t img_w = frame_width;
    uint32_t img_h = frame_height;

    /* STEP 0: Convert YUV422P → RGB for SCRFD */
    SCB_InvalidateDCache_by_Addr((uint32_t*)frame_data, img_w * img_h * 2);

    el_img_t src_img = {};
    src_img.data = frame_data;
    src_img.size = img_w * img_h * 2;
    src_img.width = (uint16_t)img_w;
    src_img.height = (uint16_t)img_h;
    src_img.format = EL_PIXEL_FORMAT_YUV422;
    src_img.rotate = EL_PIXEL_ROTATE_0;

    el_img_t dst_img = {};
    dst_img.data = (uint8_t *)fd_resized_img;
    dst_img.size = FD_INPUT_TENSOR_WIDTH * FD_INPUT_TENSOR_HEIGHT * 3;
    dst_img.width = FD_INPUT_TENSOR_WIDTH;
    dst_img.height = FD_INPUT_TENSOR_HEIGHT;
    dst_img.format = EL_PIXEL_FORMAT_RGB888;
    dst_img.rotate = EL_PIXEL_ROTATE_0;

    edgelab::el_img_convert(&src_img, &dst_img);

    /* STEP 1: Copy to SCRFD input tensor */
    float scale_w = (float)img_w / FD_INPUT_TENSOR_WIDTH;
    float scale_h = (float)img_h / FD_INPUT_TENSOR_HEIGHT;

    if (fd_input->type == kTfLiteInt8) {
        uint8_t *src = (uint8_t *)fd_resized_img;
        int8_t *dst = fd_input->data.int8;
        int8_t zp = (int8_t)fd_input->params.zero_point;
        for (int i = 0; i < fd_input->bytes; i++) {
            dst[i] = (int8_t)((int)src[i] + zp);
        }
    } else {
        memcpy(fd_input->data.uint8, (uint8_t *)fd_resized_img,
               FD_INPUT_TENSOR_WIDTH * FD_INPUT_TENSOR_HEIGHT * FD_INPUT_TENSOR_CHANNEL);
    }

    /* STEP 2: Run SCRFD */
    if (fd_input->data.data) {
        SCB_CleanDCache_by_Addr((uint32_t*)fd_input->data.data, fd_input->bytes);
    }

    TfLiteStatus invoke_status = fd_int_ptr->Invoke();
    if (invoke_status != kTfLiteOk) return -1;

    /* Invalidate D-Cache for SCRFD outputs */
    for (int i = 0; i < SCRFD_NUM_STRIDES; i++) {
        if (fd_score_tensors[i] && fd_score_tensors[i]->data.data)
            SCB_InvalidateDCache_by_Addr((uint32_t*)fd_score_tensors[i]->data.data, fd_score_tensors[i]->bytes);
        if (fd_bbox_tensors[i] && fd_bbox_tensors[i]->data.data)
            SCB_InvalidateDCache_by_Addr((uint32_t*)fd_bbox_tensors[i]->data.data, fd_bbox_tensors[i]->bytes);
        if (fd_kps_tensors[i] && fd_kps_tensors[i]->data.data)
            SCB_InvalidateDCache_by_Addr((uint32_t*)fd_kps_tensors[i]->data.data, fd_kps_tensors[i]->bytes);
    }

    /* STEP 3: Post-processing */
    /* Set coordinate mapping: simple resize (no letterbox padding) */
    scrfd_net.scale_x = scale_w;
    scrfd_net.scale_y = scale_h;
    scrfd_net.pad_x = 0;
    scrfd_net.pad_y = 0;

    int num_faces = 0;
    auto faces = scrfd_detect(&scrfd_net, (int)img_w, (int)img_h, &num_faces);

    if (num_faces <= 0) {
        alg_result->num_tracked_human_targets = 0;
        return 0;
    }

    scrfd_face *best_face = scrfd_get_best_face(faces, MIN_FACE_SIZE);
    if (best_face == nullptr || best_face->score <= 0) {
        scrfd_free_dets(faces);
        alg_result->num_tracked_human_targets = 0;
        return 0;
    }

    /* Fill detection result */
    alg_result->num_tracked_human_targets = 1;
    alg_result->ht[0].upper_body_score = (uint32_t)(best_face->score * 100);
    alg_result->ht[0].upper_body_bbox.x = (uint32_t)best_face->bbox.x;
    alg_result->ht[0].upper_body_bbox.y = (uint32_t)best_face->bbox.y;
    alg_result->ht[0].upper_body_bbox.width = (uint32_t)best_face->bbox.w;
    alg_result->ht[0].upper_body_bbox.height = (uint32_t)best_face->bbox.h;

    scrfd_free_dets(faces);
    return 0;
}

int cv_face_embedding_deinit()
{
    /* No camera cleanup needed — sscma_micro owns the camera pipeline */
    xprintf("Face embedding deinitialized\n");
    return 0;
}
