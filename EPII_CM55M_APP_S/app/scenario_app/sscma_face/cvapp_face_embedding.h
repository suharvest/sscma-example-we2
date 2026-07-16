/*
 * cvapp_face_embedding.h
 *
 *  Created on: Dec 11, 2024
 *      Author: Face Embedding App (MobileFaceNet 128D)
 */

#ifndef APP_SCENARIO_APP_TFLM_FACE_EMBEDDING_CVAPP_FACE_EMBEDDING_H_
#define APP_SCENARIO_APP_TFLM_FACE_EMBEDDING_CVAPP_FACE_EMBEDDING_H_

#include "spi_protocol.h"
#include "face_embedding_protocol.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize face embedding models (detection + embedding)
 *
 * @param security_enable Enable security features
 * @param privilege_enable Enable privilege features
 * @param fd_model_addr Flash address of face detection model (SCRFD)
 * @param embedding_model_addr Flash address of face embedding model (MobileFaceNet)
 * @return int 0 on success, -1 on error
 */
int cv_face_embedding_init(bool security_enable, bool privilege_enable,
                            uint32_t fd_model_addr, uint32_t embedding_model_addr);

/**
 * @brief Run face embedding pipeline
 *
 * Takes a YUV422P frame from sscma_micro's camera, converts to RGB888 using
 * el_img_convert(), performs face detection using SCRFD, aligns detected faces
 * using 5-point landmarks, and extracts 128D embeddings using MobileFaceNet.
 *
 * @param frame_data Pointer to YUV422P frame data from camera
 * @param frame_width Frame width
 * @param frame_height Frame height
 * @param alg_result Basic detection result structure
 * @param embedding_result Face embedding result structure (128D)
 * @return int 0 on success, -1 on error
 */
int cv_face_embedding_run(uint8_t *frame_data, uint32_t frame_width, uint32_t frame_height,
                           struct_algoResult *alg_result,
                           face_embedding_msg_t *embedding_result);

/**
 * @brief Run face detection only (skip embedding) for low-latency frames
 */
int cv_face_detect_only(uint8_t *frame_data, uint32_t frame_width, uint32_t frame_height,
                         struct_algoResult *alg_result);

typedef struct {
    uint8_t valid;
    const uint8_t *emb_input_data;
    uint32_t emb_input_bytes;
    int32_t emb_input_type;
    int32_t emb_input_zp;
    float emb_input_scale;
    int32_t emb_input_dims[4];
    int32_t emb_input_dims_count;
    const uint8_t *emb_output_data;
    uint32_t emb_output_bytes;
    int32_t emb_output_type;
    int32_t emb_output_zp;
    float emb_output_scale;
    int32_t emb_output_dims[4];
    int32_t emb_output_dims_count;
} face_debug_tensors_t;

/**
 * @brief Get tensors from the last successful MobileFaceNet invocation.
 *
 * Pointers remain owned by the face embedding runtime and are valid until the
 * next face embedding invocation or model reinitialization.
 */
int cv_face_embedding_get_debug_tensors(face_debug_tensors_t *out);

/**
 * @brief Run MobileFaceNet on a deterministic synthetic input.
 *
 * This bypasses camera, SCRFD, and face alignment. It is intended for backend
 * equivalence diagnostics between device Ethos-U and local TFLite CPU.
 */
int cv_face_embedding_run_fixed_input_test(uint32_t seed);

/**
 * @brief Run MobileFaceNet using an int8 input tensor stored in flash.
 *
 * This bypasses camera, SCRFD, and alignment. The input bytes must already
 * match the model input tensor shape and quantization.
 */
int cv_face_embedding_run_flash_input_test(uint32_t input_flash_addr, uint32_t input_bytes);

/**
 * @brief Run the full pipeline (SCRFD + alignment + MobileFaceNet) against a
 *        YUV422 frame held in flash, standing in for the camera.
 *
 * Bypasses only the sensor and ISP, which makes it possible to tell an image
 * quality problem apart from a detection, alignment, or NPU problem. The frame
 * is read in place through the flash alias and costs no SRAM.
 */
int cv_face_embedding_run_flash_frame_test(uint32_t frame_flash_addr, uint32_t frame_width,
                                           uint32_t frame_height,
                                           struct_algoResult *alg_result,
                                           face_embedding_msg_t *embedding_msg);

/**
 * @brief Borrow the last aligned 112x112 RGB888 face crop.
 *
 * This buffer lives outside the tensor arena, so unlike face_debug_tensors_t's
 * emb_input_data it is still valid after Invoke() has run.
 */
int cv_face_embedding_get_aligned_crop(const uint8_t **out_data, uint32_t *out_bytes);

/**
 * @brief Override face detection confidence threshold at runtime.
 *
 * Passing a value outside [0.01, 1.0] restores the build-time default.
 */
int cv_face_embedding_set_conf_threshold(float threshold);

/**
 * @brief Deinitialize face embedding
 *
 * @return int 0 on success, -1 on error
 */
int cv_face_embedding_deinit();

#ifdef __cplusplus
}
#endif

#endif /* APP_SCENARIO_APP_TFLM_FACE_EMBEDDING_CVAPP_FACE_EMBEDDING_H_ */
