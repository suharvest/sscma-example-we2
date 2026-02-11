/*
 * face_embedding_protocol.h
 *
 *  Created on: Dec 11, 2025
 *      Author: Face Embedding App
 *
 *  Description: Data structures and utilities for face embedding processing
 *
 *  Updated for MobileFaceNet with 128-dimensional embeddings
 */

#ifndef FACE_EMBEDDING_PROTOCOL_H_
#define FACE_EMBEDDING_PROTOCOL_H_

#include <stdint.h>
#include <stddef.h>
#include "common_config.h"

/* Face bounding box structure */
typedef struct {
    uint16_t x;
    uint16_t y;
    uint16_t width;
    uint16_t height;
} face_bbox_t;

/* Face quality metrics */
typedef struct {
    float yaw;      // Head rotation left/right (-180 to 180 degrees)
    float pitch;    // Head tilt up/down (-90 to 90 degrees)
    float roll;     // Head tilt side-to-side (-180 to 180 degrees)
} face_pose_t;

/* 2D landmark point */
typedef struct {
    float x;
    float y;
} face_landmark_t;

/* Face embedding result structure */
typedef struct {
    uint16_t face_id;                           // Tracking ID (0 if not tracked)
    float    embedding[EMBEDDING_OUTPUT_DIM];    // 128-dimensional embedding
    uint32_t timestamp;                         // Frame timestamp (milliseconds)
    float    confidence;                        // Face detection confidence (0.0-1.0)
    float    quality;                           // Face quality score (0.0-1.0)
    face_bbox_t bbox;                           // Face bounding box
    face_pose_t pose;                           // Face pose angles
    face_landmark_t landmarks[5];               // 5-point landmarks
} face_embedding_msg_t;

#define FACE_EMBEDDING_MSG_SIZE    sizeof(face_embedding_msg_t)

/* Function prototypes */
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Normalize embedding vector (L2 normalization)
 *
 * @param embedding Pointer to embedding array (modified in-place)
 * @param dim Dimension of embedding (should be EMBEDDING_OUTPUT_DIM)
 */
void normalize_embedding(float *embedding, int dim);

#ifdef __cplusplus
}
#endif

#endif /* FACE_EMBEDDING_PROTOCOL_H_ */
