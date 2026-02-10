/*
 * sscma_face.h
 *
 * SSCMA with Face Recognition support
 * Combines SSCMA AT command interface with face embedding capabilities
 *
 * Features:
 *   - Standard SSCMA AT commands for object detection
 *   - AT+FACE command for face recognition mode
 *   - SCRFD (160x160) for face detection with 5-point landmarks
 *   - MobileFaceNet (112x112) for 128D embedding extraction
 */

#ifndef SSCMA_FACE_H_
#define SSCMA_FACE_H_

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Face mode states */
typedef enum {
    FACE_MODE_OFF = 0,      /* Standard SSCMA mode (object detection) */
    FACE_MODE_ON = 1,       /* Face embedding mode (SCRFD + MobileFaceNet) */
} face_mode_t;

/* Face embedding result */
typedef struct {
    float embedding[128];   /* 128D face embedding (L2 normalized) */
    int bbox_x;            /* Face bounding box */
    int bbox_y;
    int bbox_w;
    int bbox_h;
    float confidence;       /* Detection confidence (0-1) */
    float quality;          /* Face quality score (0-1) */
    float landmarks[10];    /* 5-point landmarks (x1,y1,...,x5,y5) */
    bool valid;             /* True if face detected and embedding extracted */
} face_embedding_result_t;

/* API */
int face_mode_init(void);
int face_mode_deinit(void);
int face_mode_set(face_mode_t mode);
face_mode_t face_mode_get(void);

/* Called from main loop when face mode is enabled */
int face_mode_process_frame(face_embedding_result_t* result);

#ifdef __cplusplus
}
#endif

#endif /* SSCMA_FACE_H_ */
