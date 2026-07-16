/*
 * face_alignment.c
 *
 * Face alignment implementation using 5-point landmarks.
 *
 * Uses similarity transform (rotation + uniform scale + translation)
 * to align detected faces to the ArcFace canonical position.
 *
 * Created for Grove Vision AI Module V2 face recognition project.
 */

#include "face_alignment.h"
#include <math.h>
#include <string.h>

/*
 * ArcFace canonical landmark positions for 112x112 aligned face.
 * These reference positions are the standard used for training
 * face recognition models like ArcFace, CosFace, etc.
 */
static const scrfd_point2f REFERENCE_LANDMARKS[SCRFD_NUM_LANDMARKS] = {
    {38.2946f, 51.6963f},   /* Left eye */
    {73.5318f, 51.5014f},   /* Right eye */
    {56.0252f, 71.7366f},   /* Nose tip */
    {41.5493f, 92.3655f},   /* Left mouth corner */
    {70.7299f, 92.2041f}    /* Right mouth corner */
};

void compute_face_alignment(
    const scrfd_point2f* landmarks,
    affine_transform_t* transform
) {
    if (landmarks == NULL || transform == NULL) return;

    /*
     * Least-squares similarity transform over all 5 landmarks (the standard
     * ArcFace alignment, closed form, no SVD needed).
     *
     * The earlier version solved the transform from the two eyes alone. Two
     * points only pin down roll, so any out-of-plane yaw or a noisy single
     * landmark left the crop tilted or off-centre -- exactly the "face isn't
     * upright" symptom. Fitting all 5 points (eyes, nose, mouth corners)
     * distributes landmark noise and frames the face the way the reference
     * template expects.
     *
     * For src_i -> dst_i we minimise sum|dst_i - (s*R*src_i + t)|^2 with a 2D
     * similarity (rotation + uniform scale + translation):
     *   sx = src - mean(src),  dx = dst - mean(dst)
     *   a = sum(dx.x*sx.x + dx.y*sx.y),  b = sum(dx.y*sx.x - dx.x*sx.y)
     *   d = sum(sx.x^2 + sx.y^2)
     *   s*R = (1/d) * [[a, -b], [b, a]],  t = mean(dst) - s*R*mean(src)
     */
    const int N = SCRFD_NUM_LANDMARKS;

    float src_mx = 0.0f, src_my = 0.0f, dst_mx = 0.0f, dst_my = 0.0f;
    for (int i = 0; i < N; i++) {
        src_mx += landmarks[i].x;          src_my += landmarks[i].y;
        dst_mx += REFERENCE_LANDMARKS[i].x; dst_my += REFERENCE_LANDMARKS[i].y;
    }
    src_mx /= N; src_my /= N; dst_mx /= N; dst_my /= N;

    float a = 0.0f, b = 0.0f, d = 0.0f;
    for (int i = 0; i < N; i++) {
        float sx = landmarks[i].x - src_mx;
        float sy = landmarks[i].y - src_my;
        float dx = REFERENCE_LANDMARKS[i].x - dst_mx;
        float dy = REFERENCE_LANDMARKS[i].y - dst_my;
        a += dx * sx + dy * sy;
        b += dy * sx - dx * sy;
        d += sx * sx + sy * sy;
    }

    if (d < 1e-6f) {
        /* Degenerate landmarks (all coincident); fall back to identity. */
        transform->m[0] = 1.0f; transform->m[1] = 0.0f; transform->m[2] = 0.0f;
        transform->m[3] = 0.0f; transform->m[4] = 1.0f; transform->m[5] = 0.0f;
        return;
    }

    float sc  = a / d;   /* s * cos(theta) */
    float ss  = b / d;   /* s * sin(theta) */

    transform->m[0] = sc;
    transform->m[1] = -ss;
    transform->m[2] = dst_mx - (sc * src_mx - ss * src_my);

    transform->m[3] = ss;
    transform->m[4] = sc;
    transform->m[5] = dst_my - (ss * src_mx + sc * src_my);
}

void invert_affine_transform(
    const affine_transform_t* forward,
    affine_transform_t* inverse
) {
    if (forward == NULL || inverse == NULL) return;

    /*
     * For affine transform [a, b, c; d, e, f]:
     *
     * The 2x2 rotation/scale part has inverse:
     *   [a, b]^-1 = 1/det * [e, -b; -d, a]
     *
     * where det = a*e - b*d
     *
     * Full inverse transform:
     *   x = inv_a * (x' - c) + inv_b * (y' - f)
     *   y = inv_d * (x' - c) + inv_e * (y' - f)
     */

    float a = forward->m[0];
    float b = forward->m[1];
    float c = forward->m[2];
    float d = forward->m[3];
    float e = forward->m[4];
    float f = forward->m[5];

    float det = a * e - b * d;
    float inv_det = 1.0f / (det + 1e-8f);

    inverse->m[0] = e * inv_det;
    inverse->m[1] = -b * inv_det;
    inverse->m[2] = (b * f - e * c) * inv_det;

    inverse->m[3] = -d * inv_det;
    inverse->m[4] = a * inv_det;
    inverse->m[5] = (d * c - a * f) * inv_det;
}

void apply_face_alignment(
    const uint8_t* src_image,
    int src_w,
    int src_h,
    uint8_t* dst_image,
    const affine_transform_t* transform
) {
    if (src_image == NULL || dst_image == NULL || transform == NULL) return;

    /*
     * Source image format: RGB interleaved (from yuv422_to_rgb888)
     * [R, G, B, R, G, B, ...]
     *
     * Destination image format: RGB interleaved
     * [R, G, B, R, G, B, ...]
     */

    /* Compute inverse transform for backward mapping */
    affine_transform_t inv_transform;
    invert_affine_transform(transform, &inv_transform);

    float inv_a = inv_transform.m[0];
    float inv_b = inv_transform.m[1];
    float inv_c = inv_transform.m[2];
    float inv_d = inv_transform.m[3];
    float inv_e = inv_transform.m[4];
    float inv_f = inv_transform.m[5];

    /* Iterate over destination pixels */
    for (int dy = 0; dy < ALIGNED_FACE_HEIGHT; dy++) {
        for (int dx = 0; dx < ALIGNED_FACE_WIDTH; dx++) {
            /* Backward map: find source coordinates for this destination pixel */
            float sx = inv_a * dx + inv_b * dy + inv_c;
            float sy = inv_d * dx + inv_e * dy + inv_f;

            /* Destination pixel index (RGB interleaved) */
            int dst_idx = (dy * ALIGNED_FACE_WIDTH + dx) * 3;

            /* Integer source coordinates for bilinear interpolation */
            int ix = (int)floorf(sx);
            int iy = (int)floorf(sy);

            /* Check bounds with margin for interpolation */
            if (ix >= 0 && ix < src_w - 1 && iy >= 0 && iy < src_h - 1) {
                /* Bilinear interpolation weights */
                float fx = sx - ix;
                float fy = sy - iy;
                float w00 = (1.0f - fx) * (1.0f - fy);
                float w01 = fx * (1.0f - fy);
                float w10 = (1.0f - fx) * fy;
                float w11 = fx * fy;

                /* Source pixel indices (RGB interleaved: 3 bytes per pixel) */
                int src_base00 = (iy * src_w + ix) * 3;
                int src_base01 = src_base00 + 3;
                int src_base10 = src_base00 + src_w * 3;
                int src_base11 = src_base10 + 3;

                /* Interpolate R channel */
                float r_val = w00 * src_image[src_base00] +
                              w01 * src_image[src_base01] +
                              w10 * src_image[src_base10] +
                              w11 * src_image[src_base11];
                dst_image[dst_idx] = (uint8_t)(r_val + 0.5f);

                /* Interpolate G channel */
                float g_val = w00 * src_image[src_base00 + 1] +
                              w01 * src_image[src_base01 + 1] +
                              w10 * src_image[src_base10 + 1] +
                              w11 * src_image[src_base11 + 1];
                dst_image[dst_idx + 1] = (uint8_t)(g_val + 0.5f);

                /* Interpolate B channel */
                float b_val = w00 * src_image[src_base00 + 2] +
                              w01 * src_image[src_base01 + 2] +
                              w10 * src_image[src_base10 + 2] +
                              w11 * src_image[src_base11 + 2];
                dst_image[dst_idx + 2] = (uint8_t)(b_val + 0.5f);
            } else {
                /* Out of bounds - use black padding */
                dst_image[dst_idx] = 0;
                dst_image[dst_idx + 1] = 0;
                dst_image[dst_idx + 2] = 0;
            }
        }
    }
}

float estimate_face_quality(const scrfd_point2f* landmarks) {
    if (landmarks == NULL) return 0.0f;

    /* Roll is deliberately NOT scored. compute_face_alignment() rotates the face
     * upright from the eye vector, so in-plane tilt is fully corrected before the
     * crop reaches the model -- penalising it here punishes something the very
     * next stage undoes. What the eyes-only (2-point) similarity transform CANNOT
     * remove is out-of-plane yaw, so that is what quality measures.
     *
     * The previous version multiplied a roll term, a yaw term and a harsh mouth
     * term together; a normal frontal face scored ~0.07, far below the (never
     * wired) MIN_FACE_QUALITY gate. */
    const scrfd_point2f* le = &landmarks[SCRFD_LM_LEFT_EYE];
    const scrfd_point2f* re = &landmarks[SCRFD_LM_RIGHT_EYE];
    float eye_dx = re->x - le->x;
    float eye_dy = re->y - le->y;
    float interocular = sqrtf(eye_dx * eye_dx + eye_dy * eye_dy);  /* roll-invariant */
    if (interocular < 1.0f) return 0.0f;

    float eye_cx = (le->x + re->x) * 0.5f;

    /* Nose horizontal offset from the eye centre is the primary yaw proxy; the
     * mouth centre offset confirms it at quarter weight. Both are normalised by
     * the interocular distance so the score is scale invariant. */
    float nose_off  = fabsf(landmarks[SCRFD_LM_NOSE].x - eye_cx) / interocular;
    float mouth_cx  = (landmarks[SCRFD_LM_LEFT_MOUTH].x + landmarks[SCRFD_LM_RIGHT_MOUTH].x) * 0.5f;
    float mouth_off = fabsf(mouth_cx - eye_cx) / interocular;

    /* Frontal keeps both offsets near 0 -> quality ~1. A ~45 deg profile pushes
     * the nose offset past ~0.5 -> quality ~0. */
    float yaw = 0.75f * nose_off + 0.25f * mouth_off;
    float quality = 1.0f - 1.5f * yaw;

    return fmaxf(0.0f, fminf(1.0f, quality));
}

void estimate_face_pose(
    const scrfd_point2f* landmarks,
    float* yaw,
    float* pitch,
    float* roll
) {
    if (landmarks == NULL) return;

    /* Eye vector for roll estimation */
    float eye_dx = landmarks[SCRFD_LM_RIGHT_EYE].x - landmarks[SCRFD_LM_LEFT_EYE].x;
    float eye_dy = landmarks[SCRFD_LM_RIGHT_EYE].y - landmarks[SCRFD_LM_LEFT_EYE].y;

    if (roll != NULL) {
        /* Roll: rotation in image plane (degrees) */
        *roll = atan2f(eye_dy, eye_dx) * 180.0f / 3.14159265f;
    }

    /* Eye center for yaw/pitch estimation */
    float eye_center_x = (landmarks[SCRFD_LM_LEFT_EYE].x + landmarks[SCRFD_LM_RIGHT_EYE].x) / 2.0f;
    float eye_center_y = (landmarks[SCRFD_LM_LEFT_EYE].y + landmarks[SCRFD_LM_RIGHT_EYE].y) / 2.0f;
    float eye_width = sqrtf(eye_dx * eye_dx + eye_dy * eye_dy);

    if (yaw != NULL) {
        /* Yaw: left-right rotation (rough estimate from nose position) */
        float nose_offset = landmarks[SCRFD_LM_NOSE].x - eye_center_x;
        float norm_offset = nose_offset / (eye_width / 2.0f + 1e-6f);
        *yaw = norm_offset * 45.0f;  /* Scale to approximate degrees */
    }

    if (pitch != NULL) {
        /* Pitch: up-down rotation (rough estimate from nose-eye distance) */
        float nose_eye_dist = landmarks[SCRFD_LM_NOSE].y - eye_center_y;
        float expected_dist = eye_width * 0.6f;  /* Typical ratio */
        float ratio = nose_eye_dist / (expected_dist + 1e-6f);
        *pitch = (ratio - 1.0f) * 30.0f;  /* Scale deviation to degrees */
    }
}
