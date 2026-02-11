/*
 * face_embedding_protocol.c
 *
 *  Created on: Oct 23, 2025
 *      Author: Face Recognition App
 *
 *  Description: Utility functions for face embedding processing
 */

#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include "face_embedding_protocol.h"

void normalize_embedding(float *embedding, int dim) {
    float norm = 0.0f;
    for (int i = 0; i < dim; i++) {
        norm += embedding[i] * embedding[i];
    }
    norm = sqrtf(norm);

    if (norm > 1e-6f) {
        for (int i = 0; i < dim; i++) {
            embedding[i] /= norm;
        }
    }
}
