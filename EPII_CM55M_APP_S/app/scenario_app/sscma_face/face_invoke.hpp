/*
 * face_invoke.hpp
 *
 * Face Mode Invoke Handler for SSCMA (sscma_micro compatible version)
 *
 * When face mode is enabled, this replaces the standard INVOKE callback
 * to run face detection (SCRFD) + embedding (MobileFaceNet) instead of
 * object detection.
 *
 * Output JSON format:
 * {
 *   "type": 1,
 *   "name": "INVOKE",
 *   "code": 0,
 *   "data": {
 *     "mode": "face",
 *     "count": 1,
 *     "faces": [{
 *       "box": [x, y, w, h],
 *       "score": 95,
 *       "quality": 0.85,
 *       "landmarks": [[x1,y1], [x2,y2], ...],
 *       "embedding": [f1, f2, ..., f128]
 *     }],
 *     "resolution": [width, height]
 *   }
 * }
 */

#pragma once

#ifdef SSCMA_FACE

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>

#include "sscma/definations.hpp"
#include "sscma/static_resource.hpp"
#include "sscma/utility.hpp"

/* Include face embedding pipeline */
extern "C" {
#include "cvapp_face_embedding.h"
#include "common_config.h"
#include "face_embedding_protocol.h"
#include "WE2_core.h"  /* SCB_InvalidateDCache_by_Addr */
}
#include "send_result.h"

namespace sscma::callback {

using namespace sscma::utility;

/* Format float without %f (newlib-nano doesn't support float formatting).
 * Writes "[-]int.frac" with 'decimals' digits after the point. */
static int fmt_fixed(char* buf, int size, float val, int decimals) {
    int neg = (val < 0);
    if (neg) val = -val;
    int mult = 1;
    for (int d = 0; d < decimals; d++) mult *= 10;
    int scaled = (int)(val * mult + 0.5f);
    return snprintf(buf, size, "%s%d.%0*d",
                    neg ? "-" : "", scaled / mult, decimals, scaled % mult);
}

class FaceInvoke final : public std::enable_shared_from_this<FaceInvoke> {
public:
    std::shared_ptr<FaceInvoke> getptr() { return shared_from_this(); }

    [[nodiscard]] static std::shared_ptr<FaceInvoke> create(
        std::string cmd, int32_t n_times, bool differed, bool results_only, void* caller) {
        return std::shared_ptr<FaceInvoke>{
            new FaceInvoke{std::move(cmd), n_times, differed, results_only, caller}
        };
    }

    ~FaceInvoke() {
        static_resource->is_invoke = false;
    }

    inline void run() { prepare(); }

protected:
    FaceInvoke(std::string cmd, int32_t n_times, bool differed, bool results_only, void* caller)
        : _cmd{cmd},
          _n_times{n_times},
          _results_only{results_only},
          _caller{caller},
          _task_id{static_resource->current_task_id.load(std::memory_order_seq_cst)},
          _times{0},
          _ret{EL_OK} {
        static_resource->is_invoke = true;
    }

private:
    void prepare() {
        /* Send immediate reply so ESP32 sscma_client doesn't time out */
        direct_reply();

        /* Initialize face embedding models if not already done */
        if (!initFaceModels()) [[unlikely]] {
            return;
        }

        /* Start event loop - camera stream is started in each event_loop iteration */
        static_resource->executor->add_task(
            [_this = std::move(getptr())](const std::atomic<bool>&) {
                _this->event_loop();
            });
    }

    bool initFaceModels() {
        static bool models_initialized = false;
        if (!models_initialized) {
            int ret = cv_face_embedding_init(
                true, true,
                FACE_DETECT_FLASH_ADDR,
                FACE_EMBEDDING_FLASH_ADDR);
            if (ret != 0) {
                EL_LOGW("[FaceInvoke] Face embedding init failed: %d", ret);
                _ret = EL_EIO;
                return false;
            }
            models_initialized = true;
            EL_LOGI("[FaceInvoke] Face models initialized");
        }
        return true;
    }

    void direct_reply() {
        auto ss{concat_strings("\r{\"type\": 0, \"name\": \"",
                               _cmd,
                               "\", \"code\": ",
                               std::to_string(_ret),
                               ", \"data\": {\"mode\": \"face\"}}\n")};
        static_cast<Transport*>(_caller)->send_bytes(ss.c_str(), ss.size());
    }

    void event_reply(const struct_algoResult& algo_result,
                     const face_embedding_msg_t& embedding_result,
                     int width, int height,
                     const el_img_t& jpeg_frame) {
        /* Use JPEG frame obtained from sscma_micro's camera driver
         * (via get_processed_frame) before stop_stream was called */
        el_img_t jpeg_img = jpeg_frame;
        el_img_t* jpeg_ptr = (jpeg_img.data && jpeg_img.size > 0) ? &jpeg_img : nullptr;

        /* Build JSON response */
        std::string response = concat_strings(
            "\r{\"type\": 1, \"name\": \"",
            _cmd,
            "\", \"code\": ",
            std::to_string(_ret),
            ", \"data\": {\"mode\": \"face\", \"count\": ",
            std::to_string(_times));

        /* Include image only for UART caller (face registration app).
         * SPI caller (ESP32) doesn't need image — avoids RX buffer overflow. */
        bool include_image = (static_cast<Transport*>(_caller)->type == EL_TRANSPORT_UART);
        response += ", ";
        response += img_2_json_str(include_image ? jpeg_ptr : nullptr);
        response += ", ";
        response += concat_strings("\"resolution\": [",
                                   std::to_string(width), ", ",
                                   std::to_string(height), "]");

        /* Add faces array */
        if (algo_result.num_tracked_human_targets > 0) {
            /* Build face JSON with embedding.
             * Uses fmt_fixed() instead of %f because newlib-nano
             * doesn't support float formatting in snprintf. */
            char face_buf[4096];  // Enough for one face with 128D embedding
            int len = snprintf(face_buf, sizeof(face_buf),
                ", \"faces\": [{\"box\": [%d, %d, %d, %d], \"score\": %d, \"quality\": ",
                embedding_result.bbox.x, embedding_result.bbox.y,
                embedding_result.bbox.width, embedding_result.bbox.height,
                (int)(embedding_result.confidence * 100));
            len += fmt_fixed(face_buf + len, sizeof(face_buf) - len,
                             embedding_result.quality, 2);
            len += snprintf(face_buf + len, sizeof(face_buf) - len,
                ", \"landmarks\": [[%d, %d], [%d, %d], [%d, %d], [%d, %d], [%d, %d]], "
                "\"embedding\": [",
                (int)embedding_result.landmarks[0].x, (int)embedding_result.landmarks[0].y,
                (int)embedding_result.landmarks[1].x, (int)embedding_result.landmarks[1].y,
                (int)embedding_result.landmarks[2].x, (int)embedding_result.landmarks[2].y,
                (int)embedding_result.landmarks[3].x, (int)embedding_result.landmarks[3].y,
                (int)embedding_result.landmarks[4].x, (int)embedding_result.landmarks[4].y);

            /* Add embedding values using fmt_fixed (no %f) */
            for (int i = 0; i < EMBEDDING_OUTPUT_DIM; i++) {
                len += fmt_fixed(face_buf + len, sizeof(face_buf) - len,
                                 embedding_result.embedding[i], 4);
                if (i < EMBEDDING_OUTPUT_DIM - 1) {
                    len += snprintf(face_buf + len, sizeof(face_buf) - len, ", ");
                }
            }
            len += snprintf(face_buf + len, sizeof(face_buf) - len, "]}]");

            response += face_buf;
        } else {
            response += ", \"faces\": []";
        }

        response += "}}\n";

        static_cast<Transport*>(_caller)->send_bytes(response.c_str(), response.size());
    }

    /* Run embedding every N frames to reduce latency.
     * Detection runs every frame, embedding only on EMBED_INTERVAL frames.
     * This frees the executor for AT commands between heavy frames. */
    static constexpr int EMBED_INTERVAL = 3;

    void event_loop() {
        if ((_n_times >= 0) & (_times++ >= _n_times)) [[unlikely]]
            return;

        /* FIX 1: Check stop token early — allows AT+BREAK to interrupt immediately */
        if (static_resource->current_task_id.load(std::memory_order_seq_cst) != _task_id) [[unlikely]]
            return;

        auto camera = static_resource->device->get_camera();
        el_img_t frame = {};

        _ret = camera->start_stream();
        if (_ret != EL_OK) [[unlikely]] {
            EL_LOGW("[FaceInvoke] Camera start failed: %d", _ret);
            return;
        }

        _ret = camera->get_frame(&frame);
        if (_ret != EL_OK) [[unlikely]] {
            camera->stop_stream();
            EL_LOGW("[FaceInvoke] Get frame failed: %d", _ret);
            return;
        }

        if (frame.data && frame.size > 0) {
            SCB_InvalidateDCache_by_Addr((uint32_t*)frame.data, frame.size);
        }

        el_img_t jpeg_frame = {};
        camera->get_processed_frame(&jpeg_frame);

        /* FIX 2: SCRFD runs every frame (~4ms), MobileFaceNet only every EMBED_INTERVAL frames (~15ms).
         * Skip frames get fresh bbox from SCRFD but reuse last embedding.
         * This keeps detection responsive while reducing avg frame time. */
        struct_algoResult algo_result = {};
        face_embedding_msg_t embedding_result = {};
        bool run_embedding = (_times % EMBED_INTERVAL == 0);

        if (run_embedding) {
            /* Full pipeline: SCRFD + alignment + MobileFaceNet */
            int ret = cv_face_embedding_run(frame.data, frame.width, frame.height,
                                             &algo_result, &embedding_result);
            if (ret != 0) {
                EL_LOGW("[FaceInvoke] Face embedding run returned: %d", ret);
            }
            /* Cache embedding for skip frames */
            _last_embedding_result = embedding_result;
        } else {
            /* Detection only: fresh bbox, reuse last embedding */
            cv_face_detect_only(frame.data, frame.width, frame.height, &algo_result);
            embedding_result = _last_embedding_result;
            /* Update bbox in embedding result to match fresh detection */
            if (algo_result.num_tracked_human_targets > 0) {
                embedding_result.bbox.x = (uint16_t)algo_result.ht[0].upper_body_bbox.x;
                embedding_result.bbox.y = (uint16_t)algo_result.ht[0].upper_body_bbox.y;
                embedding_result.bbox.width = (uint16_t)algo_result.ht[0].upper_body_bbox.width;
                embedding_result.bbox.height = (uint16_t)algo_result.ht[0].upper_body_bbox.height;
                embedding_result.confidence = algo_result.ht[0].upper_body_score / 100.0f;
            }
        }

        camera->stop_stream();

        /* FIX 1 (continued): Check again after inference — catch commands that arrived during processing */
        if (static_resource->current_task_id.load(std::memory_order_seq_cst) != _task_id) [[unlikely]]
            return;

        int width = frame.width;
        int height = frame.height;

        event_reply(algo_result, embedding_result, width, height, jpeg_frame);

        /* FIX 3: Yield before scheduling next frame — gives pending AT commands
         * a chance to execute between inference cycles.
         * Without this, face loop monopolizes the executor. */
        static_resource->device->yield();

        /* Schedule next iteration */
        static_resource->executor->add_task(
            [_this = std::move(getptr())](const std::atomic<bool>& stop_token) {
                if (stop_token.load(std::memory_order_seq_cst)) [[unlikely]]
                    return;
                _this->event_loop();
            });
    }

private:
    std::string _cmd;
    int32_t _n_times;
    bool _results_only;
    void* _caller;

    std::size_t _task_id;
    int32_t _times;
    el_err_code_t _ret;

    /* Cached embedding for skip frames (FIX 2) */
    face_embedding_msg_t _last_embedding_result = {};
};

}  // namespace sscma::callback

#endif  // SSCMA_FACE
