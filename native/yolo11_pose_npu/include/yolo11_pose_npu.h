#ifndef TWOPOINT_YOLO11_POSE_NPU_H
#define TWOPOINT_YOLO11_POSE_NPU_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define POSE_INPUT_WIDTH 640
#define POSE_INPUT_HEIGHT 640
#define POSE_INPUT_CHANNELS 3
#define POSE_KEYPOINT_COUNT 5

typedef struct PoseDetection {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    float keypoints[POSE_KEYPOINT_COUNT * 3];
} PoseDetection;

typedef struct PoseNativeTiming {
    uint64_t preprocess_ns;
    uint64_t input_flush_ns;
    uint64_t network_ns;
    uint64_t output_map_ns;
    uint64_t decode_ns;
    uint64_t output_unmap_ns;
    uint64_t result_copy_ns;
    uint64_t total_ns;
} PoseNativeTiming;

void *pose_create(const char *model_path, float score_threshold, float nms_threshold);

int pose_infer_rgb640(
    void *context,
    const uint8_t *rgb_data,
    size_t rgb_size,
    PoseDetection *detections,
    int max_detections
);

int pose_infer_bgr(
    void *context,
    const uint8_t *bgr_data,
    size_t bgr_size,
    int width,
    int height,
    size_t row_stride,
    PoseDetection *detections,
    int max_detections
);

int pose_last_timing(void *context, PoseNativeTiming *timing);

uint32_t pose_driver_version(void *context);

const char *pose_last_error(void *context);

void pose_destroy(void *context);

#ifdef __cplusplus
}
#endif

#endif
