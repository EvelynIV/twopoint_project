#ifndef TWOPOINT_YOLO11_POSE_POSTPROCESS_H
#define TWOPOINT_YOLO11_POSE_POSTPROCESS_H

#include <array>
#include <vector>

namespace twopoint::pose {

constexpr int kInputSize = 640;
constexpr int kKeypointCount = 5;
constexpr int kOutputCount = 9;

struct DecodedDetection {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    std::array<float, kKeypointCount * 3> keypoints;
};

std::vector<DecodedDetection> decode_yolo11_pose(
    const std::array<const float *, kOutputCount> &outputs,
    float score_threshold,
    float nms_threshold,
    int max_detections
);

}  // namespace twopoint::pose

#endif
