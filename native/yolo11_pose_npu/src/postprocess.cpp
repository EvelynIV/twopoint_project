#include "postprocess.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <vector>

namespace twopoint::pose {
namespace {

constexpr int kRegMax = 16;
constexpr std::array<int, 3> kStrides{8, 16, 32};

float sigmoid(float value) {
    if (value >= 0.0F) {
        const float exp_value = std::exp(-value);
        return 1.0F / (1.0F + exp_value);
    }
    const float exp_value = std::exp(value);
    return exp_value / (1.0F + exp_value);
}

float dfl_distance(const float *data, int grid_size, int grid_index, int side) {
    const int channel_offset = side * kRegMax;
    float max_logit = -std::numeric_limits<float>::infinity();
    for (int bin = 0; bin < kRegMax; ++bin) {
        max_logit = std::max(max_logit, data[(channel_offset + bin) * grid_size + grid_index]);
    }

    float denominator = 0.0F;
    float weighted_sum = 0.0F;
    for (int bin = 0; bin < kRegMax; ++bin) {
        const float probability = std::exp(
            data[(channel_offset + bin) * grid_size + grid_index] - max_logit
        );
        denominator += probability;
        weighted_sum += static_cast<float>(bin) * probability;
    }
    return denominator > 0.0F ? weighted_sum / denominator : 0.0F;
}

float intersection_over_union(const DecodedDetection &left, const DecodedDetection &right) {
    const float intersection_x1 = std::max(left.x1, right.x1);
    const float intersection_y1 = std::max(left.y1, right.y1);
    const float intersection_x2 = std::min(left.x2, right.x2);
    const float intersection_y2 = std::min(left.y2, right.y2);
    const float intersection_width = std::max(0.0F, intersection_x2 - intersection_x1);
    const float intersection_height = std::max(0.0F, intersection_y2 - intersection_y1);
    const float intersection = intersection_width * intersection_height;
    const float left_area = std::max(0.0F, left.x2 - left.x1) * std::max(0.0F, left.y2 - left.y1);
    const float right_area = std::max(0.0F, right.x2 - right.x1) * std::max(0.0F, right.y2 - right.y1);
    const float union_area = left_area + right_area - intersection;
    return union_area > 0.0F ? intersection / union_area : 0.0F;
}

void decode_scale(
    int stride,
    const float *box_data,
    const float *score_data,
    const float *keypoint_data,
    float score_threshold,
    std::vector<DecodedDetection> &proposals
) {
    const int grid_width = kInputSize / stride;
    const int grid_height = kInputSize / stride;
    const int grid_size = grid_width * grid_height;

    for (int y = 0; y < grid_height; ++y) {
        for (int x = 0; x < grid_width; ++x) {
            const int grid_index = y * grid_width + x;
            const float score = sigmoid(score_data[grid_index]);
            if (score < score_threshold) {
                continue;
            }

            const float center_x = static_cast<float>(x) + 0.5F;
            const float center_y = static_cast<float>(y) + 0.5F;
            DecodedDetection detection{};
            detection.x1 = (center_x - dfl_distance(box_data, grid_size, grid_index, 0)) * stride;
            detection.y1 = (center_y - dfl_distance(box_data, grid_size, grid_index, 1)) * stride;
            detection.x2 = (center_x + dfl_distance(box_data, grid_size, grid_index, 2)) * stride;
            detection.y2 = (center_y + dfl_distance(box_data, grid_size, grid_index, 3)) * stride;
            detection.x1 = std::clamp(detection.x1, 0.0F, static_cast<float>(kInputSize - 1));
            detection.y1 = std::clamp(detection.y1, 0.0F, static_cast<float>(kInputSize - 1));
            detection.x2 = std::clamp(detection.x2, 0.0F, static_cast<float>(kInputSize - 1));
            detection.y2 = std::clamp(detection.y2, 0.0F, static_cast<float>(kInputSize - 1));
            detection.score = score;

            for (int keypoint = 0; keypoint < kKeypointCount; ++keypoint) {
                const int channel = keypoint * 3;
                detection.keypoints[channel] =
                    (static_cast<float>(x) + keypoint_data[channel * grid_size + grid_index] * 2.0F) * stride;
                detection.keypoints[channel + 1] =
                    (static_cast<float>(y) + keypoint_data[(channel + 1) * grid_size + grid_index] * 2.0F) * stride;
                detection.keypoints[channel + 2] = sigmoid(
                    keypoint_data[(channel + 2) * grid_size + grid_index]
                );
            }
            proposals.push_back(detection);
        }
    }
}

}  // namespace

std::vector<DecodedDetection> decode_yolo11_pose(
    const std::array<const float *, kOutputCount> &outputs,
    float score_threshold,
    float nms_threshold,
    int max_detections
) {
    std::vector<DecodedDetection> proposals;
    for (std::size_t scale = 0; scale < kStrides.size(); ++scale) {
        const std::size_t output_offset = scale * 3;
        decode_scale(
            kStrides[scale],
            outputs[output_offset],
            outputs[output_offset + 1],
            outputs[output_offset + 2],
            score_threshold,
            proposals
        );
    }

    std::sort(
        proposals.begin(),
        proposals.end(),
        [](const DecodedDetection &left, const DecodedDetection &right) {
            return left.score > right.score;
        }
    );

    std::vector<DecodedDetection> selected;
    selected.reserve(std::min<int>(max_detections, static_cast<int>(proposals.size())));
    for (const DecodedDetection &proposal : proposals) {
        const bool suppressed = std::any_of(
            selected.begin(),
            selected.end(),
            [&proposal, nms_threshold](const DecodedDetection &kept) {
                return intersection_over_union(proposal, kept) > nms_threshold;
            }
        );
        if (suppressed) {
            continue;
        }
        selected.push_back(proposal);
        if (static_cast<int>(selected.size()) >= max_detections) {
            break;
        }
    }
    return selected;
}

}  // namespace twopoint::pose
