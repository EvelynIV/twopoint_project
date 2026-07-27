#include "postprocess.h"

#include <array>
#include <cmath>
#include <iostream>
#include <vector>

namespace {

bool close_to(float left, float right, float tolerance = 0.05F) {
    return std::fabs(left - right) <= tolerance;
}

}  // namespace

int main() {
    constexpr std::array<int, 3> grid_sizes{80 * 80, 40 * 40, 20 * 20};
    std::array<std::vector<float>, twopoint::pose::kOutputCount> storage;
    for (std::size_t scale = 0; scale < grid_sizes.size(); ++scale) {
        storage[scale * 3].assign(64 * grid_sizes[scale], -20.0F);
        storage[scale * 3 + 1].assign(grid_sizes[scale], -20.0F);
        storage[scale * 3 + 2].assign(15 * grid_sizes[scale], 0.0F);
    }

    const int grid_width = 80;
    const int x = 10;
    const int y = 12;
    const int grid_index = y * grid_width + x;
    storage[1][grid_index] = 10.0F;
    for (int side = 0; side < 4; ++side) {
        storage[0][(side * 16 + 2) * grid_sizes[0] + grid_index] = 20.0F;
    }
    for (int keypoint = 0; keypoint < twopoint::pose::kKeypointCount; ++keypoint) {
        const int channel = keypoint * 3;
        storage[2][channel * grid_sizes[0] + grid_index] = 0.25F;
        storage[2][(channel + 1) * grid_sizes[0] + grid_index] = 0.5F;
        storage[2][(channel + 2) * grid_sizes[0] + grid_index] = 10.0F;
    }

    std::array<const float *, twopoint::pose::kOutputCount> outputs{};
    for (std::size_t index = 0; index < outputs.size(); ++index) {
        outputs[index] = storage[index].data();
    }
    const auto detections = twopoint::pose::decode_yolo11_pose(outputs, 0.4F, 0.45F, 100);

    if (detections.size() != 1) {
        std::cerr << "expected one detection, got " << detections.size() << '\n';
        return 1;
    }
    const auto &detection = detections.front();
    if (detection.score <= 0.99F ||
        !close_to(detection.x1, 68.0F) ||
        !close_to(detection.y1, 84.0F) ||
        !close_to(detection.x2, 100.0F) ||
        !close_to(detection.y2, 116.0F) ||
        !close_to(detection.keypoints[0], 84.0F) ||
        !close_to(detection.keypoints[1], 104.0F) ||
        detection.keypoints[2] <= 0.99F) {
        std::cerr << "decoded values do not match the synthetic YOLO11 pose head\n";
        return 1;
    }
    return 0;
}
