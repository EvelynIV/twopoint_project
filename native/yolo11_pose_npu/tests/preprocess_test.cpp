#include "preprocess.h"

#include <cstdint>
#include <iostream>
#include <vector>

int main() {
    constexpr int source_width = 1280;
    constexpr int source_height = 720;
    constexpr int destination_size = 640;
    constexpr std::uint8_t blue = 10;
    constexpr std::uint8_t green = 20;
    constexpr std::uint8_t red = 30;

    std::vector<std::uint8_t> source(
        static_cast<std::size_t>(source_width) * source_height * 3
    );
    for (std::size_t offset = 0; offset < source.size(); offset += 3) {
        source[offset] = blue;
        source[offset + 1] = green;
        source[offset + 2] = red;
    }
    std::vector<std::uint8_t> destination(
        static_cast<std::size_t>(destination_size) * destination_size * 3,
        0
    );
    twopoint::pose::LetterboxGeometry geometry{};
    if (!twopoint::pose::letterbox_bgr_to_rgb(
            source.data(),
            source.size(),
            source_width,
            source_height,
            static_cast<std::size_t>(source_width) * 3,
            destination.data(),
            destination.size(),
            destination_size,
            destination_size,
            114,
            &geometry
        )) {
        std::cerr << "fused preprocessing failed\n";
        return 1;
    }
    if (geometry.scale != 0.5 ||
        geometry.resized_width != 640 ||
        geometry.resized_height != 360 ||
        geometry.left != 0 ||
        geometry.top != 140) {
        std::cerr << "unexpected letterbox geometry\n";
        return 1;
    }

    const auto pixel = [&destination](int x, int y, int channel) {
        return destination[(static_cast<std::size_t>(y) * destination_size + x) * 3 + channel];
    };
    if (pixel(0, 0, 0) != 114 ||
        pixel(639, 139, 2) != 114 ||
        pixel(0, 140, 0) != red ||
        pixel(0, 140, 1) != green ||
        pixel(0, 140, 2) != blue ||
        pixel(639, 499, 0) != red ||
        pixel(0, 500, 0) != 114) {
        std::cerr << "letterbox pixels do not match expected RGB and padding values\n";
        return 1;
    }

    const std::vector<std::uint8_t> small_source{
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
        10, 11, 12,
    };
    if (!twopoint::pose::letterbox_bgr_to_rgb(
            small_source.data(),
            small_source.size(),
            2,
            2,
            2 * 3,
            destination.data(),
            destination.size(),
            destination_size,
            destination_size,
            114
        )) {
        std::cerr << "generic bilinear preprocessing failed\n";
        return 1;
    }
    if (pixel(0, 0, 0) != 3 ||
        pixel(0, 0, 1) != 2 ||
        pixel(0, 0, 2) != 1 ||
        pixel(639, 0, 0) != 6 ||
        pixel(0, 639, 0) != 9 ||
        pixel(639, 639, 0) != 12) {
        std::cerr << "generic bilinear border replication or BGR conversion is incorrect\n";
        return 1;
    }
    return 0;
}
