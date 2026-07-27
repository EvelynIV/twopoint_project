#include "preprocess.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#if defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace twopoint::pose {
namespace {

constexpr int kChannels = 3;

#if defined(__aarch64__) || defined(__ARM_NEON)
uint8x16_t average_2x2(
    uint8x16_t top_first,
    uint8x16_t top_second,
    uint8x16_t bottom_first,
    uint8x16_t bottom_second
) {
    const uint16x8_t first_sum = vaddq_u16(
        vpaddlq_u8(top_first),
        vpaddlq_u8(bottom_first)
    );
    const uint16x8_t second_sum = vaddq_u16(
        vpaddlq_u8(top_second),
        vpaddlq_u8(bottom_second)
    );
    const uint16x8_t rounding = vdupq_n_u16(2);
    return vcombine_u8(
        vshrn_n_u16(vaddq_u16(first_sum, rounding), 2),
        vshrn_n_u16(vaddq_u16(second_sum, rounding), 2)
    );
}
#endif

std::uint8_t bilinear_channel(
    const std::uint8_t *top_left,
    const std::uint8_t *top_right,
    const std::uint8_t *bottom_left,
    const std::uint8_t *bottom_right,
    int channel,
    double x_weight,
    double y_weight
) {
    const double top =
        static_cast<double>(top_left[channel]) +
        (static_cast<double>(top_right[channel]) - top_left[channel]) * x_weight;
    const double bottom =
        static_cast<double>(bottom_left[channel]) +
        (static_cast<double>(bottom_right[channel]) - bottom_left[channel]) * x_weight;
    const double value = top + (bottom - top) * y_weight;
    return static_cast<std::uint8_t>(std::clamp(std::nearbyint(value), 0.0, 255.0));
}

void fill_padding(
    std::uint8_t *destination,
    int destination_width,
    int destination_height,
    const LetterboxGeometry &geometry,
    std::uint8_t padding_value
) {
    const std::size_t row_bytes = static_cast<std::size_t>(destination_width) * kChannels;
    const int content_bottom = geometry.top + geometry.resized_height;
    const int content_right = geometry.left + geometry.resized_width;
    for (int y = 0; y < destination_height; ++y) {
        std::uint8_t *row = destination + static_cast<std::size_t>(y) * row_bytes;
        if (y < geometry.top || y >= content_bottom) {
            std::memset(row, padding_value, row_bytes);
            continue;
        }
        if (geometry.left > 0) {
            std::memset(
                row,
                padding_value,
                static_cast<std::size_t>(geometry.left) * kChannels
            );
        }
        if (content_right < destination_width) {
            std::memset(
                row + static_cast<std::size_t>(content_right) * kChannels,
                padding_value,
                static_cast<std::size_t>(destination_width - content_right) * kChannels
            );
        }
    }
}

void convert_same_size(
    const std::uint8_t *source,
    std::size_t source_stride,
    std::uint8_t *destination,
    int width,
    int height,
    int destination_width,
    const LetterboxGeometry &geometry
) {
    const std::size_t destination_stride =
        static_cast<std::size_t>(destination_width) * kChannels;
    for (int y = 0; y < height; ++y) {
        const std::uint8_t *source_row = source + static_cast<std::size_t>(y) * source_stride;
        std::uint8_t *destination_row =
            destination +
            static_cast<std::size_t>(geometry.top + y) * destination_stride +
            static_cast<std::size_t>(geometry.left) * kChannels;
        for (int x = 0; x < width; ++x) {
            const std::uint8_t *bgr = source_row + static_cast<std::size_t>(x) * kChannels;
            std::uint8_t *rgb = destination_row + static_cast<std::size_t>(x) * kChannels;
            rgb[0] = bgr[2];
            rgb[1] = bgr[1];
            rgb[2] = bgr[0];
        }
    }
}

void convert_exact_half_size(
    const std::uint8_t *source,
    std::size_t source_stride,
    std::uint8_t *destination,
    int resized_width,
    int resized_height,
    int destination_width,
    const LetterboxGeometry &geometry
) {
    const std::size_t destination_stride =
        static_cast<std::size_t>(destination_width) * kChannels;
    for (int y = 0; y < resized_height; ++y) {
        const std::uint8_t *source_top =
            source + static_cast<std::size_t>(y * 2) * source_stride;
        const std::uint8_t *source_bottom = source_top + source_stride;
        std::uint8_t *destination_row =
            destination +
            static_cast<std::size_t>(geometry.top + y) * destination_stride +
            static_cast<std::size_t>(geometry.left) * kChannels;
        int x = 0;
#if defined(__aarch64__) || defined(__ARM_NEON)
        for (; x + 16 <= resized_width; x += 16) {
            const std::size_t source_offset = static_cast<std::size_t>(x * 2) * kChannels;
            const uint8x16x3_t top_first = vld3q_u8(source_top + source_offset);
            const uint8x16x3_t top_second =
                vld3q_u8(source_top + source_offset + 16 * kChannels);
            const uint8x16x3_t bottom_first = vld3q_u8(source_bottom + source_offset);
            const uint8x16x3_t bottom_second =
                vld3q_u8(source_bottom + source_offset + 16 * kChannels);
            uint8x16x3_t rgb{};
            rgb.val[0] = average_2x2(
                top_first.val[2],
                top_second.val[2],
                bottom_first.val[2],
                bottom_second.val[2]
            );
            rgb.val[1] = average_2x2(
                top_first.val[1],
                top_second.val[1],
                bottom_first.val[1],
                bottom_second.val[1]
            );
            rgb.val[2] = average_2x2(
                top_first.val[0],
                top_second.val[0],
                bottom_first.val[0],
                bottom_second.val[0]
            );
            vst3q_u8(destination_row + static_cast<std::size_t>(x) * kChannels, rgb);
        }
#endif
        for (; x < resized_width; ++x) {
            const std::size_t source_offset = static_cast<std::size_t>(x * 2) * kChannels;
            const std::uint8_t *top_left = source_top + source_offset;
            const std::uint8_t *top_right = top_left + kChannels;
            const std::uint8_t *bottom_left = source_bottom + source_offset;
            const std::uint8_t *bottom_right = bottom_left + kChannels;
            std::uint8_t *rgb = destination_row + static_cast<std::size_t>(x) * kChannels;
            for (int rgb_channel = 0; rgb_channel < kChannels; ++rgb_channel) {
                const int bgr_channel = 2 - rgb_channel;
                const unsigned int sum =
                    top_left[bgr_channel] +
                    top_right[bgr_channel] +
                    bottom_left[bgr_channel] +
                    bottom_right[bgr_channel];
                rgb[rgb_channel] = static_cast<std::uint8_t>((sum + 2U) / 4U);
            }
        }
    }
}

void convert_bilinear(
    const std::uint8_t *source,
    std::size_t source_stride,
    int source_width,
    int source_height,
    std::uint8_t *destination,
    int destination_width,
    const LetterboxGeometry &geometry
) {
    const std::size_t destination_stride =
        static_cast<std::size_t>(destination_width) * kChannels;
    const double x_scale =
        static_cast<double>(source_width) / geometry.resized_width;
    const double y_scale =
        static_cast<double>(source_height) / geometry.resized_height;

    for (int y = 0; y < geometry.resized_height; ++y) {
        const double source_y = (static_cast<double>(y) + 0.5) * y_scale - 0.5;
        const int source_y0 = static_cast<int>(std::floor(source_y));
        const int y0 = std::clamp(source_y0, 0, source_height - 1);
        const int y1 = std::clamp(source_y0 + 1, 0, source_height - 1);
        const double y_weight = std::clamp(source_y - source_y0, 0.0, 1.0);
        const std::uint8_t *top_row =
            source + static_cast<std::size_t>(y0) * source_stride;
        const std::uint8_t *bottom_row =
            source + static_cast<std::size_t>(y1) * source_stride;
        std::uint8_t *destination_row =
            destination +
            static_cast<std::size_t>(geometry.top + y) * destination_stride +
            static_cast<std::size_t>(geometry.left) * kChannels;

        for (int x = 0; x < geometry.resized_width; ++x) {
            const double source_x = (static_cast<double>(x) + 0.5) * x_scale - 0.5;
            const int source_x0 = static_cast<int>(std::floor(source_x));
            const int x0 = std::clamp(source_x0, 0, source_width - 1);
            const int x1 = std::clamp(source_x0 + 1, 0, source_width - 1);
            const double x_weight = std::clamp(source_x - source_x0, 0.0, 1.0);
            const std::uint8_t *top_left =
                top_row + static_cast<std::size_t>(x0) * kChannels;
            const std::uint8_t *top_right =
                top_row + static_cast<std::size_t>(x1) * kChannels;
            const std::uint8_t *bottom_left =
                bottom_row + static_cast<std::size_t>(x0) * kChannels;
            const std::uint8_t *bottom_right =
                bottom_row + static_cast<std::size_t>(x1) * kChannels;
            std::uint8_t *rgb = destination_row + static_cast<std::size_t>(x) * kChannels;
            for (int rgb_channel = 0; rgb_channel < kChannels; ++rgb_channel) {
                rgb[rgb_channel] = bilinear_channel(
                    top_left,
                    top_right,
                    bottom_left,
                    bottom_right,
                    2 - rgb_channel,
                    x_weight,
                    y_weight
                );
            }
        }
    }
}

}  // namespace

LetterboxGeometry calculate_letterbox_geometry(
    int source_width,
    int source_height,
    int destination_width,
    int destination_height
) {
    const double scale = std::min(
        static_cast<double>(destination_width) / source_width,
        static_cast<double>(destination_height) / source_height
    );
    const int resized_width = static_cast<int>(std::nearbyint(source_width * scale));
    const int resized_height = static_cast<int>(std::nearbyint(source_height * scale));
    return {
        scale,
        resized_width,
        resized_height,
        (destination_width - resized_width) / 2,
        (destination_height - resized_height) / 2,
    };
}

bool letterbox_bgr_to_rgb(
    const std::uint8_t *source,
    std::size_t source_size,
    int source_width,
    int source_height,
    std::size_t source_stride,
    std::uint8_t *destination,
    std::size_t destination_size,
    int destination_width,
    int destination_height,
    std::uint8_t padding_value,
    LetterboxGeometry *geometry
) {
    if (source == nullptr || destination == nullptr ||
        source_width <= 0 || source_height <= 0 ||
        destination_width <= 0 || destination_height <= 0) {
        return false;
    }
    const std::size_t minimum_source_stride =
        static_cast<std::size_t>(source_width) * kChannels;
    const std::size_t required_source_size =
        static_cast<std::size_t>(source_height - 1) * source_stride + minimum_source_stride;
    const std::size_t required_destination_size =
        static_cast<std::size_t>(destination_width) * destination_height * kChannels;
    if (source_stride < minimum_source_stride ||
        source_size < required_source_size ||
        destination_size < required_destination_size) {
        return false;
    }

    const LetterboxGeometry calculated = calculate_letterbox_geometry(
        source_width,
        source_height,
        destination_width,
        destination_height
    );
    if (calculated.resized_width <= 0 || calculated.resized_height <= 0 ||
        calculated.left < 0 || calculated.top < 0 ||
        calculated.left + calculated.resized_width > destination_width ||
        calculated.top + calculated.resized_height > destination_height) {
        return false;
    }

    fill_padding(
        destination,
        destination_width,
        destination_height,
        calculated,
        padding_value
    );
    if (source_width == calculated.resized_width &&
        source_height == calculated.resized_height) {
        convert_same_size(
            source,
            source_stride,
            destination,
            source_width,
            source_height,
            destination_width,
            calculated
        );
    } else if (
        source_width == calculated.resized_width * 2 &&
        source_height == calculated.resized_height * 2
    ) {
        convert_exact_half_size(
            source,
            source_stride,
            destination,
            calculated.resized_width,
            calculated.resized_height,
            destination_width,
            calculated
        );
    } else {
        convert_bilinear(
            source,
            source_stride,
            source_width,
            source_height,
            destination,
            destination_width,
            calculated
        );
    }
    if (geometry != nullptr) {
        *geometry = calculated;
    }
    return true;
}

}  // namespace twopoint::pose
