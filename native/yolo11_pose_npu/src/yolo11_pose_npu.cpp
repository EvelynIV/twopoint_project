#include "yolo11_pose_npu.h"

#include "preprocess.h"
#include "postprocess.h"

#include <vip_lite.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>

namespace {

constexpr std::size_t kInputByteCount =
    static_cast<std::size_t>(POSE_INPUT_WIDTH) * POSE_INPUT_HEIGHT * POSE_INPUT_CHANNELS;

thread_local std::string global_last_error;
std::mutex runtime_mutex;
bool runtime_reserved = false;

struct Tensor {
    vip_buffer_create_params_t parameters{};
    vip_buffer buffer = VIP_NULL;
    vip_uint32_t elements = 0;
};

struct PoseContext {
    vip_network network = VIP_NULL;
    Tensor input;
    std::array<Tensor, twopoint::pose::kOutputCount> outputs{};
    std::array<int, twopoint::pose::kOutputCount> semantic_to_actual{};
    float score_threshold = 0.4F;
    float nms_threshold = 0.45F;
    std::string last_error;
    std::thread::id owner_thread;
    bool runtime_initialized = false;
    bool network_prepared = false;
    bool owns_runtime_reservation = false;
    PoseNativeTiming last_timing{};

    ~PoseContext() {
        cleanup();
    }

    bool fail(const std::string &message) {
        last_error = message;
        global_last_error = message;
        return false;
    }

    bool check(vip_status_e status, const char *operation) {
        if (status == VIP_SUCCESS) {
            return true;
        }
        std::ostringstream stream;
        stream << operation << " failed with VIPLite status " << static_cast<int>(status);
        return fail(stream.str());
    }

    bool require_owner_thread() {
        if (std::this_thread::get_id() == owner_thread) {
            return true;
        }
        return fail("VIPLite context must be used and destroyed on the thread that created it");
    }

    void cleanup() {
        if (owner_thread != std::thread::id{} && std::this_thread::get_id() != owner_thread) {
            global_last_error = "VIPLite context destruction attempted from a different thread";
            return;
        }
        if (network != VIP_NULL && network_prepared) {
            vip_finish_network(network);
            network_prepared = false;
        }
        if (input.buffer != VIP_NULL) {
            vip_destroy_buffer(input.buffer);
            input.buffer = VIP_NULL;
        }
        for (Tensor &output : outputs) {
            if (output.buffer != VIP_NULL) {
                vip_destroy_buffer(output.buffer);
                output.buffer = VIP_NULL;
            }
        }
        if (network != VIP_NULL) {
            vip_destroy_network(network);
            network = VIP_NULL;
        }
        if (runtime_initialized) {
            vip_destroy();
            runtime_initialized = false;
        }
        if (owns_runtime_reservation) {
            std::lock_guard<std::mutex> lock(runtime_mutex);
            runtime_reserved = false;
            owns_runtime_reservation = false;
        }
    }
};

vip_status_e query_tensor(
    vip_network network,
    bool input,
    vip_uint32_t index,
    Tensor &tensor
) {
    auto query = [network, input, index](vip_enum property, void *value) {
        return input
            ? vip_query_input(network, index, property, value)
            : vip_query_output(network, index, property, value);
    };

    tensor.parameters.memory_type = VIP_BUFFER_MEMORY_TYPE_DEFAULT;
    vip_status_e status = query(VIP_BUFFER_PROP_DATA_FORMAT, &tensor.parameters.data_format);
    if (status != VIP_SUCCESS) {
        return status;
    }
    status = query(VIP_BUFFER_PROP_NUM_OF_DIMENSION, &tensor.parameters.num_of_dims);
    if (status != VIP_SUCCESS) {
        return status;
    }
    status = query(VIP_BUFFER_PROP_SIZES_OF_DIMENSION, tensor.parameters.sizes);
    if (status != VIP_SUCCESS) {
        return status;
    }
    status = query(VIP_BUFFER_PROP_QUANT_FORMAT, &tensor.parameters.quant_format);
    if (status != VIP_SUCCESS) {
        return status;
    }
    if (tensor.parameters.quant_format == VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT) {
        status = query(
            VIP_BUFFER_PROP_FIXED_POINT_POS,
            &tensor.parameters.quant_data.dfp.fixed_point_pos
        );
    } else if (tensor.parameters.quant_format == VIP_BUFFER_QUANTIZE_TF_ASYMM) {
        status = query(VIP_BUFFER_PROP_TF_SCALE, &tensor.parameters.quant_data.affine.scale);
        if (status == VIP_SUCCESS) {
            status = query(
                VIP_BUFFER_PROP_TF_ZERO_POINT,
                &tensor.parameters.quant_data.affine.zeroPoint
            );
        }
    }
    if (status != VIP_SUCCESS) {
        return status;
    }

    tensor.elements = 1;
    for (vip_uint32_t dimension = 0; dimension < tensor.parameters.num_of_dims; ++dimension) {
        tensor.elements *= tensor.parameters.sizes[dimension];
    }
    return VIP_SUCCESS;
}

bool has_shape(const Tensor &tensor, int width, int height, int channels) {
    return tensor.parameters.num_of_dims == 4 &&
        tensor.parameters.sizes[0] == static_cast<vip_uint32_t>(width) &&
        tensor.parameters.sizes[1] == static_cast<vip_uint32_t>(height) &&
        tensor.parameters.sizes[2] == static_cast<vip_uint32_t>(channels) &&
        tensor.parameters.sizes[3] == 1;
}

bool initialize_context(PoseContext &context, const char *model_path) {
    {
        std::lock_guard<std::mutex> lock(runtime_mutex);
        if (runtime_reserved) {
            return context.fail("only one A733 VIPLite pose context is supported per process");
        }
        runtime_reserved = true;
        context.owns_runtime_reservation = true;
    }

    if (!context.check(vip_init(), "vip_init")) {
        return false;
    }
    context.runtime_initialized = true;

    if (!context.check(
            vip_create_network(model_path, 0, VIP_CREATE_NETWORK_FROM_FILE, &context.network),
            "vip_create_network"
        )) {
        return false;
    }

    vip_uint32_t input_count = 0;
    vip_uint32_t output_count = 0;
    if (!context.check(
            vip_query_network(context.network, VIP_NETWORK_PROP_INPUT_COUNT, &input_count),
            "query input count"
        ) || !context.check(
            vip_query_network(context.network, VIP_NETWORK_PROP_OUTPUT_COUNT, &output_count),
            "query output count"
        )) {
        return false;
    }
    if (input_count != 1 || output_count != twopoint::pose::kOutputCount) {
        std::ostringstream stream;
        stream << "expected 1 input and 9 outputs, got " << input_count << " input(s) and "
               << output_count << " output(s)";
        return context.fail(stream.str());
    }

    if (!context.check(query_tensor(context.network, true, 0, context.input), "query input tensor")) {
        return false;
    }
    if (!has_shape(context.input, 3, POSE_INPUT_WIDTH, POSE_INPUT_HEIGHT) ||
        context.input.parameters.data_format != VIP_BUFFER_FORMAT_UINT8 ||
        context.input.elements != kInputByteCount) {
        std::ostringstream stream;
        stream << "expected UINT8 input shape [3,640,640,1], got ["
               << context.input.parameters.sizes[0] << ','
               << context.input.parameters.sizes[1] << ','
               << context.input.parameters.sizes[2] << ','
               << context.input.parameters.sizes[3] << "] format="
               << context.input.parameters.data_format;
        return context.fail(stream.str());
    }
    if (!context.check(
            vip_create_buffer(
                &context.input.parameters,
                sizeof(context.input.parameters),
                &context.input.buffer
            ),
            "create input buffer"
        )) {
        return false;
    }

    context.semantic_to_actual.fill(-1);
    for (vip_uint32_t index = 0; index < output_count; ++index) {
        Tensor &tensor = context.outputs[index];
        if (!context.check(query_tensor(context.network, false, index, tensor), "query output tensor")) {
            return false;
        }
        if (tensor.parameters.data_format != VIP_BUFFER_FORMAT_FP32) {
            std::ostringstream stream;
            stream << "output " << index << " is not FP32; format=" << tensor.parameters.data_format;
            return context.fail(stream.str());
        }

        int scale_slot = -1;
        if (tensor.parameters.sizes[0] == 80 && tensor.parameters.sizes[1] == 80) {
            scale_slot = 0;
        } else if (tensor.parameters.sizes[0] == 40 && tensor.parameters.sizes[1] == 40) {
            scale_slot = 1;
        } else if (tensor.parameters.sizes[0] == 20 && tensor.parameters.sizes[1] == 20) {
            scale_slot = 2;
        }
        int head_slot = -1;
        if (tensor.parameters.sizes[2] == 64) {
            head_slot = 0;
        } else if (tensor.parameters.sizes[2] == 1) {
            head_slot = 1;
        } else if (tensor.parameters.sizes[2] == POSE_KEYPOINT_COUNT * 3) {
            head_slot = 2;
        }
        if (tensor.parameters.num_of_dims != 4 || tensor.parameters.sizes[3] != 1 ||
            scale_slot < 0 || head_slot < 0) {
            std::ostringstream stream;
            stream << "unexpected output " << index << " shape ["
                   << tensor.parameters.sizes[0] << ','
                   << tensor.parameters.sizes[1] << ','
                   << tensor.parameters.sizes[2] << ','
                   << tensor.parameters.sizes[3] << ']';
            return context.fail(stream.str());
        }
        const int semantic_slot = scale_slot * 3 + head_slot;
        if (context.semantic_to_actual[semantic_slot] != -1) {
            return context.fail("duplicate YOLO11 pose output head shape");
        }
        context.semantic_to_actual[semantic_slot] = static_cast<int>(index);

        if (!context.check(
                vip_create_buffer(&tensor.parameters, sizeof(tensor.parameters), &tensor.buffer),
                "create output buffer"
            )) {
            return false;
        }
    }
    if (std::any_of(
            context.semantic_to_actual.begin(),
            context.semantic_to_actual.end(),
            [](int value) { return value < 0; }
        )) {
        return context.fail("model is missing one or more expected YOLO11 pose output heads");
    }

    if (!context.check(vip_prepare_network(context.network), "vip_prepare_network")) {
        return false;
    }
    context.network_prepared = true;
    if (!context.check(vip_set_input(context.network, 0, context.input.buffer), "vip_set_input")) {
        return false;
    }
    for (vip_uint32_t index = 0; index < output_count; ++index) {
        if (!context.check(
                vip_set_output(context.network, index, context.outputs[index].buffer),
                "vip_set_output"
            )) {
            return false;
        }
    }
    return true;
}

using SteadyClock = std::chrono::steady_clock;

std::uint64_t elapsed_ns(
    const SteadyClock::time_point &started,
    const SteadyClock::time_point &finished = SteadyClock::now()
) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(finished - started).count()
    );
}

int run_prepared_input(
    PoseContext &context,
    PoseDetection *detections,
    int max_detections,
    const SteadyClock::time_point &total_started
) {
    const auto input_flush_started = SteadyClock::now();
    if (!context.check(
            vip_flush_buffer(context.input.buffer, VIP_BUFFER_OPER_TYPE_FLUSH),
            "flush input buffer"
        )) {
        return -1;
    }
    context.last_timing.input_flush_ns = elapsed_ns(input_flush_started);

    const auto network_started = SteadyClock::now();
    if (!context.check(vip_run_network(context.network), "vip_run_network")) {
        return -1;
    }
    context.last_timing.network_ns = elapsed_ns(network_started);

    std::array<const float *, twopoint::pose::kOutputCount> semantic_outputs{};
    std::array<void *, twopoint::pose::kOutputCount> mapped_outputs{};
    const auto output_map_started = SteadyClock::now();
    for (std::size_t semantic = 0; semantic < semantic_outputs.size(); ++semantic) {
        const int actual = context.semantic_to_actual[semantic];
        Tensor &tensor = context.outputs[actual];
        if (!context.check(
                vip_flush_buffer(tensor.buffer, VIP_BUFFER_OPER_TYPE_INVALIDATE),
                "invalidate output buffer"
            )) {
            for (std::size_t mapped = 0; mapped < semantic; ++mapped) {
                vip_unmap_buffer(context.outputs[context.semantic_to_actual[mapped]].buffer);
            }
            return -1;
        }
        mapped_outputs[semantic] = vip_map_buffer(tensor.buffer);
        if (mapped_outputs[semantic] == nullptr) {
            context.fail("vip_map_buffer returned null for output");
            for (std::size_t mapped = 0; mapped < semantic; ++mapped) {
                vip_unmap_buffer(context.outputs[context.semantic_to_actual[mapped]].buffer);
            }
            return -1;
        }
        semantic_outputs[semantic] = static_cast<const float *>(mapped_outputs[semantic]);
    }
    context.last_timing.output_map_ns = elapsed_ns(output_map_started);

    const auto decode_started = SteadyClock::now();
    const std::vector<twopoint::pose::DecodedDetection> decoded =
        twopoint::pose::decode_yolo11_pose(
            semantic_outputs,
            context.score_threshold,
            context.nms_threshold,
            max_detections
        );
    context.last_timing.decode_ns = elapsed_ns(decode_started);

    const auto output_unmap_started = SteadyClock::now();
    for (std::size_t semantic = 0; semantic < semantic_outputs.size(); ++semantic) {
        vip_unmap_buffer(context.outputs[context.semantic_to_actual[semantic]].buffer);
    }
    context.last_timing.output_unmap_ns = elapsed_ns(output_unmap_started);

    const auto result_copy_started = SteadyClock::now();
    const int count = std::min<int>(max_detections, static_cast<int>(decoded.size()));
    for (int index = 0; index < count; ++index) {
        const twopoint::pose::DecodedDetection &source = decoded[index];
        PoseDetection &destination = detections[index];
        destination.x1 = source.x1;
        destination.y1 = source.y1;
        destination.x2 = source.x2;
        destination.y2 = source.y2;
        destination.score = source.score;
        std::copy(source.keypoints.begin(), source.keypoints.end(), destination.keypoints);
    }
    context.last_timing.result_copy_ns = elapsed_ns(result_copy_started);
    context.last_timing.total_ns = elapsed_ns(total_started);
    context.last_error.clear();
    return count;
}

}  // namespace

extern "C" void *pose_create(
    const char *model_path,
    float score_threshold,
    float nms_threshold
) {
    global_last_error.clear();
    if (model_path == nullptr || *model_path == '\0') {
        global_last_error = "model path is empty";
        return nullptr;
    }
    if (!std::filesystem::is_regular_file(model_path)) {
        global_last_error = std::string("model file does not exist: ") + model_path;
        return nullptr;
    }
    if (!(score_threshold >= 0.0F && score_threshold <= 1.0F)) {
        global_last_error = "score threshold must be between 0 and 1";
        return nullptr;
    }
    if (!(nms_threshold >= 0.0F && nms_threshold <= 1.0F)) {
        global_last_error = "NMS threshold must be between 0 and 1";
        return nullptr;
    }

    auto context = std::make_unique<PoseContext>();
    context->owner_thread = std::this_thread::get_id();
    context->score_threshold = score_threshold;
    context->nms_threshold = nms_threshold;
    if (!initialize_context(*context, model_path)) {
        global_last_error = context->last_error;
        return nullptr;
    }
    return context.release();
}

extern "C" int pose_infer_rgb640(
    void *opaque_context,
    const uint8_t *rgb_data,
    size_t rgb_size,
    PoseDetection *detections,
    int max_detections
) {
    const auto total_started = SteadyClock::now();
    auto *context = static_cast<PoseContext *>(opaque_context);
    if (context == nullptr) {
        global_last_error = "pose context is null";
        return -1;
    }
    if (!context->require_owner_thread()) {
        return -1;
    }
    context->last_timing = {};
    if (rgb_data == nullptr || rgb_size != kInputByteCount) {
        std::ostringstream stream;
        stream << "expected exactly " << kInputByteCount << " RGB input bytes, got " << rgb_size;
        context->fail(stream.str());
        return -1;
    }
    if (detections == nullptr || max_detections <= 0) {
        context->fail("detections buffer is null or max_detections is not positive");
        return -1;
    }

    const auto preprocess_started = SteadyClock::now();
    void *input_data = vip_map_buffer(context->input.buffer);
    if (input_data == nullptr) {
        context->fail("vip_map_buffer returned null for input");
        return -1;
    }
    std::memcpy(input_data, rgb_data, kInputByteCount);
    vip_unmap_buffer(context->input.buffer);
    context->last_timing.preprocess_ns = elapsed_ns(preprocess_started);
    return run_prepared_input(*context, detections, max_detections, total_started);
}

extern "C" int pose_infer_bgr(
    void *opaque_context,
    const uint8_t *bgr_data,
    size_t bgr_size,
    int width,
    int height,
    size_t row_stride,
    PoseDetection *detections,
    int max_detections
) {
    const auto total_started = SteadyClock::now();
    auto *context = static_cast<PoseContext *>(opaque_context);
    if (context == nullptr) {
        global_last_error = "pose context is null";
        return -1;
    }
    if (!context->require_owner_thread()) {
        return -1;
    }
    context->last_timing = {};
    if (bgr_data == nullptr || width <= 0 || height <= 0) {
        context->fail("BGR input pointer is null or dimensions are not positive");
        return -1;
    }
    if (detections == nullptr || max_detections <= 0) {
        context->fail("detections buffer is null or max_detections is not positive");
        return -1;
    }

    const auto preprocess_started = SteadyClock::now();
    void *input_data = vip_map_buffer(context->input.buffer);
    if (input_data == nullptr) {
        context->fail("vip_map_buffer returned null for input");
        return -1;
    }
    const bool preprocessed = twopoint::pose::letterbox_bgr_to_rgb(
        bgr_data,
        bgr_size,
        width,
        height,
        row_stride,
        static_cast<std::uint8_t *>(input_data),
        kInputByteCount,
        POSE_INPUT_WIDTH,
        POSE_INPUT_HEIGHT,
        114
    );
    vip_unmap_buffer(context->input.buffer);
    context->last_timing.preprocess_ns = elapsed_ns(preprocess_started);
    if (!preprocessed) {
        context->fail("invalid BGR input buffer or stride for fused preprocessing");
        return -1;
    }
    return run_prepared_input(*context, detections, max_detections, total_started);
}

extern "C" int pose_last_timing(void *opaque_context, PoseNativeTiming *timing) {
    auto *context = static_cast<PoseContext *>(opaque_context);
    if (context == nullptr || timing == nullptr || !context->require_owner_thread()) {
        return -1;
    }
    *timing = context->last_timing;
    return 0;
}

extern "C" uint32_t pose_driver_version(void *opaque_context) {
    auto *context = static_cast<PoseContext *>(opaque_context);
    if (context == nullptr || !context->require_owner_thread()) {
        return 0;
    }
    return vip_get_version();
}

extern "C" const char *pose_last_error(void *opaque_context) {
    auto *context = static_cast<PoseContext *>(opaque_context);
    return context == nullptr ? global_last_error.c_str() : context->last_error.c_str();
}

extern "C" void pose_destroy(void *opaque_context) {
    auto *context = static_cast<PoseContext *>(opaque_context);
    if (context == nullptr) {
        return;
    }
    if (!context->require_owner_thread()) {
        global_last_error = context->last_error;
        return;
    }
    delete context;
}
