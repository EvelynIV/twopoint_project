# A733 NPU pose backend

## Camera capture

The realtime path resolves exactly one stable
`/dev/v4l/by-id/*-video-index0` camera and captures through GStreamer. Camera
dimensions and FPS come from `.env` rather than task JSON:

```dotenv
TWOPOINT_CAMERA_WIDTH=1280
TWOPOINT_CAMERA_HEIGHT=720
TWOPOINT_CAMERA_FPS=30
```

The capture pipeline requests MJPEG explicitly, drops stale compressed and
decoded frames, and publishes only the newest copied BGR frame. Use the same
path without starting the motor or laser through:

```bash
poetry run python tests/test_cam.py --duration 10
```

The third vision backend runs `model-bin/pose/best_pcq_a733.nb` through the
Cubie A7A VIPLite v2.0 runtime. Build its native bridge on the A7A with:

```bash
cmake \
  -S native/yolo11_pose_npu \
  -B build/npu \
  -DAI_SDK_ROOT=/home/radxa/repositories/ai-sdk \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/npu --parallel
cmake --build build/npu --target test
```

The runtime configuration is:

```dotenv
TWOPOINT_VISION_BACKEND=npu_pose
TWOPOINT_NPU_MODEL_PATH=model-bin/pose/best_pcq_a733.nb
TWOPOINT_NPU_LIBRARY_PATH=build/npu/libyolo11_pose_npu.so
TWOPOINT_NPU_BOX_CONFIDENCE_THRESHOLD=0.4
TWOPOINT_NPU_NMS_THRESHOLD=0.45
TWOPOINT_IMG_SIZE=640
```

`TWOPOINT_NPU_BOX_CONFIDENCE_THRESHOLD` filters pose boxes before target
selection. Keypoint 0 is the semantic target center and keypoints 1-4 are the
four target corners. Among accepted boxes, the center keypoint with the highest
confidence is used directly; there is no separately configurable center-point
confidence threshold.

For a video hardware check, the script reads the NPU model settings from `.env`
and writes an annotated MP4:

```bash
poetry run python tests/npu_pose_demo.py path/to/input.mp4 \
  --output outputs/npu_pose_demo.mp4
```
