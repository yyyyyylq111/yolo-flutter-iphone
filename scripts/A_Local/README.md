# A — Local inference deliverables

## Included implementation
The runnable Flutter implementation is in:
- `example/lib/presentation/screens/single_image_screen.dart` (copy: `A_Local/single_image_screen.dart`)
- `example/lib/services/local_inference_service.dart` (copy: `A_Local/local_service.dart`)
- `example/ios/Runner/AppDelegate.swift` (iOS CPU/RSS sampler)

The screen provides single-image detection and batch benchmark execution.

## Benchmark protocol implemented
1. Select the fixed test image set using **批量基准测试**.
2. The app executes 10 warm-up inferences on the first image. Each is recorded with `is_warmup=true`.
3. The app executes 3 measured inferences for every selected image.
4. Export the four files from the share sheet:
   - `local_results_*.csv`
   - `mobile_resources_*.csv`
   - `local_detections_*.json`
   - `local_summary_*.json`
5. Use only `is_warmup=false` rows for formal statistics.

## Measurement definitions
- `inference_ms`: native YOLO `inferenceMs` model-inference timing.
- `end_to_end_ms` / `latency_ms`: wall-clock `predict()` call plus Dart-side detection normalization.
- `postprocess_ms`: native YOLO `postMs`; Dart-side result normalization is recorded separately as `dart_postprocess_ms`.
- `preprocess_ms`: native YOLO `preMs` timing returned by the iOS plugin.
- `fps`: 1000 / `end_to_end_ms`; `inference_fps` is 1000 / `inference_ms`.
- `cpu_percent`: app-process CPU time during the sample interval, normalized by the iPhone's active processor count, measured through iOS Mach APIs.
- `memory_mb`: app-process resident memory (RSS), sampled before and after inference.
- `request_bytes`: 0 for Local; original image size is recorded as `input_bytes`.

The native resource bridge reports process-level CPU and memory, not whole-device CPU load. CPU can be null if iOS denies or the native sampler is unavailable. Do not replace null with zero.

## Generate the mobile resource chart on a computer
Install Python dependencies:
```bash
python -m pip install matplotlib
```
Run:
```bash
python A_Local/generate_mobile_resource_chart.py path/to/mobile_resources.csv
```
The script writes `mobile_resource_chart.png` beside the CSV. It plots per-run process CPU utilization and resident memory.

## Important experimental controls
Keep the model (`yolo26n`), iPhone 15, confidence, IoU, test images, and app build fixed across all formal runs. Record the actual network label in the UI; Local does not send images over the network. For the formal experiment, use the agreed 60-image dataset and preserve the original image IDs in filenames (for example `img001.jpg`).
