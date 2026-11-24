// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
// Member A: local inference benchmark + iOS process resource sampling.

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

class LocalInferenceOutput {
  const LocalInferenceOutput({
    required this.requestId,
    required this.imageId,
    required this.detections,
    required this.annotatedImage,
    required this.preprocessMs,
    required this.inferenceMs,
    required this.postprocessMs,
    required this.dartPostprocessMs,
    required this.endToEndMs,
    required this.cpuPercent,
    required this.memoryBeforeMb,
    required this.memoryAfterMb,
    required this.inputBytes,
    required this.isWarmup,
  });

  final String requestId;
  final String imageId;
  final List<Map<String, dynamic>> detections;
  final Uint8List? annotatedImage;
  final double? preprocessMs;
  final double inferenceMs;
  final double? postprocessMs;
  final double dartPostprocessMs;
  final double endToEndMs;
  final double? cpuPercent;
  final double? memoryBeforeMb;
  final double? memoryAfterMb;
  final int inputBytes;
  final bool isWarmup;

  double get fps => endToEndMs > 0 ? 1000 / endToEndMs : 0;
}

class LocalExportFiles {
  const LocalExportFiles({
    required this.resultsCsv,
    required this.resourcesCsv,
    required this.detectionsJson,
    required this.summaryJson,
  });

  final File resultsCsv;
  final File resourcesCsv;
  final File detectionsJson;
  final File summaryJson;
}

class LocalInferenceService {
  LocalInferenceService({
    String modelPath = 'yolo26n',
    required this.deviceId,
    required this.network,
  }) : _yolo = YOLO(modelPath: modelPath);

  static const MethodChannel _resourceChannel =
      MethodChannel('ultralytics_yolo_example/device_resources');

  final String deviceId;
  String network;
  final YOLO _yolo;

  bool _ready = false;
  bool get isReady => _ready;
  int _counter = 0;

  final List<Map<String, dynamic>> _rows = [];
  final List<Map<String, dynamic>> _resources = [];
  final List<Map<String, dynamic>> _detectionDocuments = [];

  int get recordedRunCount =>
      _rows.where((row) => row['is_warmup'] != true).length;
  int get warmupRunCount =>
      _rows.where((row) => row['is_warmup'] == true).length;

  Future<void> initialize() async {
    if (_ready) return;
    await _yolo.loadModel();
    _ready = true;
  }

  Future<LocalInferenceOutput> inferLocal({
    required String imageId,
    required Uint8List imageBytes,
    bool isWarmup = false,
  }) async {
    if (!_ready) throw StateError('YOLO model is not ready.');
    if (imageBytes.isEmpty) throw ArgumentError('Image bytes are empty.');

    final requestId = 'local_${(++_counter).toString().padLeft(5, '0')}';
    final timestamp = DateTime.now().toUtc();
    final before = await _readResourceSnapshot();
    _recordResource(requestId, imageId, 'before_inference', before);

    final endToEndWatch = Stopwatch()..start();
    final predictWatch = Stopwatch()..start();
    late final Map<String, dynamic> result;

    try {
      result = await _yolo.predict(imageBytes);
    } catch (error) {
      predictWatch.stop();
      endToEndWatch.stop();
      final afterFailure = await _readResourceSnapshot();
      _recordResource(
        requestId,
        imageId,
        'after_failed_inference',
        afterFailure,
      );
      _rows.add(_buildRow(
        requestId: requestId,
        imageId: imageId,
        timestamp: timestamp,
        preprocessMs: null,
        inferenceMs: predictWatch.elapsedMicroseconds / 1000.0,
        postprocessMs: null,
        dartPostprocessMs: 0,
        endToEndMs: endToEndWatch.elapsedMicroseconds / 1000.0,
        inputBytes: imageBytes.length,
        success: false,
        isWarmup: isWarmup,
        cpuPercent: _number(afterFailure?['cpu_percent']),
        memoryBeforeMb: _number(before?['memory_mb']),
        memoryAfterMb: _number(afterFailure?['memory_mb']),
        error: error.toString(),
      ));
      rethrow;
    }
    predictWatch.stop();

    final dartPostWatch = Stopwatch()..start();
    final detections = (result['detections'] as List?)
            ?.whereType<Map>()
            .map((item) => _normalizeDetection(
                  Map<String, dynamic>.from(item),
                ))
            .toList(growable: false) ??
        const <Map<String, dynamic>>[];
    dartPostWatch.stop();
    endToEndWatch.stop();

    // The iOS plugin's YOLOInstanceManager returns these native timings in ms.
    // Fall back to the wall-clock predict duration only if an older plugin
    // build does not include inferenceMs.
    final nativePreprocessMs = _number(result['preMs']);
    final nativeInferenceMs =
        _number(result['inferenceMs']) ??
        predictWatch.elapsedMicroseconds / 1000.0;
    final nativePostprocessMs = _number(result['postMs']);
    final nativePipelineMs = _number(result['speed']);

    final after = await _readResourceSnapshot();
    _recordResource(requestId, imageId, 'after_inference', after);

    final dartPostprocessMs = dartPostWatch.elapsedMicroseconds / 1000.0;
    final endToEndMs = endToEndWatch.elapsedMicroseconds / 1000.0;
    final cpuPercent = _number(after?['cpu_percent']);
    final memoryBefore = _number(before?['memory_mb']);
    final memoryAfter = _number(after?['memory_mb']);

    _rows.add(_buildRow(
      requestId: requestId,
      imageId: imageId,
      timestamp: timestamp,
      preprocessMs: nativePreprocessMs,
      inferenceMs: nativeInferenceMs,
      postprocessMs: nativePostprocessMs,
      dartPostprocessMs: dartPostprocessMs,
      endToEndMs: endToEndMs,
      nativePipelineMs: nativePipelineMs,
      inputBytes: imageBytes.length,
      success: true,
      isWarmup: isWarmup,
      cpuPercent: cpuPercent,
      memoryBeforeMb: memoryBefore,
      memoryAfterMb: memoryAfter,
    ));

    _detectionDocuments.add({
      'request_id': requestId,
      'image_id': imageId,
      'mode': 'local',
      'device_id': deviceId,
      'model': 'yolo26n',
      'timestamp_utc': timestamp.toIso8601String(),
      'is_warmup': isWarmup,
      'detections': detections,
    });

    return LocalInferenceOutput(
      requestId: requestId,
      imageId: imageId,
      detections: detections,
      annotatedImage: result['annotatedImage'] as Uint8List?,
      preprocessMs: nativePreprocessMs,
      inferenceMs: nativeInferenceMs,
      postprocessMs: nativePostprocessMs,
      dartPostprocessMs: dartPostprocessMs,
      endToEndMs: endToEndMs,
      cpuPercent: cpuPercent,
      memoryBeforeMb: memoryBefore,
      memoryAfterMb: memoryAfter,
      inputBytes: imageBytes.length,
      isWarmup: isWarmup,
    );
  }

  Map<String, dynamic> _buildRow({
    required String requestId,
    required String imageId,
    required DateTime timestamp,
    required double? preprocessMs,
    required double inferenceMs,
    required double? postprocessMs,
    required double dartPostprocessMs,
    required double endToEndMs,
    double? nativePipelineMs,
    required int inputBytes,
    required bool success,
    required bool isWarmup,
    required double? cpuPercent,
    required double? memoryBeforeMb,
    required double? memoryAfterMb,
    String? error,
  }) {
    return {
      // Unified A/B/C/D handoff schema.
      'request_id': requestId,
      'image_id': imageId,
      'mode': 'local',
      'device_id': deviceId,
      'network': network,
      'latency_ms': _round(endToEndMs),
      'end_to_end_ms': _round(endToEndMs),
      'inference_ms': _round(inferenceMs),
      'preprocess_ms': preprocessMs == null ? null : _round(preprocessMs),
      'postprocess_ms': postprocessMs == null ? null : _round(postprocessMs),
      'request_bytes': 0,
      'success': success,
      // Additional A-side instrumentation.
      'timestamp_utc': timestamp.toIso8601String(),
      'input_bytes': inputBytes,
      'fps': endToEndMs > 0 ? _round(1000 / endToEndMs) : null,
      'inference_fps': inferenceMs > 0 ? _round(1000 / inferenceMs) : null,
      'native_pipeline_ms':
          nativePipelineMs == null ? null : _round(nativePipelineMs),
      'dart_postprocess_ms': _round(dartPostprocessMs),
      'cpu_percent': cpuPercent == null ? null : _round(cpuPercent),
      'memory_before_mb':
          memoryBeforeMb == null ? null : _round(memoryBeforeMb),
      'memory_after_mb':
          memoryAfterMb == null ? null : _round(memoryAfterMb),
      'is_warmup': isWarmup,
      'error': error,
    };
  }

  Future<Map<String, dynamic>?> _readResourceSnapshot() async {
    if (!Platform.isIOS) return null;
    try {
      return await _resourceChannel.invokeMapMethod<String, dynamic>(
        'snapshot',
      );
    } on MissingPluginException {
      return null;
    } on PlatformException {
      return null;
    }
  }

  void _recordResource(
    String requestId,
    String imageId,
    String phase,
    Map<String, dynamic>? snapshot,
  ) {
    _resources.add({
      'request_id': requestId,
      'image_id': imageId,
      'phase': phase,
      'timestamp_utc': DateTime.now().toUtc().toIso8601String(),
      'cpu_percent':
          phase.startsWith('before_') ? null : snapshot?['cpu_percent'],
      'memory_mb': snapshot?['memory_mb'],
      'physical_memory_mb': snapshot?['physical_memory_mb'],
      'available': snapshot?['available'] ?? false,
    });
  }

  Map<String, dynamic> buildSummary() {
    final measured = _rows
        .where((row) =>
            row['is_warmup'] != true && row['success'] == true)
        .toList();
    final latency = measured
        .map((row) => (row['end_to_end_ms'] as num).toDouble())
        .toList()
      ..sort();
    final inference = measured
        .map((row) => (row['inference_ms'] as num).toDouble())
        .toList();
    final preprocess = measured
        .map((row) => row['preprocess_ms'])
        .whereType<num>()
        .map((value) => value.toDouble())
        .toList();
    final postprocess = measured
        .map((row) => row['postprocess_ms'])
        .whereType<num>()
        .map((value) => value.toDouble())
        .toList();
    final cpu = measured
        .map((row) => row['cpu_percent'])
        .whereType<num>()
        .map((value) => value.toDouble())
        .toList();
    final memory = measured
        .map((row) => row['memory_after_mb'])
        .whereType<num>()
        .map((value) => value.toDouble())
        .toList();

    double? mean(List<double> values) => values.isEmpty
        ? null
        : values.reduce((a, b) => a + b) / values.length;
    double? percentile(List<double> sorted, double p) =>
        sorted.isEmpty ? null : sorted[((sorted.length - 1) * p).ceil()];

    final meanLatency = mean(latency);
    return {
      'mode': 'local',
      'device_id': deviceId,
      'network': network,
      'model': 'yolo26n',
      'measured_runs': measured.length,
      'warmup_runs': warmupRunCount,
      'failed_runs': _rows.where((row) => row['success'] != true).length,
      'mean_end_to_end_ms': _roundNullable(meanLatency),
      'median_end_to_end_ms': _roundNullable(
        latency.isEmpty
            ? null
            : latency.length.isOdd
                ? latency[latency.length ~/ 2]
                : (latency[latency.length ~/ 2 - 1] +
                        latency[latency.length ~/ 2]) /
                    2,
      ),
      'p95_end_to_end_ms': _roundNullable(percentile(latency, 0.95)),
      'mean_preprocess_ms': _roundNullable(mean(preprocess)),
      'mean_inference_ms': _roundNullable(mean(inference)),
      'mean_postprocess_ms': _roundNullable(mean(postprocess)),
      'mean_fps': _roundNullable(
        mean(latency.where((value) => value > 0).map((value) => 1000 / value).toList()),
      ),
      'throughput_images_per_second':
          meanLatency == null || meanLatency <= 0
              ? null
              : _round(1000 / meanLatency),
      'mean_cpu_percent': _roundNullable(mean(cpu)),
      'mean_memory_after_mb': _roundNullable(mean(memory)),
      'cpu_measurement_available': cpu.isNotEmpty,
      'stage_timing_source': 'Native YOLO result fields: preMs, inferenceMs, postMs.',
      'cpu_measurement_scope':
          'App process CPU normalized by active processor count; not whole-device CPU.',
      'memory_measurement_scope': 'App process resident memory (RSS).',
      'recorded_at_utc': DateTime.now().toUtc().toIso8601String(),
    };
  }

  Future<LocalExportFiles> exportResults() async {
    final directory = await getApplicationDocumentsDirectory();
    final stamp = DateTime.now()
        .toUtc()
        .toIso8601String()
        .replaceAll(':', '-')
        .replaceAll('.', '-');
    final resultsCsv = File('${directory.path}/local_results_$stamp.csv');
    final resourcesCsv = File('${directory.path}/mobile_resources_$stamp.csv');
    final detectionsJson = File('${directory.path}/local_detections_$stamp.json');
    final summaryJson = File('${directory.path}/local_summary_$stamp.json');

    const resultColumns = [
      'request_id',
      'image_id',
      'mode',
      'device_id',
      'network',
      'latency_ms',
      'end_to_end_ms',
      'inference_ms',
      'preprocess_ms',
      'postprocess_ms',
      'request_bytes',
      'success',
      'timestamp_utc',
      'input_bytes',
      'fps',
      'inference_fps',
      'native_pipeline_ms',
      'dart_postprocess_ms',
      'cpu_percent',
      'memory_before_mb',
      'memory_after_mb',
      'is_warmup',
      'error',
    ];
    const resourceColumns = [
      'request_id',
      'image_id',
      'phase',
      'timestamp_utc',
      'cpu_percent',
      'memory_mb',
      'physical_memory_mb',
      'available',
    ];

    await resultsCsv.writeAsString(
      _toCsv(resultColumns, _rows),
      flush: true,
    );
    await resourcesCsv.writeAsString(
      _toCsv(resourceColumns, _resources),
      flush: true,
    );
    await detectionsJson.writeAsString(
      const JsonEncoder.withIndent('  ').convert(_detectionDocuments),
      flush: true,
    );
    await summaryJson.writeAsString(
      const JsonEncoder.withIndent('  ').convert(buildSummary()),
      flush: true,
    );

    return LocalExportFiles(
      resultsCsv: resultsCsv,
      resourcesCsv: resourcesCsv,
      detectionsJson: detectionsJson,
      summaryJson: summaryJson,
    );
  }

  static String _toCsv(
    List<String> columns,
    List<Map<String, dynamic>> rows,
  ) {
    final lines = <String>[
      columns.join(','),
      ...rows.map(
        (row) => columns.map((column) => _csv(row[column])).join(','),
      ),
    ];
    return '${lines.join('\n')}\n';
  }

  static String _csv(dynamic value) {
    if (value == null) return '';
    final escaped = value.toString().replaceAll('"', '""');
    return (escaped.contains(',') ||
            escaped.contains('"') ||
            escaped.contains('\n') ||
            escaped.contains('\r'))
        ? '"$escaped"'
        : escaped;
  }

  static double? _number(dynamic value) =>
      value is num ? value.toDouble() : null;
  static double _round(double value) =>
      double.parse(value.toStringAsFixed(3));
  static double? _roundNullable(double? value) =>
      value == null ? null : _round(value);

  static Map<String, dynamic> _normalizeDetection(
    Map<String, dynamic> raw,
  ) {
    final output = Map<String, dynamic>.from(raw);
    dynamic pick(List<String> keys) {
      for (final key in keys) {
        if (raw[key] != null) return raw[key];
      }
      return null;
    }

    final classId = pick([
      'class_id',
      'classId',
      'classIndex',
      'class_index',
      'index',
    ]);
    final confidence = pick(['confidence', 'conf', 'score']);
    final className = pick(['class_name', 'className', 'name', 'label']);
    final box = pick(['bbox', 'boundingBox', 'bounding_box', 'box']);

    if (classId != null) output['class_id'] = classId;
    if (confidence != null) output['confidence'] = confidence;
    if (className != null) output['class_name'] = className;
    if (box is List) output['bbox'] = box;
    if (box is Map) {
      final x1 = box['x1'] ?? box['left'];
      final y1 = box['y1'] ?? box['top'];
      final x2 = box['x2'] ?? box['right'];
      final y2 = box['y2'] ?? box['bottom'];
      if (x1 != null && y1 != null && x2 != null && y2 != null) {
        output['bbox'] = [x1, y1, x2, y2];
      }
    }
    return output;
  }

  Future<void> dispose() async {
    _ready = false;
    _yolo.dispose();
  }
}
