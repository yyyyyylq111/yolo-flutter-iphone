// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
// Local inference + benchmark record/export service.

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:path_provider/path_provider.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

class LocalInferenceOutput {
  const LocalInferenceOutput({
    required this.imageId,
    required this.detections,
    required this.annotatedImage,
    required this.latencyMs,
  });

  final String imageId;
  final List<Map<String, dynamic>> detections;
  final Uint8List? annotatedImage;
  final double latencyMs;
}

class LocalExportFiles {
  const LocalExportFiles({required this.csv, required this.json});

  final File csv;
  final File json;
}

class LocalInferenceService {
  LocalInferenceService({
    String modelPath = 'yolo26n',
    this.deviceId = 'iphone15',
    this.network = 'unknown',
  }) : _yolo = YOLO(modelPath: modelPath);

  final String deviceId;
  final String network;
  final YOLO _yolo;

  bool _ready = false;
  bool get isReady => _ready;

  int _counter = 0;
  final List<Map<String, dynamic>> _rows = [];
  final List<Map<String, dynamic>> _documents = [];

  int get recordedRunCount => _documents.length;

  int get successfulRunCount =>
      _documents.where((item) => item['success'] == true).length;

  int get failedRunCount =>
      _documents.where((item) => item['success'] != true).length;

  int get totalDetectionCount => _documents.fold<int>(
        0,
        (total, item) =>
            total + ((item['detection_count'] as num?)?.toInt() ?? 0),
      );

  Future<void> initialize() async {
    if (_ready) return;
    await _yolo.loadModel();
    _ready = true;
  }

  Future<LocalInferenceOutput> inferLocal({
    required String imageId,
    required Uint8List imageBytes,
  }) async {
    if (!_ready) throw StateError('YOLO model is not ready.');

    _counter++;
    final requestId = 'local_${_counter.toString().padLeft(4, '0')}';
    final startedAt = DateTime.now().toUtc();
    final watch = Stopwatch()..start();

    try {
      final result = await _yolo.predict(imageBytes);
      watch.stop();
      final elapsed = watch.elapsedMicroseconds / 1000.0;

      final detections = (result['detections'] as List?)
              ?.whereType<Map>()
              .map((item) => _normalizeDetection(
                    Map<String, dynamic>.from(item),
                  ))
              .toList(growable: false) ??
          const <Map<String, dynamic>>[];

      final annotated = result['annotatedImage'] as Uint8List?;
      final completedAt = DateTime.now().toUtc();

      _rows.add(_runRow(
        requestId: requestId,
        imageId: imageId,
        startedAt: startedAt,
        completedAt: completedAt,
        elapsedMs: elapsed,
        requestBytes: imageBytes.length,
        success: true,
        detectionCount: detections.length,
      ));

      _documents.add({
        'request_id': requestId,
        'image_id': imageId,
        'mode': 'local',
        'device_id': deviceId,
        'network': network,
        'started_at': startedAt.toIso8601String(),
        'completed_at': completedAt.toIso8601String(),
        'latency_ms': _rounded(elapsed),
        'inference_ms': _rounded(elapsed),
        'request_bytes': imageBytes.length,
        'success': true,
        'detection_count': detections.length,
        'detections': detections,
      });

      return LocalInferenceOutput(
        imageId: imageId,
        detections: detections,
        annotatedImage: annotated,
        latencyMs: elapsed,
      );
    } catch (error) {
      watch.stop();
      final elapsed = watch.elapsedMicroseconds / 1000.0;
      final completedAt = DateTime.now().toUtc();

      _rows.add(_runRow(
        requestId: requestId,
        imageId: imageId,
        startedAt: startedAt,
        completedAt: completedAt,
        elapsedMs: elapsed,
        requestBytes: imageBytes.length,
        success: false,
        detectionCount: 0,
      ));

      _documents.add({
        'request_id': requestId,
        'image_id': imageId,
        'mode': 'local',
        'device_id': deviceId,
        'network': network,
        'started_at': startedAt.toIso8601String(),
        'completed_at': completedAt.toIso8601String(),
        'latency_ms': _rounded(elapsed),
        'inference_ms': _rounded(elapsed),
        'request_bytes': imageBytes.length,
        'success': false,
        'detection_count': 0,
        'detections': <Map<String, dynamic>>[],
        'error': error.toString(),
      });
      rethrow;
    }
  }

  Map<String, dynamic> _runRow({
    required String requestId,
    required String imageId,
    required DateTime startedAt,
    required DateTime completedAt,
    required double elapsedMs,
    required int requestBytes,
    required bool success,
    required int detectionCount,
  }) {
    return {
      'request_id': requestId,
      'image_id': imageId,
      'mode': 'local',
      'device_id': deviceId,
      'network': network,
      'started_at': startedAt.toIso8601String(),
      'completed_at': completedAt.toIso8601String(),
      'latency_ms': _rounded(elapsedMs),
      // predict() is one plugin operation; native sub-stages are not exposed.
      'inference_ms': _rounded(elapsedMs),
      'request_bytes': requestBytes,
      'success': success,
      'detection_count': detectionCount,
    };
  }

  /// Exports one CSV row per detected object and a JSON document containing
  /// every inference run and its complete normalized detection maps.
  Future<LocalExportFiles> exportResults() async {
    final dir = await getApplicationDocumentsDirectory();
    await dir.create(recursive: true);

    final stamp = DateTime.now()
        .toUtc()
        .toIso8601String()
        .replaceAll(':', '-')
        .replaceAll('.', '-');
    final csv = File('${dir.path}/yolo_results_$stamp.csv');
    final jsonFile = File('${dir.path}/yolo_detections_$stamp.json');

    const columns = <String>[
      'request_id',
      'image_id',
      'mode',
      'device_id',
      'network',
      'started_at',
      'completed_at',
      'latency_ms',
      'inference_ms',
      'request_bytes',
      'success',
      'detection_count',
      'detection_index',
      'class_name',
      'class_id',
      'confidence',
      'bbox_x1',
      'bbox_y1',
      'bbox_x2',
      'bbox_y2',
      'bbox_x1_normalized',
      'bbox_y1_normalized',
      'bbox_x2_normalized',
      'bbox_y2_normalized',
      'bbox_json',
      'detection_json',
    ];

    final csvLines = <String>[columns.join(',')];

    for (final document in _documents) {
      final detections = (document['detections'] as List?)
              ?.whereType<Map>()
              .toList(growable: false) ??
          const <Map>[];

      if (detections.isEmpty) {
        final row = <String, dynamic>{
          ...document,
          'detection_index': '',
          'class_name': '',
          'class_id': '',
          'confidence': '',
          'bbox_x1': '',
          'bbox_y1': '',
          'bbox_x2': '',
          'bbox_y2': '',
          'bbox_x1_normalized': '',
          'bbox_y1_normalized': '',
          'bbox_x2_normalized': '',
          'bbox_y2_normalized': '',
          'bbox_json': '',
          'detection_json': '',
        };
        csvLines.add(columns.map((key) => _csv(row[key])).join(','));
        continue;
      }

      for (var index = 0; index < detections.length; index++) {
        final detection = Map<String, dynamic>.from(detections[index]);
        final box = _bboxValues(detection['bbox']);
        final normalizedBox = _bboxValues(detection['normalizedBox']);
        final row = <String, dynamic>{
          ...document,
          'detection_index': index + 1,
          'class_name': _className(detection),
          'class_id': detection['class_id'] ?? '',
          'confidence': detection['confidence'] ?? '',
          'bbox_x1': box[0],
          'bbox_y1': box[1],
          'bbox_x2': box[2],
          'bbox_y2': box[3],
          'bbox_x1_normalized': normalizedBox[0],
          'bbox_y1_normalized': normalizedBox[1],
          'bbox_x2_normalized': normalizedBox[2],
          'bbox_y2_normalized': normalizedBox[3],
          'bbox_json': detection['bbox'] == null
              ? ''
              : jsonEncode(detection['bbox']),
          'detection_json': jsonEncode(detection),
        };
        csvLines.add(columns.map((key) => _csv(row[key])).join(','));
      }
    }

    // UTF-8 BOM improves Chinese text compatibility when opening CSV in Excel.
    await csv.writeAsString(
      '\uFEFF${csvLines.join('\n')}\n',
      encoding: utf8,
      flush: true,
    );

    final exportDocument = {
      'schema_version': 1,
      'exported_at': DateTime.now().toUtc().toIso8601String(),
      'device_id': deviceId,
      'network': network,
      'run_count': _documents.length,
      'successful_run_count': successfulRunCount,
      'failed_run_count': failedRunCount,
      'total_detection_count': totalDetectionCount,
      'runs': _documents,
    };

    await jsonFile.writeAsString(
      const JsonEncoder.withIndent('  ').convert(exportDocument),
      encoding: utf8,
      flush: true,
    );

    return LocalExportFiles(csv: csv, json: jsonFile);
  }

  static Map<String, dynamic> _normalizeDetection(
    Map<String, dynamic> raw,
  ) {
    final out = Map<String, dynamic>.from(raw);

    dynamic pick(List<String> keys) {
      for (final key in keys) {
        if (raw[key] != null) return raw[key];
      }
      return null;
    }

    final id = pick([
      'class_id', 'classId', 'classIndex', 'class_index', 'index',
    ]);
    final name = pick([
      'class_name', 'className', 'label', 'name', 'class',
    ]);
    final confidence = pick(['confidence', 'conf', 'score']);
    final rawBox = pick(['bbox', 'boundingBox', 'bounding_box', 'box']);

    if (id != null) out['class_id'] = id;
    if (name != null) out['class_name'] = name;
    if (confidence != null) out['confidence'] = confidence;

    final normalizedBox = _normalizeBox(rawBox);
    if (normalizedBox != null) out['bbox'] = normalizedBox;

    final rawNormalizedBox = _normalizeBox(raw['normalizedBox']);
    if (rawNormalizedBox != null) {
      out['normalizedBox'] = rawNormalizedBox;
    }
    return out;
  }

  static List<dynamic>? _normalizeBox(dynamic box) {
    if (box is List) return box;
    if (box is! Map) return null;

    final x1 = box['x1'] ?? box['left'] ?? box['x_min'];
    final y1 = box['y1'] ?? box['top'] ?? box['y_min'];
    final x2 = box['x2'] ?? box['right'] ?? box['x_max'];
    final y2 = box['y2'] ?? box['bottom'] ?? box['y_max'];
    if (x1 != null && y1 != null && x2 != null && y2 != null) {
      return [x1, y1, x2, y2];
    }

    final x = _asNum(box['x']);
    final y = _asNum(box['y']);
    final width = _asNum(box['width'] ?? box['w']);
    final height = _asNum(box['height'] ?? box['h']);
    if (x != null && y != null && width != null && height != null) {
      return [x, y, x + width, y + height];
    }
    return null;
  }

  static num? _asNum(dynamic value) {
    if (value is num) return value;
    if (value is String) return num.tryParse(value);
    return null;
  }

  static String _className(Map<String, dynamic> detection) {
    final value = detection['class_name'] ??
        detection['label'] ??
        detection['name'] ??
        detection['class'];
    return value?.toString() ?? '';
  }

  static List<dynamic> _bboxValues(dynamic bbox) {
    if (bbox is List && bbox.length >= 4) {
      return [bbox[0], bbox[1], bbox[2], bbox[3]];
    }
    return ['', '', '', ''];
  }

  static double _rounded(double value) =>
      double.parse(value.toStringAsFixed(3));

  static String _csv(dynamic value) {
    final string = (value ?? '').toString().replaceAll('"', '""');
    if (string.contains(',') ||
        string.contains('"') ||
        string.contains('\n') ||
        string.contains('\r')) {
      return '"$string"';
    }
    return string;
  }

  Future<void> dispose() async {
    _ready = false;
    _yolo.dispose();
  }
}
