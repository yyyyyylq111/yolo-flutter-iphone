// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
// Local inference + benchmark record/export service for member A.

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
  int get recordedRunCount => _rows.length;

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
    final watch = Stopwatch()..start();

    try {
      final result = await _yolo.predict(imageBytes);
      watch.stop();
      final elapsed = watch.elapsedMicroseconds / 1000.0;
      final detections = (result['detections'] as List?)
              ?.whereType<Map>()
              .map((e) => _normalizeDetection(Map<String, dynamic>.from(e)))
              .toList(growable: false) ??
          const <Map<String, dynamic>>[];
      final annotated = result['annotatedImage'] as Uint8List?;

      _rows.add(_row(requestId, imageId, elapsed, true));
      _documents.add({
        'request_id': requestId,
        'image_id': imageId,
        'mode': 'local',
        'detections': detections,
      });
      return LocalInferenceOutput(
        imageId: imageId,
        detections: detections,
        annotatedImage: annotated,
        latencyMs: elapsed,
      );
    } catch (_) {
      watch.stop();
      _rows.add(_row(requestId, imageId, watch.elapsedMicroseconds / 1000.0, false));
      rethrow;
    }
  }

  Map<String, dynamic> _row(String requestId, String imageId, double ms, bool ok) => {
        'request_id': requestId,
        'image_id': imageId,
        'mode': 'local',
        'device_id': deviceId,
        'network': network,
        'latency_ms': double.parse(ms.toStringAsFixed(3)),
        // This Flutter API exposes predict() as one operation. This is the
        // measured predict() duration, not a separately instrumented native stage.
        'inference_ms': double.parse(ms.toStringAsFixed(3)),
        'request_bytes': 0,
        'success': ok,
      };

  Future<LocalExportFiles> exportResults() async {
    final dir = await getApplicationDocumentsDirectory();
    final stamp = DateTime.now().toIso8601String().replaceAll(':', '-');
    final csv = File('${dir.path}/local_results_$stamp.csv');
    final json = File('${dir.path}/local_detections_$stamp.json');
    const columns = [
      'request_id', 'image_id', 'mode', 'device_id', 'network',
      'latency_ms', 'inference_ms', 'request_bytes', 'success',
    ];
    final lines = <String>[
      columns.join(','),
      ..._rows.map((r) => columns.map((k) => _csv(r[k])).join(',')),
    ];
    await csv.writeAsString('${lines.join('\n')}\n', flush: true);
    await json.writeAsString(
      const JsonEncoder.withIndent('  ').convert(_documents),
      flush: true,
    );
    return LocalExportFiles(csv: csv, json: json);
  }

  static Map<String, dynamic> _normalizeDetection(Map<String, dynamic> raw) {
    final out = Map<String, dynamic>.from(raw); // retain original plugin fields
    dynamic pick(List<String> keys) {
      for (final key in keys) {
        if (raw[key] != null) return raw[key];
      }
      return null;
    }

    final id = pick(['class_id', 'classId', 'classIndex', 'class_index', 'index']);
    final confidence = pick(['confidence', 'conf', 'score']);
    final box = pick(['bbox', 'boundingBox', 'bounding_box', 'box']);
    if (id != null) out['class_id'] = id;
    if (confidence != null) out['confidence'] = confidence;
    if (box is List) out['bbox'] = box;
    if (box is Map) {
      final x1 = box['x1'] ?? box['left'];
      final y1 = box['y1'] ?? box['top'];
      final x2 = box['x2'] ?? box['right'];
      final y2 = box['y2'] ?? box['bottom'];
      if (x1 != null && y1 != null && x2 != null && y2 != null) {
        out['bbox'] = [x1, y1, x2, y2];
      }
    }
    return out;
  }

  static String _csv(dynamic value) {
    final s = (value ?? '').toString().replaceAll('"', '""');
    return (s.contains(',') || s.contains('"') || s.contains('\n')) ? '"$s"' : s;
  }

  Future<void> dispose() async {
    _ready = false;
    _yolo.dispose();
  }
}
