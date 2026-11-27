
// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'dart:io';
import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import 'package:app_links/app_links.dart';
import 'package:flutter/material.dart';
import 'package:package_info_plus/package_info_plus.dart';
import 'package:path_provider/path_provider.dart';
import 'package:share_plus/share_plus.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';
import 'package:ultralytics_yolo/core/yolo_model_manager.dart';
import 'package:ultralytics_yolo/core/yolo_model_resolver.dart';

/// Real-time YOLO camera inference screen.
class CameraInferenceScreen extends StatefulWidget {
  const CameraInferenceScreen({super.key});

  @override
  State<CameraInferenceScreen> createState() =>
      _CameraInferenceScreenState();
}

class _CameraInferenceScreenState
    extends State<CameraInferenceScreen> {
  String? _versionLabel;
  YOLOInstalledModel? _model;
  StreamSubscription<Uri>? _links;
  StreamSubscription<DownloadProgress>? _progress;
  String? _importId;
  Future<void>? _activeImport;
  double _fraction = 0;

  // Each entry is one native inference-frame event, including its detections
  // and frame-level timing/size metadata.
  final List<Map<String, dynamic>> _inferenceFrames = <Map<String, dynamic>>[];
  static const int _maxRecordedFrames = 20000;

  void _onStreamingData(Map<String, dynamic> data) {
    // Convert through JSON to detach the event map from any plugin-owned objects
    // and ensure the eventual export contains JSON-safe values.
    final detached = Map<String, dynamic>.from(
      jsonDecode(jsonEncode(data)) as Map,
    );
    _inferenceFrames.add(detached);
    if (_inferenceFrames.length > _maxRecordedFrames) {
      _inferenceFrames.removeRange(
        0,
        _inferenceFrames.length - _maxRecordedFrames,
      );
    }
  }

  static String _csv(dynamic value) {
    final text = (value ?? '').toString().replaceAll('"', '""');
    if (text.contains(',') ||
        text.contains('"') ||
        text.contains('\n') ||
        text.contains('\r')) {
      return '"$text"';
    }
    return text;
  }

  static dynamic _firstValue(Map<dynamic, dynamic> map, List<String> keys) {
    for (final key in keys) {
      if (map[key] != null) return map[key];
    }
    return '';
  }

  Future<List<XFile>> _createCaptureExports(
    Uint8List imageBytes,
    List<Map<String, dynamic>> frames,
  ) async {
    final dir = await getTemporaryDirectory();
    final stamp = DateTime.now().toUtc().toIso8601String()
        .replaceAll(':', '-')
        .replaceAll('.', '-');
    final imageFile = File('${dir.path}/yolo_capture_$stamp.jpg');
    final csvFile = File('${dir.path}/yolo_detections_$stamp.csv');
    final jsonFile = File('${dir.path}/yolo_detections_$stamp.json');

    await imageFile.writeAsBytes(imageBytes, flush: true);

    const columns = <String>[
      'frame_number',
      'timestamp',
      'fps',
      'processing_time_ms',
      'image_width',
      'image_height',
      'detection_index',
      'class_name',
      'class_index',
      'confidence',
      'bbox_json',
      'detection_json',
    ];
    final csvLines = <String>[columns.join(',')];
    var totalDetections = 0;

    for (final frame in frames) {
      final rawDetections = frame['detections'];
      final detections = rawDetections is List
          ? rawDetections.whereType<Map>().toList(growable: false)
          : const <Map>[];
      totalDetections += detections.length;

      final frameValues = <String, dynamic>{
        'frame_number': frame['frameNumber'] ?? frame['frame_number'] ?? '',
        'timestamp': frame['timestamp'] ?? '',
        'fps': frame['fps'] ?? '',
        'processing_time_ms':
            frame['processingTimeMs'] ?? frame['processing_time_ms'] ?? '',
        'image_width': frame['imageWidth'] ?? frame['image_width'] ?? '',
        'image_height': frame['imageHeight'] ?? frame['image_height'] ?? '',
      };

      if (detections.isEmpty) {
        final row = <String, dynamic>{
          ...frameValues,
          'detection_index': '',
          'class_name': '',
          'class_index': '',
          'confidence': '',
          'bbox_json': '',
          'detection_json': '',
        };
        csvLines.add(columns.map((key) => _csv(row[key])).join(','));
        continue;
      }

      for (var index = 0; index < detections.length; index++) {
        final detection = Map<String, dynamic>.from(detections[index]);
        final row = <String, dynamic>{
          ...frameValues,
          'detection_index': index + 1,
          'class_name': _firstValue(
            detection,
            ['className', 'class_name', 'label', 'name', 'class'],
          ),
          'class_index': _firstValue(
            detection,
            ['classIndex', 'class_index', 'classId', 'class_id'],
          ),
          'confidence': _firstValue(
            detection,
            ['confidence', 'conf', 'score'],
          ),
          'bbox_json': jsonEncode(
            detection['boundingBox'] ??
                detection['normalizedBox'] ??
                detection['bbox'] ??
                '',
          ),
          'detection_json': jsonEncode(detection),
        };
        csvLines.add(columns.map((key) => _csv(row[key])).join(','));
      }
    }

    await csvFile.writeAsString(
      '${csvLines.join('\n')}\n',
      encoding: utf8,
      flush: true,
    );

    final exportDocument = <String, dynamic>{
      'schema_version': 1,
      'exported_at': DateTime.now().toUtc().toIso8601String(),
      'source': 'YOLOView real-time camera stream',
      'recorded_frame_count': frames.length,
      'total_detection_count': totalDetections,
      'frames': frames,
    };
    await jsonFile.writeAsString(
      const JsonEncoder.withIndent('  ').convert(exportDocument),
      encoding: utf8,
      flush: true,
    );

    return <XFile>[
      XFile(imageFile.path),
      XFile(csvFile.path),
      XFile(jsonFile.path),
    ];
  }

  @override
  void initState() {
    super.initState();
    _loadVersion();

    _links = AppLinks().uriLinkStream.listen(
      (uri) => unawaited(_importModel(uri.toString())),
    );

    _progress = YOLOModelManager.downloadProgress.listen((progress) {
      if (mounted && progress.modelId == _importId) {
        setState(() => _fraction = progress.fraction);
      }
    });
  }

  @override
  void dispose() {
    _links?.cancel();
    _progress?.cancel();
    super.dispose();
  }

  Future<void> _importModel(String link) async {
    if (_activeImport != null) return;

    _activeImport = _resolveAndImportModel(link);

    try {
      await _activeImport;
    } finally {
      _activeImport = null;
    }
  }

  Future<void> _resolveAndImportModel(String link) async {
    try {
      final uri = Uri.parse(link);
      final parts = uri.pathSegments;

      if (uri.scheme != 'https' ||
          uri.host != 'platform.ultralytics.com' ||
          uri.hasPort ||
          uri.userInfo.isNotEmpty ||
          parts.length != 3 ||
          parts[0] != 'deploy' ||
          parts[1] != 'mobile' ||
          !RegExp(r'^[0-9a-f]{24}$').hasMatch(parts[2]) ||
          uri.fragment.length > 128) {
        throw const FormatException(
          'Scan a model QR code from Ultralytics Platform.',
        );
      }

      final client = HttpClient();
      Map<String, dynamic> descriptor;

      try {
        final request = await client.postUrl(
          Uri.https(
            'platform.ultralytics.com',
            '/api/mobile/models/${parts[2]}',
          ),
        );

        request.headers.contentType = ContentType.json;
        request.write(jsonEncode(Uri.splitQueryString(uri.fragment)));

        final response = await request.close();

        if (response.statusCode != HttpStatus.ok) {
          throw const FormatException(
            'Import link expired or unavailable. Create a new QR code in Platform.',
          );
        }

        descriptor =
            jsonDecode(await utf8.decoder.bind(response).join())
                as Map<String, dynamic>;
      } finally {
        client.close(force: true);
      }

      final task = YOLOTaskParsing.tryParse(
        descriptor['task'] as String?,
      );

      final format = descriptor['format'];

      final extension = {
        'litert': 'tflite',
        'coreml': 'mlpackage.zip',
      }[format];

      final download = Uri.parse(
        descriptor['downloadUrl'] as String,
      );

      if (descriptor['version'] != 1 ||
          descriptor['exportId'] != parts[2] ||
          task == null ||
          extension == null ||
          descriptor['filename'] != '${parts[2]}.$extension' ||
          download.scheme != 'https' ||
          (descriptor['size'] as num) <= 0 ||
          (Platform.isAndroid
              ? format != 'litert'
              : format == 'litert')) {
        throw const FormatException(
          'This export is incompatible with this device.',
        );
      }

      final existing =
          (await YOLOModelManager.installedModels())
              .where((model) => model.id == parts[2])
              .firstOrNull;

      if (!mounted) return;

      if (existing != null) {
        setState(() => _model = existing);
        return;
      }

      final name = descriptor['name'] as String;

      final confirmed = await showDialog<bool>(
        context: context,
        builder: (context) => AlertDialog(
          title: Text(name),
          content: Text(
            '${task.label} · '
            '${((descriptor['size'] as num) / (1024 * 1024)).toStringAsFixed(1)} MB\n'
            'Download for offline use?',
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context, false),
              child: const Text('Cancel'),
            ),
            TextButton(
              onPressed: () => Navigator.pop(context, true),
              child: const Text('Download'),
            ),
          ],
        ),
      );

      if (!mounted || confirmed != true) return;

      setState(() {
        _importId = parts[2];
        _fraction = 0;
      });

      YOLOModelManager.clearDownloadCancellation(parts[2]);

      final resolved = await YOLOModelResolver.resolve(
        modelPath: download.toString(),
        task: task,
        filename: descriptor['filename'] as String,
        expectedSize: (descriptor['size'] as num).toInt(),
      );

      final model = YOLOInstalledModel(
        id: parts[2],
        name: name,
        task: task,
        path: resolved.modelPath,
      );

      await YOLOModelManager.saveInstalledModel(model);

      if (mounted) {
        setState(() => _model = model);
      }
    } catch (error) {
      if (mounted) {
        final message = switch (error) {
          FormatException() => error.message,
          YOLOException() => error.message,
          _ => 'Model import failed. Check your connection and scan a new code.',
        };

        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text(message)),
        );
      }
    } finally {
      if (mounted) {
        setState(() => _importId = null);
      }
    }
  }

  Future<void> _loadVersion() async {
    final info = await PackageInfo.fromPlatform();

    if (!mounted) return;

    setState(() {
      _versionLabel = 'v${info.version} (${info.buildNumber})';
    });
  }

  Future<void> _onCapture(Uint8List bytes) async {
    // Snapshot the frames at the moment the user taps Share. Later camera
    // events must not change the data paired with this captured image.
    final frames = _inferenceFrames
        .map((frame) => Map<String, dynamic>.from(frame))
        .toList(growable: false);

    final box = context.findRenderObject() as RenderBox?;
    final origin = box != null && box.hasSize
        ? box.localToGlobal(Offset.zero) & box.size
        : null;

    try {
      final files = await _createCaptureExports(bytes, frames);
      if (!mounted) return;

      await SharePlus.instance.share(
        ShareParams(
          files: files,
          text: 'YOLO real-time detection export '
              '(${frames.length} inference frames)',
          sharePositionOrigin: origin,
        ),
      );
    } catch (error) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Export failed: $error')),
        );
      }
    }
  }

  // Navigate to the existing single-image detection screen.
  void _openGalleryDetection() {
    Navigator.of(context).pushNamed('/single');
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        children: [
          YOLOShowcase(
            versionLabel: _versionLabel,
            onCapture: _onCapture,
            onStreamingData: _onStreamingData,
            model: _model,
            onImportModel: _importModel,
          ),

          // Gallery detection button.
          if (_importId == null)
            Positioned(
              top: 8,
              right: 12,
              child: SafeArea(
                child: FilledButton.tonalIcon(
                  onPressed: _openGalleryDetection,
                  icon: const Icon(Icons.photo_library),
                  label: const Text('相册检测'),
                ),
              ),
            ),

          // Model download overlay.
          if (_importId != null)
            Positioned.fill(
              child: ModelSwitchLoadingOverlay(
                statusText: _fraction < 1
                    ? 'Downloading model'
                    : 'Installing model',
                progress: _fraction < 1 ? _fraction : null,
                onCancel: _fraction >= 1
                    ? null
                    : () => YOLOModelManager.cancelDownload(
                          _importId!,
                        ),
              ),
            ),
        ],
      ),
    );
  }
}
