// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'dart:io';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:package_info_plus/package_info_plus.dart';
import 'package:path_provider/path_provider.dart';
import 'package:share_plus/share_plus.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

import 'package:ultralytics_yolo_example/services/local_inference_service.dart';

class SingleImageScreen extends StatefulWidget {
  const SingleImageScreen({super.key});

  @override
  State<SingleImageScreen> createState() => _SingleImageScreenState();
}

class _SingleImageScreenState extends State<SingleImageScreen> {
  final _picker = ImagePicker();
  late final LocalInferenceService _localService;

  List<YOLOResult> _detections = const [];
  Uint8List? _imageBytes;
  Uint8List? _annotatedImage;
  bool _isModelReady = false;
  bool _isInferring = false;
  bool _isExporting = false;
  String? _versionLabel;
  String? _lastImageId;
  double? _lastLatencyMs;
  int? _lastImageWidth;
  int? _lastImageHeight;
  double? _lastEndToEndMs;
  double? _lastAnnotationMs;

  @override
  void initState() {
    super.initState();
    _localService = LocalInferenceService(
      modelPath: 'yolo26n',
      deviceId: 'iphone15', // Set this to the actual test-device label.
      network: 'unknown', // Set this in the experiment protocol if known.
    );
    _loadVersion();
    _initializeYOLO();
  }

  Future<void> _loadVersion() async {
    try {
      final info = await PackageInfo.fromPlatform();
      if (!mounted) return;
      setState(() => _versionLabel = 'v${info.version} (${info.buildNumber})');
    } catch (error) {
      debugPrint('Could not read app version: $error');
    }
  }

  Future<void> _initializeYOLO() async {
    try {
      await _localService.initialize();
      if (mounted) setState(() => _isModelReady = true);
    } catch (error) {
      if (mounted) _showSnackBar('Error loading model: $error');
    }
  }


  /// Renders the detection boxes and labels into a real PNG image.
  /// Uses normalized coordinates when available so boxes stay aligned with
  /// the decoded image dimensions.
  Future<Uint8List> _renderAnnotatedImage(
    Uint8List sourceBytes,
    List<Map<String, dynamic>> detections,
  ) async {
    final codec = await ui.instantiateImageCodec(sourceBytes);
    final frame = await codec.getNextFrame();
    final sourceImage = frame.image;

    try {
      final width = sourceImage.width;
      final height = sourceImage.height;
      final recorder = ui.PictureRecorder();
      final canvas = Canvas(
        recorder,
        Rect.fromLTWH(0, 0, width.toDouble(), height.toDouble()),
      );

      canvas.drawImage(
        sourceImage,
        Offset.zero,
        Paint()..filterQuality = FilterQuality.high,
      );

      final boxPaint = Paint()
        ..color = const Color(0xFF00E676)
        ..style = PaintingStyle.stroke
        ..strokeWidth = (width < height ? width : height) * 0.006;

      final labelBackgroundPaint = Paint()
        ..color = const Color(0xDD00A85A)
        ..style = PaintingStyle.fill;

      final strokeWidth = boxPaint.strokeWidth.clamp(2.0, 6.0).toDouble();

      for (final detection in detections) {
        final coordinates = _detectionRect(detection, width, height);
        if (coordinates == null) continue;

        final rect = coordinates;
        canvas.drawRect(rect, boxPaint);

        final className = (detection['class_name'] ??
                detection['className'] ??
                detection['label'] ??
                detection['name'] ??
                detection['class'] ??
                'object')
            .toString();
        final confidenceValue = detection['confidence'] ??
            detection['conf'] ??
            detection['score'];
        final confidence = confidenceValue is num
            ? '${(confidenceValue * 100).toStringAsFixed(1)}%'
            : '';
        final label = confidence.isEmpty
            ? className
            : '$className $confidence';

        final textPainter = TextPainter(
          text: TextSpan(
            text: label,
            style: TextStyle(
              color: const Color(0xFFFFFFFF),
              fontSize: (width * 0.025).clamp(12.0, 22.0).toDouble(),
              fontWeight: FontWeight.w600,
            ),
          ),
          textDirection: TextDirection.ltr,
          maxLines: 1,
        )..layout(maxWidth: width.toDouble());

        final horizontalPadding = strokeWidth * 1.5;
        final verticalPadding = strokeWidth;
        final labelWidth = textPainter.width + horizontalPadding * 2;
        final labelHeight = textPainter.height + verticalPadding * 2;
        final labelLeft = rect.left
            .clamp(
              0.0,
              (width - labelWidth).clamp(0.0, width.toDouble()),
            )
            .toDouble();
        final labelTop = (rect.top - labelHeight)
            .clamp(
              0.0,
              (height - labelHeight).clamp(0.0, height.toDouble()),
            )
            .toDouble();

        canvas.drawRRect(
          RRect.fromRectAndRadius(
            Rect.fromLTWH(labelLeft, labelTop, labelWidth, labelHeight),
            const Radius.circular(3),
          ),
          labelBackgroundPaint,
        );
        textPainter.paint(
          canvas,
          Offset(labelLeft + horizontalPadding, labelTop + verticalPadding),
        );
      }

      final picture = recorder.endRecording();
      final renderedImage = await picture.toImage(width, height);
      try {
        final byteData = await renderedImage.toByteData(
          format: ui.ImageByteFormat.png,
        );
        if (byteData == null) {
          throw StateError('Could not encode annotated image as PNG.');
        }
        return byteData.buffer.asUint8List(
          byteData.offsetInBytes,
          byteData.lengthInBytes,
        );
      } finally {
        renderedImage.dispose();
        picture.dispose();
      }
    } finally {
      sourceImage.dispose();
      codec.dispose();
    }
  }

  Rect? _detectionRect(
    Map<String, dynamic> detection,
    int imageWidth,
    int imageHeight,
  ) {
    final normalized = detection['normalizedBox'];
    List<num>? values;

    if (normalized is List && normalized.length >= 4) {
      final parsed = normalized.take(4).map(_asNum).toList();
      if (parsed.every((value) => value != null)) {
        values = parsed.cast<num>();
        final left = (values[0] * imageWidth).toDouble();
        final top = (values[1] * imageHeight).toDouble();
        final right = (values[2] * imageWidth).toDouble();
        final bottom = (values[3] * imageHeight).toDouble();
        return _clampedRect(left, top, right, bottom, imageWidth, imageHeight);
      }
    }

    final rawBox = detection['boundingBox'] ??
        detection['bounding_box'] ??
        detection['bbox'] ??
        detection['box'];

    if (rawBox is Map) {
      final left = _asNum(rawBox['left'] ?? rawBox['x1'] ?? rawBox['x_min']);
      final top = _asNum(rawBox['top'] ?? rawBox['y1'] ?? rawBox['y_min']);
      final right =
          _asNum(rawBox['right'] ?? rawBox['x2'] ?? rawBox['x_max']);
      final bottom =
          _asNum(rawBox['bottom'] ?? rawBox['y2'] ?? rawBox['y_max']);
      if (left != null && top != null && right != null && bottom != null) {
        return _clampedRect(
          left.toDouble(),
          top.toDouble(),
          right.toDouble(),
          bottom.toDouble(),
          imageWidth,
          imageHeight,
        );
      }
    } else if (rawBox is List && rawBox.length >= 4) {
      final parsed = rawBox.take(4).map(_asNum).toList();
      if (parsed.every((value) => value != null)) {
        final box = parsed.cast<num>();
        return _clampedRect(
          box[0].toDouble(),
          box[1].toDouble(),
          box[2].toDouble(),
          box[3].toDouble(),
          imageWidth,
          imageHeight,
        );
      }
    }

    return null;
  }

  num? _asNum(dynamic value) {
    if (value is num) return value;
    if (value is String) return num.tryParse(value);
    return null;
  }

  Rect? _clampedRect(
    double left,
    double top,
    double right,
    double bottom,
    int imageWidth,
    int imageHeight,
  ) {
    final x1 = left.clamp(0.0, imageWidth.toDouble()).toDouble();
    final y1 = top.clamp(0.0, imageHeight.toDouble()).toDouble();
    final x2 = right.clamp(0.0, imageWidth.toDouble()).toDouble();
    final y2 = bottom.clamp(0.0, imageHeight.toDouble()).toDouble();
    if (x2 <= x1 || y2 <= y1) return null;
    return Rect.fromLTRB(x1, y1, x2, y2);
  }

  Future<({int width, int height})> _readImageDimensions(
    Uint8List bytes,
  ) async {
    final codec = await ui.instantiateImageCodec(bytes);
    try {
      final frame = await codec.getNextFrame();
      final dimensions = (width: frame.image.width, height: frame.image.height);
      frame.image.dispose();
      return dimensions;
    } finally {
      codec.dispose();
    }
  }

  Future<void> _pickAndPredict() async {
    if (!_isModelReady || _isInferring || _isExporting) return;

    try {
      final file = await _picker.pickImage(source: ImageSource.gallery);
      if (file == null || !mounted) return;

      final endToEndWatch = Stopwatch()..start();
      final bytes = await file.readAsBytes();
      final dimensions = await _readImageDimensions(bytes);
      final imageId = _imageIdFromFile(file);

      setState(() {
        _isInferring = true;
        _imageBytes = bytes;
        _annotatedImage = null;
        _detections = const [];
        _lastImageId = imageId;
        _lastLatencyMs = null;
        _lastEndToEndMs = null;
        _lastAnnotationMs = null;
        _lastImageWidth = dimensions.width;
        _lastImageHeight = dimensions.height;
      });

      final output = await _localService.inferLocal(
        imageId: imageId,
        imageBytes: bytes,
      );

      if (!mounted) return;
      final detections = output.detections
          .map((item) => YOLOResult.fromMap(item))
          .toList(growable: false);

      // Render the annotation locally from the original pixels and the
      // returned boxes. This guarantees the exported image contains boxes
      // even when the native YOLO plugin does not return annotatedImage.
      final annotationWatch = Stopwatch()..start();
      final annotatedImage = await _renderAnnotatedImage(
        bytes,
        output.detections,
      );
      annotationWatch.stop();
      endToEndWatch.stop();
      final annotationMs = annotationWatch.elapsedMicroseconds / 1000.0;
      final endToEndMs = endToEndWatch.elapsedMicroseconds / 1000.0;
      _localService.completeRunMetrics(
        requestId: output.requestId,
        endToEndMs: endToEndMs,
        imageWidth: dimensions.width,
        imageHeight: dimensions.height,
        annotationMs: annotationMs,
        imageFormat: file.name.contains('.')
            ? file.name.split('.').last.toLowerCase()
            : 'unknown',
      );

      if (!mounted) return;
      setState(() {
        _detections = detections;
        _annotatedImage = annotatedImage;
        _lastLatencyMs = output.latencyMs;
        _lastEndToEndMs = endToEndMs;
        _lastAnnotationMs = annotationMs;
        _isInferring = false;
      });
    } catch (error) {
      if (!mounted) return;
      setState(() => _isInferring = false);
      _showSnackBar('Inference failed: $error');
    }
  }

  String _imageIdFromFile(XFile file) {
    final name = file.name;
    final dot = name.lastIndexOf('.');
    final stem = dot > 0 ? name.substring(0, dot) : name;
    return stem.isEmpty
        ? 'img_${DateTime.now().millisecondsSinceEpoch}'
        : stem;
  }

  String _mimeType(String extension) {
    switch (extension) {
      case 'png':
        return 'image/png';
      case 'webp':
        return 'image/webp';
      case 'heic':
        return 'image/heic';
      default:
        return 'image/jpeg';
    }
  }

  Future<void> _exportResults() async {
    if (_isInferring || _isExporting) return;
    if (_localService.recordedRunCount == 0) {
      _showSnackBar('No inference records to export yet.');
      return;
    }
    if (_imageBytes == null || _lastImageId == null) {
      _showSnackBar('Run an image inference before exporting.');
      return;
    }
    if (_annotatedImage == null) {
      _showSnackBar('标注图片尚未生成，请重新运行图片检测后再导出。');
      return;
    }

    setState(() => _isExporting = true);

    try {
      // Write the CSV and JSON records first.
      final resultFiles = await _localService.exportResults();

      // Export only the rendered, annotated PNG. Never silently substitute
      // the original unannotated photo.
      final imageBytes = _annotatedImage!;
      const imageExtension = 'png';

      final tempDir = await getTemporaryDirectory();
      final safeImageId = _lastImageId!.replaceAll(
        RegExp(r'[^a-zA-Z0-9_-]'),
        '_',
      );
      final imageFile = File(
        '${tempDir.path}/yolo_${safeImageId}_${DateTime.now().millisecondsSinceEpoch}.$imageExtension',
      );
      await imageFile.writeAsBytes(imageBytes, flush: true);

      final renderObject = context.findRenderObject();
      Rect? shareOrigin;
      if (renderObject is RenderBox && renderObject.hasSize) {
        shareOrigin = renderObject.localToGlobal(Offset.zero) & renderObject.size;
      }

      final shareResult = await SharePlus.instance.share(
        ShareParams(
          files: [
            XFile(resultFiles.csv.path, mimeType: 'text/csv'),
            XFile(resultFiles.json.path, mimeType: 'application/json'),
            XFile(imageFile.path, mimeType: _mimeType(imageExtension)),
          ],
          text: 'YOLO inference export: CSV results, JSON detections, and annotated image.',
          subject: 'YOLO inference results',
          sharePositionOrigin: shareOrigin,
        ),
      );

      if (!mounted) return;
      if (shareResult.status == ShareResultStatus.unavailable) {
        _showSnackBar('iOS 分享面板不可用，请检查系统分享功能。');
      } else if (shareResult.status == ShareResultStatus.success) {
        _showSnackBar('分享操作已完成。如果刚才选择了“存储到文件”，请在“文件”App 中查看。');
      } else {
        _showSnackBar('分享面板已关闭；如未保存文件，可再次点击导出。');
      }
    } catch (error) {
      if (mounted) _showSnackBar('Export failed: $error');
    } finally {
      if (mounted) setState(() => _isExporting = false);
    }
  }

  void _showSnackBar(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(message)),
    );
  }

  @override
  void dispose() {
    _localService.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final displayedImage = _annotatedImage ?? _imageBytes;
    final canExport = _localService.recordedRunCount > 0 &&
        !_isInferring &&
        !_isExporting &&
        _imageBytes != null;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Single Image Inference'),
        actions: [
          IconButton(
            tooltip: '导出 CSV、JSON 和检测图片',
            onPressed: canExport ? _exportResults : null,
            icon: _isExporting
                ? const SizedBox(
                    width: 20,
                    height: 20,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  )
                : const Icon(Icons.ios_share),
          ),
        ],
      ),
      body: Stack(
        children: [
          Column(
            children: [
              Padding(
                padding: const EdgeInsets.fromLTRB(12, 12, 12, 6),
                child: SizedBox(
                  width: double.infinity,
                  child: FilledButton.icon(
                    onPressed: _isModelReady && !_isInferring && !_isExporting
                        ? _pickAndPredict
                        : null,
                    icon: const Icon(Icons.photo_library_outlined),
                    label: const Text('Pick image & run inference'),
                  ),
                ),
              ),
              Padding(
                padding: const EdgeInsets.fromLTRB(12, 6, 12, 12),
                child: SizedBox(
                  width: double.infinity,
                  child: FilledButton.tonalIcon(
                    onPressed: canExport ? _exportResults : null,
                    icon: _isExporting
                        ? const SizedBox(
                            width: 18,
                            height: 18,
                            child: CircularProgressIndicator(strokeWidth: 2),
                          )
                        : const Icon(Icons.file_download_outlined),
                    label: Text(
                      _isExporting
                          ? '正在生成测试数据...'
                          : '导出测试数据（CSV + JSON）',
                    ),
                  ),
                ),
              ),
              if (!_isModelReady)
                const Padding(
                  padding: EdgeInsets.all(8),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      SizedBox(
                        width: 18,
                        height: 18,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      ),
                      SizedBox(width: 12),
                      Text('Preparing YOLO26 model...'),
                    ],
                  ),
                ),
              if (_isInferring) const LinearProgressIndicator(minHeight: 2),
              if (_lastLatencyMs != null)
                Padding(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 16,
                    vertical: 6,
                  ),
                  child: Row(
                    children: [
                      Expanded(
                        child: Text(
                          'Image: ${_lastImageId ?? '-'}',
                          overflow: TextOverflow.ellipsis,
                        ),
                      ),
                      Text('推理 ${_lastLatencyMs!.toStringAsFixed(1)} ms'),
                    ],
                  ),
                ),
              if (_lastEndToEndMs != null)
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 2),
                  child: Text(
                    '端到端 ${_lastEndToEndMs!.toStringAsFixed(1)} ms  |  '
                    '标注 ${_lastAnnotationMs!.toStringAsFixed(1)} ms  |  '
                    '${_lastImageWidth}×${_lastImageHeight}',
                    style: Theme.of(context).textTheme.bodySmall,
                  ),
                ),
              if (_lastLatencyMs != null)
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 2),
                  child: Text(
                    '当前推理 FPS：${(1000.0 / _lastLatencyMs!).toStringAsFixed(2)}',
                    style: Theme.of(context).textTheme.bodySmall,
                  ),
                ),
              Expanded(
                child: ListView(
                  padding: const EdgeInsets.only(bottom: 48),
                  children: [
                    if (displayedImage != null)
                      Padding(
                        padding: const EdgeInsets.symmetric(horizontal: 12),
                        child: ClipRRect(
                          borderRadius: BorderRadius.circular(12),
                          child: Image.memory(
                            displayedImage,
                            fit: BoxFit.contain,
                          ),
                        ),
                      ),
                    const SizedBox(height: 12),
                    if (_imageBytes != null)
                      Padding(
                        padding: const EdgeInsets.symmetric(horizontal: 16),
                        child: Text(
                          'Detections (${_detections.length})',
                          style: Theme.of(context).textTheme.titleMedium,
                        ),
                      ),
                    for (final detection in _detections)
                      ListTile(
                        dense: true,
                        leading: const Icon(Icons.crop_din),
                        title: Text(detection.className),
                        trailing: Text(
                          '${(detection.confidence * 100).toStringAsFixed(1)}%',
                        ),
                      ),
                    if (_detections.isEmpty && displayedImage != null)
                      const Padding(
                        padding: EdgeInsets.all(16),
                        child: Text('No objects detected.'),
                      ),
                    if (_localService.recordedRunCount > 0)
                      Padding(
                        padding: const EdgeInsets.all(16),
                        child: Text(
                          '当前会话已记录 ${_localService.recordedRunCount} 次图片检测，'
                          '成功 ${_localService.successfulRunCount} 次，'
                          '失败 ${_localService.failedRunCount} 次，'
                          '共检测到 ${_localService.totalDetectionCount} 个目标。'
                          '平均推理 FPS：${_localService.averageFps.toStringAsFixed(2)}。'
                          '\n点击“导出测试数据”可通过 iPhone 分享面板保存 CSV、JSON 和标注图片。',
                          style: Theme.of(context).textTheme.bodySmall,
                        ),
                      ),
                  ],
                ),
              ),
            ],
          ),
          if (_versionLabel != null)
            Positioned(
              left: 8,
              bottom: 8,
              child: SafeArea(
                child: Text(
                  _versionLabel!,
                  style: Theme.of(context).textTheme.bodySmall?.copyWith(
                        color: Theme.of(context).colorScheme.onSurfaceVariant,
                      ),
                ),
              ),
            ),
        ],
      ),
    );
  }
}
