// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'dart:io';
import 'dart:typed_data';

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
  String? _lastImageExtension;
  double? _lastLatencyMs;

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

  Future<void> _pickAndPredict() async {
    if (!_isModelReady || _isInferring || _isExporting) return;

    try {
      final file = await _picker.pickImage(source: ImageSource.gallery);
      if (file == null || !mounted) return;

      final bytes = await file.readAsBytes();
      final imageId = _imageIdFromFile(file);
      final extension = _extensionFromName(file.name);

      setState(() {
        _isInferring = true;
        _imageBytes = bytes;
        _annotatedImage = null;
        _detections = const [];
        _lastImageId = imageId;
        _lastImageExtension = extension;
        _lastLatencyMs = null;
      });

      final output = await _localService.inferLocal(
        imageId: imageId,
        imageBytes: bytes,
      );

      if (!mounted) return;
      final detections = output.detections
          .map((item) => YOLOResult.fromMap(item))
          .toList(growable: false);

      setState(() {
        _detections = detections;
        _annotatedImage = output.annotatedImage;
        _lastLatencyMs = output.latencyMs;
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

  String _extensionFromName(String name) {
    final dot = name.lastIndexOf('.');
    if (dot < 0 || dot == name.length - 1) return 'jpg';
    final extension = name.substring(dot + 1).toLowerCase();
    const supported = {'jpg', 'jpeg', 'png', 'webp', 'heic'};
    return supported.contains(extension) ? extension : 'jpg';
  }

  String _extensionFromBytes(Uint8List bytes, {String fallback = 'jpg'}) {
    if (bytes.length >= 8 &&
        bytes[0] == 0x89 &&
        bytes[1] == 0x50 &&
        bytes[2] == 0x4E &&
        bytes[3] == 0x47) {
      return 'png';
    }
    if (bytes.length >= 3 &&
        bytes[0] == 0xFF &&
        bytes[1] == 0xD8 &&
        bytes[2] == 0xFF) {
      return 'jpg';
    }
    if (bytes.length >= 12 &&
        String.fromCharCodes(bytes.sublist(0, 4)) == 'RIFF' &&
        String.fromCharCodes(bytes.sublist(8, 12)) == 'WEBP') {
      return 'webp';
    }
    return fallback;
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

    setState(() => _isExporting = true);

    try {
      // Write the CSV and JSON records first.
      final resultFiles = await _localService.exportResults();

      // Prefer the image with YOLO boxes. Fall back to the original image.
      final imageBytes = _annotatedImage ?? _imageBytes!;
      final imageExtension = _extensionFromBytes(
        imageBytes,
        fallback: _annotatedImage == null
            ? (_lastImageExtension ?? 'jpg')
            : 'jpg',
      );

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
        _showSnackBar('The iOS share sheet is unavailable.');
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
            tooltip: 'Share CSV, JSON and detected image',
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
                padding: const EdgeInsets.all(12),
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
                      Text('${_lastLatencyMs!.toStringAsFixed(2)} ms'),
                    ],
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
                          'Recorded runs: ${_localService.recordedRunCount}. '
                          'Tap the share icon above to export CSV, JSON and the detected image.',
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
