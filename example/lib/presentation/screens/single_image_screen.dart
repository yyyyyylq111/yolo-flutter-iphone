// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:package_info_plus/package_info_plus.dart';
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
  String? _versionLabel;
  String? _lastImageId;
  double? _lastLatencyMs;

  @override
  void initState() {
    super.initState();
    _localService = LocalInferenceService(
      modelPath: 'yolo26n',
      deviceId: 'iphone15', // change to your actual test-device label if needed
      network: 'unknown', // record the test network in the experiment protocol
    );
    _loadVersion();
    _initializeYOLO();
  }

  Future<void> _loadVersion() async {
    final info = await PackageInfo.fromPlatform();
    if (!mounted) return;
    setState(() => _versionLabel = 'v${info.version} (${info.buildNumber})');
  }

  Future<void> _initializeYOLO() async {
    try {
      await _localService.initialize();
      if (mounted) setState(() => _isModelReady = true);
    } catch (e) {
      if (mounted) _showSnackBar('Error loading model: $e');
    }
  }

  Future<void> _pickAndPredict() async {
    if (!_isModelReady || _isInferring) return;
    try {
      final file = await _picker.pickImage(source: ImageSource.gallery);
      if (file == null || !mounted) return;
      final bytes = await file.readAsBytes();
      final imageId = _imageIdFromFile(file);
      setState(() {
        _isInferring = true;
        _imageBytes = bytes;
        _annotatedImage = null;
        _detections = const [];
        _lastImageId = imageId;
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
    } catch (e) {
      if (!mounted) return;
      setState(() => _isInferring = false);
      _showSnackBar('Inference failed: $e');
    }
  }

  String _imageIdFromFile(XFile file) {
    final name = file.name;
    final dot = name.lastIndexOf('.');
    final stem = dot > 0 ? name.substring(0, dot) : name;
    return stem.isEmpty ? 'img_${DateTime.now().millisecondsSinceEpoch}' : stem;
  }

  Future<void> _exportResults() async {
    if (_localService.recordedRunCount == 0) {
      _showSnackBar('No inference records to export yet.');
      return;
    }
    try {
      final files = await _localService.exportResults();
      await SharePlus.instance.share(
        ShareParams(
          files: [XFile(files.csv.path), XFile(files.json.path)],
          text: 'YOLO Local benchmark results',
        ),
      );
    } catch (e) {
      if (mounted) _showSnackBar('Export failed: $e');
    }
  }

  void _showSnackBar(String message) {
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(message)));
  }

  @override
  void dispose() {
    _localService.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final displayedImage = _annotatedImage ?? _imageBytes;
    return Scaffold(
      appBar: AppBar(
        title: const Text('Single Image Inference'),
        actions: [
          IconButton(
            tooltip: 'Export CSV and JSON',
            onPressed: _localService.recordedRunCount > 0 && !_isInferring
                ? _exportResults
                : null,
            icon: const Icon(Icons.ios_share),
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
                    onPressed: _isModelReady && !_isInferring ? _pickAndPredict : null,
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
                      SizedBox(width: 18, height: 18,
                          child: CircularProgressIndicator(strokeWidth: 2)),
                      SizedBox(width: 12),
                      Text('Preparing YOLO26 model...'),
                    ],
                  ),
                ),
              if (_isInferring) const LinearProgressIndicator(minHeight: 2),
              if (_lastLatencyMs != null)
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
                  child: Row(
                    children: [
                      Expanded(child: Text('Image: ${_lastImageId ?? '-'}',
                          overflow: TextOverflow.ellipsis)),
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
                          child: Image.memory(displayedImage, fit: BoxFit.contain),
                        ),
                      ),
                    const SizedBox(height: 12),
                    if (_detections.isNotEmpty)
                      Padding(
                        padding: const EdgeInsets.symmetric(horizontal: 16),
                        child: Text('Detections (${_detections.length})',
                            style: Theme.of(context).textTheme.titleMedium),
                      ),
                    for (final d in _detections)
                      ListTile(
                        dense: true,
                        leading: const Icon(Icons.crop_din),
                        title: Text(d.className),
                        trailing: Text('${(d.confidence * 100).toStringAsFixed(1)}%'),
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
                          'Tap the share icon above to export CSV and JSON.',
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
