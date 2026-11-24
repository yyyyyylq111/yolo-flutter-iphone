// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:package_info_plus/package_info_plus.dart';
import 'package:share_plus/share_plus.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';
import 'package:ultralytics_yolo_example/services/local_inference_service.dart';

/// Member A: iPhone local inference and repeatable benchmark screen.
/// Protocol: 10 warm-up runs (flagged/excluded) then 3 measured runs/image.
class SingleImageScreen extends StatefulWidget {
  const SingleImageScreen({super.key});
  @override
  State<SingleImageScreen> createState() => _SingleImageScreenState();
}

class _SingleImageScreenState extends State<SingleImageScreen> {
  static const _warmups = 10;
  static const _repeats = 3;
  final _picker = ImagePicker();
  late final LocalInferenceService _service;
  List<YOLOResult> _detections = const [];
  Uint8List? _sourceImage, _annotatedImage;
  LocalInferenceOutput? _lastOutput;
  Map<String,dynamic> _summary = {};
  bool _ready = false, _busy = false;
  String? _version;
  String _network = 'wifi', _progress = '';
  int _done = 0, _total = 0;

  @override
  void initState() {
    super.initState();
    _service = LocalInferenceService(deviceId: 'iphone15', network: _network);
    _loadVersion();
    _initialize();
  }

  Future<void> _loadVersion() async {
    try {
      final info = await PackageInfo.fromPlatform();
      if (mounted) setState(() => _version='v${info.version} (${info.buildNumber})');
    } catch (_) {}
  }

  Future<void> _initialize() async {
    try {
      await _service.initialize();
      if (mounted) setState(()=>_ready=true);
    } catch(e) { if(mounted) _message('模型加载失败：$e'); }
  }

  String _imageId(XFile file) {
    final dot=file.name.lastIndexOf('.');
    return dot>0 ? file.name.substring(0,dot) : file.name;
  }

  void _applyOutput(LocalInferenceOutput output, Uint8List bytes) {
    _lastOutput=output;
    _sourceImage=bytes;
    _annotatedImage=output.annotatedImage;
    _detections=output.detections.map(YOLOResult.fromMap).toList(growable:false);
  }

  Future<void> _singleImage() async {
    if(!_ready || _busy) return;
    try {
      final file=await _picker.pickImage(source:ImageSource.gallery);
      if(file==null || !mounted) return;
      final bytes=await file.readAsBytes();
      setState(()=>_busy=true);
      final output=await _service.inferLocal(imageId:_imageId(file),imageBytes:bytes);
      if(!mounted) return;
      setState(() { _applyOutput(output,bytes); _summary=_service.buildSummary(); });
    } catch(e) { if(mounted) _message('推理失败：$e'); }
    finally { if(mounted) setState(()=>_busy=false); }
  }

  Future<void> _batchBenchmark() async {
    if(!_ready || _busy) return;
    final files=await _picker.pickMultiImage();
    if(files.isEmpty || !mounted) return;
    setState(() {
      _busy=true; _done=0;
      _total=_warmups+files.length*_repeats;
      _progress='正在准备基准测试...';
    });
    try {
      final firstBytes=await files.first.readAsBytes();
      if(firstBytes.isEmpty) throw StateError('第一张图片为空。');

      // Warm-up is explicitly marked is_warmup=true in CSV.
      for(var i=0;i<_warmups;i++) {
        if(!mounted) return;
        setState(()=>_progress='模型预热 ${i+1}/$_warmups（不纳入正式统计）');
        final out=await _service.inferLocal(
          imageId:_imageId(files.first),imageBytes:firstBytes,isWarmup:true);
        _done++;
        if(i==_warmups-1) setState(()=>_applyOutput(out,firstBytes));
        if(mounted) setState((){});
      }

      for(var imageIndex=0;imageIndex<files.length;imageIndex++) {
        final file=files[imageIndex];
        final bytes=imageIndex==0 ? firstBytes : await file.readAsBytes();
        if(bytes.isEmpty) throw StateError('${file.name} 图片为空，测试已停止。');
        for(var repeat=0;repeat<_repeats;repeat++) {
          if(!mounted) return;
          setState(()=>_progress='正式测试 ${file.name}：${repeat+1}/$_repeats');
          final out=await _service.inferLocal(imageId:_imageId(file),imageBytes:bytes);
          _done++;
          setState(()=>_applyOutput(out,bytes));
          if(mounted) setState((){});
        }
      }
      if(!mounted) return;
      setState(()=>_summary=_service.buildSummary());
      _message('测试完成：${files.length} 张图片，正式推理 ${files.length*_repeats} 次，预热 $_warmups 次。');
    } catch(e) { if(mounted) _message('批量测试中断：$e'); }
    finally {
      if(mounted) setState(()=>_busy=false);
    }
  }

  Future<void> _export() async {
    if(_service.recordedRunCount==0) { _message('暂无正式测试记录。'); return; }
    try {
      final f=await _service.exportResults();
      await SharePlus.instance.share(ShareParams(
        files:[XFile(f.resultsCsv.path),XFile(f.resourcesCsv.path),
          XFile(f.detectionsJson.path),XFile(f.summaryJson.path)],
        text:'YOLO26n Local Benchmark / iPhone 15. '
          '正式分析请筛选 is_warmup=false。',
      ));
    } catch(e) { if(mounted) _message('导出失败：$e'); }
  }

  void _message(String text) => ScaffoldMessenger.of(context)
      .showSnackBar(SnackBar(content:Text(text)));

  String _fmt(dynamic v,String suffix) => v is num ? '${v.toStringAsFixed(2)}$suffix' : '—';

  @override
  void dispose() { _service.dispose(); super.dispose(); }

  @override
  Widget build(BuildContext context) {
    final image=_annotatedImage ?? _sourceImage;
    final summary=_summary.isEmpty ? _service.buildSummary() : _summary;
    return Scaffold(
      appBar:AppBar(
        title:const Text('Local Inference & Benchmark'),
        actions:[IconButton(
          tooltip:'导出 CSV / JSON',
          onPressed:!_busy && _service.recordedRunCount>0 ? _export : null,
          icon:const Icon(Icons.ios_share),
        )],
      ),
      body:Column(children:[
        Padding(
          padding:const EdgeInsets.fromLTRB(12,8,12,4),
          child:Column(crossAxisAlignment:CrossAxisAlignment.stretch,children:[
            if(!_ready) const LinearProgressIndicator(),
            if(!_ready) const Text('正在加载 YOLO26n...',textAlign:TextAlign.center),
            Row(children:[
              const Text('实验网络标签'),
              const SizedBox(width:12),
              Expanded(child:DropdownButton<String>(
                value:_network,isExpanded:true,
                items:const [
                  DropdownMenuItem(value:'wifi',child:Text('Wi-Fi')),
                  DropdownMenuItem(value:'5g',child:Text('5G')),
                  DropdownMenuItem(value:'4g',child:Text('4G')),
                  DropdownMenuItem(value:'offline',child:Text('离线')),
                  DropdownMenuItem(value:'unknown',child:Text('未知')),
                ],
                onChanged:_busy?null:(v){if(v!=null)setState((){_network=v;_service.network=v;});},
              )),
            ]),
            FilledButton.icon(
              onPressed:_ready&&!_busy?_singleImage:null,
              icon:const Icon(Icons.photo_library_outlined),
              label:const Text('选择单张图片并检测'),
            ),
            const SizedBox(height:6),
            OutlinedButton.icon(
              onPressed:_ready&&!_busy?_batchBenchmark:null,
              icon:const Icon(Icons.speed),
              label:const Text('批量基准测试（10 次预热 + 每张图片 3 次）'),
            ),
            if(_busy) ...[
              const SizedBox(height:8),
              LinearProgressIndicator(value:_total==0?null:_done/_total),
              Text('$_progress  ($_done/$_total)',textAlign:TextAlign.center),
            ],
          ]),
        ),
        Expanded(child:ListView(padding:const EdgeInsets.only(bottom:28),children:[
          if(_lastOutput!=null) Card(
            margin:const EdgeInsets.all(12),
            child:Padding(padding:const EdgeInsets.all(12),child:Wrap(
              spacing:18,runSpacing:10,children:[
                _Metric('Image ID',_lastOutput!.imageId),
                _Metric('Preprocess',_fmt(_lastOutput!.preprocessMs,' ms')),
                _Metric('Inference','${_lastOutput!.inferenceMs.toStringAsFixed(2)} ms'),
                _Metric('Postprocess',_fmt(_lastOutput!.postprocessMs,' ms')),
                _Metric('Dart result parse','${_lastOutput!.dartPostprocessMs.toStringAsFixed(2)} ms'),
                _Metric('End-to-end','${_lastOutput!.endToEndMs.toStringAsFixed(2)} ms'),
                _Metric('FPS',_lastOutput!.fps.toStringAsFixed(2)),
                _Metric('CPU',_fmt(_lastOutput!.cpuPercent,'%')),
                _Metric('Memory',_fmt(_lastOutput!.memoryAfterMb,' MB')),
              ],
            )),
          ),
          if(image!=null) Padding(
            padding:const EdgeInsets.symmetric(horizontal:12),
            child:ClipRRect(borderRadius:BorderRadius.circular(12),
              child:Image.memory(image,fit:BoxFit.contain)),
          ),
          if(_lastOutput!=null) Padding(
            padding:const EdgeInsets.fromLTRB(16,12,16,4),
            child:Text('Detections (${_detections.length})',
              style:Theme.of(context).textTheme.titleMedium),
          ),
          for(final d in _detections) ListTile(
            dense:true,leading:const Icon(Icons.crop_square),
            title:Text(d.className),
            trailing:Text('${(d.confidence*100).toStringAsFixed(1)}%'),
          ),
          if(_lastOutput!=null && _detections.isEmpty)
            const Padding(padding:EdgeInsets.all(16),child:Text('No objects detected.')),
          Padding(
            padding:const EdgeInsets.fromLTRB(16,18,16,6),
            child:Text('正式实验统计（不含预热）',
              style:Theme.of(context).textTheme.titleMedium),
          ),
          Card(margin:const EdgeInsets.symmetric(horizontal:12),
            child:Padding(padding:const EdgeInsets.all(12),child:Wrap(
              spacing:18,runSpacing:12,children:[
                _Metric('正式次数','${summary['measured_runs']??0}'),
                _Metric('预热次数','${summary['warmup_runs']??0}'),
                _Metric('失败次数','${summary['failed_runs']??0}'),
                _Metric('平均端到端',_fmt(summary['mean_end_to_end_ms'],' ms')),
                _Metric('中位数',_fmt(summary['median_end_to_end_ms'],' ms')),
                _Metric('P95',_fmt(summary['p95_end_to_end_ms'],' ms')),
                _Metric('平均预处理',_fmt(summary['mean_preprocess_ms'],' ms')),
                _Metric('平均推理',_fmt(summary['mean_inference_ms'],' ms')),
                _Metric('平均后处理',_fmt(summary['mean_postprocess_ms'],' ms')),
                _Metric('平均 FPS',_fmt(summary['mean_fps'],' fps')),
                _Metric('平均 CPU',_fmt(summary['mean_cpu_percent'],'%')),
                _Metric('平均内存',_fmt(summary['mean_memory_after_mb'],' MB')),
              ],
            )),
          ),
          const Padding(
            padding:EdgeInsets.fromLTRB(16,12,16,20),
            child:Text(
              'CPU 与内存由 iOS 原生 Mach API 采样。CPU 是应用进程在推理时间窗内的设备算力归一化占用率；'
              '内存是应用进程常驻内存。preprocess_ms、inference_ms、postprocess_ms '
              '读取 iOS 原生 YOLO 返回的 preMs、inferenceMs、postMs；Dart 检测结果解析时间单独记录。'
              'Local 模式 request_bytes=0，'
              '原始图片字节数记录在 input_bytes。网络标签仅用于记录测试环境。',
            ),
          ),
          if(_version!=null) Padding(
            padding:const EdgeInsets.only(left:12,bottom:8),
            child:Text(_version!,style:Theme.of(context).textTheme.bodySmall),
          ),
        ])),
      ]),
    );
  }
}

class _Metric extends StatelessWidget {
  const _Metric(this.label,this.value);
  final String label,value;
  @override
  Widget build(BuildContext context)=>SizedBox(width:138,child:Column(
    crossAxisAlignment:CrossAxisAlignment.start,children:[
      Text(label,style:Theme.of(context).textTheme.bodySmall),
      const SizedBox(height:3),
      Text(value,style:Theme.of(context).textTheme.titleSmall),
    ],
  ));
}
