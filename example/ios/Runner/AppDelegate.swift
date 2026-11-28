// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import Flutter
import UIKit
import Darwin

// Canonical Flutter 3.41 UIScene setup. Plugins are registered against the implicit engine.
@main
@objc class AppDelegate: FlutterAppDelegate, FlutterImplicitEngineDelegate {
  private let performanceChannelName = "com.ultralytics.yolo_example/performance"

  override func application(
    _ application: UIApplication,
    didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
  ) -> Bool {
    return super.application(application, didFinishLaunchingWithOptions: launchOptions)
  }

  func didInitializeImplicitFlutterEngine(_ engineBridge: FlutterImplicitEngineBridge) {
    GeneratedPluginRegistrant.register(with: engineBridge.pluginRegistry)

    let channel = FlutterMethodChannel(
      name: performanceChannelName,
      binaryMessenger: engineBridge.applicationRegistrar.messenger()
    )
    channel.setMethodCallHandler { call, result in
      guard call.method == "getProcessMetrics" else {
        result(FlutterMethodNotImplemented)
        return
      }
      result(self.readProcessMetrics())
    }
  }

  private func readProcessMetrics() -> [String: Any] {
    // getrusage returns cumulative CPU time for this app process. Dart computes
    // the average process CPU percentage over each inference interval.
    var usage = rusage()
    let usageResult = getrusage(RUSAGE_SELF, &usage)
    let cpuSeconds: Double? = usageResult == 0
      ? Double(usage.ru_utime.tv_sec + usage.ru_stime.tv_sec)
        + Double(usage.ru_utime.tv_usec + usage.ru_stime.tv_usec) / 1_000_000.0
      : nil

    // Resident memory of the current app process, in bytes.
    var taskInfo = mach_task_basic_info()
    var count = mach_msg_type_number_t(
      MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size
    )
    let taskInfoResult = withUnsafeMutablePointer(to: &taskInfo) { infoPointer in
      infoPointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
        task_info(
          mach_task_self_,
          task_flavor_t(MACH_TASK_BASIC_INFO),
          $0,
          &count
        )
      }
    }
    let memoryBytes: Int64? = taskInfoResult == KERN_SUCCESS
      ? Int64(taskInfo.resident_size)
      : nil

    var metrics: [String: Any] = [:]
    if let cpuSeconds = cpuSeconds {
      metrics["cpu_time_seconds"] = cpuSeconds
    }
    if let memoryBytes = memoryBytes {
      metrics["memory_bytes"] = memoryBytes
    }
    return metrics
  }
}
