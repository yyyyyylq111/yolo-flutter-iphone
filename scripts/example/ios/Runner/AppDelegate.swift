// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import Flutter
import UIKit
import Darwin

@main
@objc class AppDelegate: FlutterAppDelegate, FlutterImplicitEngineDelegate {
  private var resourceChannel: FlutterMethodChannel?
  private var previousCpuSeconds: Double?
  private var previousCpuTimestamp: TimeInterval?

  override func application(
    _ application: UIApplication,
    didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
  ) -> Bool {
    return super.application(application, didFinishLaunchingWithOptions: launchOptions)
  }

  func didInitializeImplicitFlutterEngine(_ engineBridge: FlutterImplicitEngineBridge) {
    GeneratedPluginRegistrant.register(with: engineBridge.pluginRegistry)

    let registrar = engineBridge.pluginRegistry.registrar(forPlugin: "LocalDeviceResourceMetrics")
    let channel = FlutterMethodChannel(
      name: "ultralytics_yolo_example/device_resources",
      binaryMessenger: registrar.messenger()
    )
    resourceChannel = channel
    channel.setMethodCallHandler { [weak self] call, result in
      guard call.method == "snapshot" else {
        result(FlutterMethodNotImplemented)
        return
      }
      guard let self = self else {
        result(FlutterError(code: "UNAVAILABLE", message: "AppDelegate unavailable", details: nil))
        return
      }
      result(self.resourceSnapshot())
    }
  }

  // Samples this app process only. CPU is normalized to total device capacity
  // (0...100%); memory is resident memory in MiB.
  private func resourceSnapshot() -> [String: Any] {
    let now = Date().timeIntervalSince1970
    let cpuSeconds = processCpuSeconds()
    var cpuPercent: Double? = nil

    if let current = cpuSeconds,
       let previous = previousCpuSeconds,
       let previousTime = previousCpuTimestamp {
      let elapsedWall = now - previousTime
      let elapsedCpu = current - previous
      if elapsedWall > 0 && elapsedCpu >= 0 {
        let cores = max(1, ProcessInfo.processInfo.activeProcessorCount)
        cpuPercent = min(100.0, max(0.0, (elapsedCpu / elapsedWall) * 100.0 / Double(cores)))
      }
    }
    previousCpuSeconds = cpuSeconds
    previousCpuTimestamp = now

    let memoryBytes = residentMemoryBytes()
    var output: [String: Any] = [
      "available": memoryBytes != nil,
      "physical_memory_mb": Double(ProcessInfo.processInfo.physicalMemory) / 1_048_576.0
    ]
    if let memoryBytes = memoryBytes {
      output["memory_mb"] = Double(memoryBytes) / 1_048_576.0
    }
    if let cpuPercent = cpuPercent {
      output["cpu_percent"] = cpuPercent
    }
    return output
  }

  private func residentMemoryBytes() -> UInt64? {
    var info = mach_task_basic_info()
    var count = mach_msg_type_number_t(
      MemoryLayout<mach_task_basic_info>.size / MemoryLayout<integer_t>.size
    )
    let capacity = Int(count)
    let result = withUnsafeMutablePointer(to: &info) { pointer in
      pointer.withMemoryRebound(to: integer_t.self, capacity: capacity) {
        task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
      }
    }
    return result == KERN_SUCCESS ? info.resident_size : nil
  }

  private func processCpuSeconds() -> Double? {
    var threadList: thread_act_array_t?
    var threadCount: mach_msg_type_number_t = 0
    let kr = task_threads(mach_task_self_, &threadList, &threadCount)
    guard kr == KERN_SUCCESS, let threads = threadList else { return nil }

    var total: Double = 0
    for index in 0..<Int(threadCount) {
      var info = thread_basic_info()
      var infoCount = mach_msg_type_number_t(
        MemoryLayout<thread_basic_info>.size / MemoryLayout<integer_t>.size
      )
      let infoCapacity = Int(infoCount)
      let threadResult = withUnsafeMutablePointer(to: &info) { pointer in
        pointer.withMemoryRebound(to: integer_t.self, capacity: infoCapacity) {
          thread_info(threads[index], thread_flavor_t(THREAD_BASIC_INFO), $0, &infoCount)
        }
      }
      if threadResult == KERN_SUCCESS && (info.flags & TH_FLAGS_IDLE) == 0 {
        total += Double(info.user_time.seconds)
        total += Double(info.user_time.microseconds) / 1_000_000.0
        total += Double(info.system_time.seconds)
        total += Double(info.system_time.microseconds) / 1_000_000.0
      }
      mach_port_deallocate(mach_task_self_, threads[index])
    }

    let bytes = vm_size_t(Int(threadCount) * MemoryLayout<thread_t>.stride)
    vm_deallocate(mach_task_self_, vm_address_t(UInt(bitPattern: UnsafeRawPointer(threads))), bytes)
    return total
  }
}
