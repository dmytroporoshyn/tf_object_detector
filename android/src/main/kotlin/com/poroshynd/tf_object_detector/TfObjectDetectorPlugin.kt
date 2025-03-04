package com.poroshynd.tf_object_detector

import android.content.Context
import android.util.Log

import io.flutter.embedding.engine.plugins.FlutterPlugin
import io.flutter.embedding.engine.plugins.activity.ActivityAware
import io.flutter.embedding.engine.plugins.activity.ActivityPluginBinding
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugin.common.MethodChannel.MethodCallHandler
import io.flutter.plugin.common.MethodChannel.Result

/** TfObjectDetectorPlugin */
class TfObjectDetectorPlugin: FlutterPlugin, MethodCallHandler, ActivityAware {
  /// The MethodChannel that will the communication between Flutter and native Android
  ///
  /// This local reference serves to register the plugin with the Flutter Engine and unregister it
  /// when the Flutter Engine is detached from the Activity
  private lateinit var channel: MethodChannel
  private var objectDetectorHelper: ObjectDetectorHelper? = null
  private lateinit var context: Context

  override fun onAttachedToEngine(flutterPluginBinding: FlutterPlugin.FlutterPluginBinding) {
    context = flutterPluginBinding.applicationContext
    channel = MethodChannel(flutterPluginBinding.binaryMessenger, "tf_object_detector")
    channel.setMethodCallHandler(this)
  }

  override fun onMethodCall(call: MethodCall, result: Result) {
    Log.i("DETECTOR", call.method)
    when (call.method) {
      "init" -> {
        init(call, result)
      }
      "close" -> {
        close(call, result)
      }
      "setup" -> {
        setup(call, result)
      }
      "detect" -> {
        detect(call, result)
      }
      else ->
        result.notImplemented()
    }
  }

  private fun close(
    call: MethodCall,
    result: Result
  ) {
    try {
      objectDetectorHelper?.close()
      result.success(true)
    } catch (e: Exception) {
      result.error("Could not close detector", e.message, null)
    }
  }

  private fun init(
    call: MethodCall,
    result: Result
  ) {
    try {
      objectDetectorHelper = ObjectDetectorHelper(
        context = context
      )
      objectDetectorHelper?.init(result)
    } catch (e: Exception) {
      result.error("Could not init detector", e.message, null)
    }
  }

  private fun setup(
    call: MethodCall,
    result: Result
  ) {
    try {
      val model = call.argument<String>("model")
      val threshold: Double = call.argument<Double>("threshold") ?: 0.4
      val numThreads = call.argument<Int>("numThreads") ?: 2
      val maxResults = call.argument<Int>("maxResults") ?: 3
      val delegate = call.argument<Int>("delegate") ?: 0
      Log.i("DETECTOR", model ?: "EMPTY")
      objectDetectorHelper?.setupObjectDetector(
        model = model!!,
        threshold = threshold.toFloat(),
        numThreads = numThreads,
        maxResults = maxResults,
        currentDelegate = delegate
      )
      result.success(true)
    } catch (e: Exception) {
      result.error("Could not setup detector", e.message, null)
    }
  }

  private fun detect(
    call: MethodCall,
    result: Result
  ) {
    try {
      val bytesList = call.argument<List<ByteArray>>("bytesList")!!
      val imageHeight = call.argument<Int>("imageHeight")!!
      val imageWidth = call.argument<Int>("imageWidth")!!

      objectDetectorHelper?.detect(
        bytesList, imageHeight, imageWidth,
        imageRotation = 90,
        result
      )
    } catch (e: Exception) {
      Log.i("DETECTOR", e.stackTraceToString())
      result.error("Could not detect", e.message, null)
    }
  }

  override fun onDetachedFromEngine(binding: FlutterPlugin.FlutterPluginBinding) {
    channel.setMethodCallHandler(null)
  }

  override fun onAttachedToActivity(binding: ActivityPluginBinding) {
//    TODO("Not yet implemented")
  }

  override fun onDetachedFromActivityForConfigChanges() {
//    TODO("Not yet implemented")
  }

  override fun onReattachedToActivityForConfigChanges(binding: ActivityPluginBinding) {
//    TODO("Not yet implemented")
  }

  override fun onDetachedFromActivity() {
//    TODO("Not yet implemented")
  }
}
