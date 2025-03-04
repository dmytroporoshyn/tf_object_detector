import 'dart:typed_data';

import 'package:plugin_platform_interface/plugin_platform_interface.dart';
import 'package:tf_object_detector/domain/class_result.dart';

import 'tf_object_detector_method_channel.dart';

abstract class TfObjectDetectorPlatform extends PlatformInterface {
  /// Constructs a TfObjectDetectorPlatform.
  TfObjectDetectorPlatform() : super(token: _token);

  static final Object _token = Object();

  static TfObjectDetectorPlatform _instance = MethodChannelTfObjectDetector();

  /// The default instance of [TfObjectDetectorPlatform] to use.
  ///
  /// Defaults to [MethodChannelTfObjectDetector].
  static TfObjectDetectorPlatform get instance => _instance;

  /// Platform-specific implementations should set this with their own
  /// platform-specific class that extends [TfObjectDetectorPlatform] when
  /// they register themselves.
  static set instance(TfObjectDetectorPlatform instance) {
    PlatformInterface.verifyToken(instance, _token);
    _instance = instance;
  }

  Future<void> init() {
    throw UnimplementedError('platformVersion() has not been implemented.');
  }

  Future<void> close() {
    throw UnimplementedError('platformVersion() has not been implemented.');
  }

  Future<void> setup({
    required String modelPath,
    required List<String> labels,
    double threshold = 0.4,
    int numThreads = 2,
    int maxResults = 3,
    int delegate = 0,
  }) {
    throw UnimplementedError('platformVersion() has not been implemented.');
  }

  Future<List<ClassResult>> detect({
    required List<Uint8List> bytesList,
    required int imageHeight,
    required int imageWidth,
  }) {
    throw UnimplementedError('platformVersion() has not been implemented.');
  }
}
