import 'dart:typed_data';

import 'package:tf_object_detector/domain/class_result.dart';

import 'tf_object_detector_platform_interface.dart';

class TfObjectDetector {
  Future<void> init() {
    return TfObjectDetectorPlatform.instance.init();
  }

  Future<void> close() {
    return TfObjectDetectorPlatform.instance.close();
  }

  Future<void> setup({
    required String modelPath,
    required List<String> labels,
    double threshold = 0.4,
    int numThreads = 2,
    int maxResults = 3,
    int delegate = 0,
  }) {
    return TfObjectDetectorPlatform.instance.setup(
      modelPath: modelPath,
      maxResults: maxResults,
      numThreads: numThreads,
      threshold: threshold,
      delegate: delegate,
      labels: labels,
    );
  }

  Future<List<ClassResult>> detect({
    required List<Uint8List> bytesList,
    required int imageHeight,
    required int imageWidth,
  }) {
    return TfObjectDetectorPlatform.instance.detect(
      bytesList: bytesList,
      imageHeight: imageHeight,
      imageWidth: imageWidth,
    );
  }
}
