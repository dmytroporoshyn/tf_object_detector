import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:tf_object_detector/domain/class_result.dart';

import 'tf_object_detector_platform_interface.dart';

/// An implementation of [TfObjectDetectorPlatform] that uses method channels.
class MethodChannelTfObjectDetector extends TfObjectDetectorPlatform {
  /// The method channel used to interact with the native platform.
  @visibleForTesting
  final methodChannel = const MethodChannel('tf_object_detector');

  @override
  Future<void> init() async {
    await methodChannel.invokeMethod<bool>('init');
  }

  @override
  Future<void> close() async {
    await methodChannel.invokeMethod<bool>('close');
  }

  @override
  Future<void> setup({
    required String modelPath,
    required List<String> labels,
    double threshold = 0.4,
    int numThreads = 2,
    int maxResults = 3,
    int delegate = 0,
  }) async {
    await methodChannel.invokeMethod<bool>('setup', {
      'model': modelPath,
      'threshold': threshold,
      'numThreads': numThreads,
      'maxResults': maxResults,
      'delegate': delegate,
      'labels': labels,
    });
  }

  @override
  Future<List<ClassResult>> detect({
    required List<Uint8List> bytesList,
    required int imageHeight,
    required int imageWidth,
  }) async {
    final detections =
        await methodChannel.invokeMethod<List<Object?>>('detect', {
      'bytesList': bytesList,
      'imageHeight': imageHeight,
      'imageWidth': imageWidth,
    });
    print(detections);
    final result = detections
            ?.map(
              (e) => ClassResult.fromJson(
                (e as Map).cast<String, dynamic>(),
              ),
            )
            .toList() ??
        [];
    return result;
  }
}
