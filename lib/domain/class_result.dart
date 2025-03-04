import 'dart:ui';

class ClassResult {
  ClassResult({
    required this.rectResult,
    required this.label,
    required this.score,
  });

  ClassResult.fromJson(Map<String, dynamic> json)
      : this(
          rectResult: RectResult.fromJson(
            (json['rect'] as Map).cast<String, dynamic>(),
          ),
          score: json['score'] as double,
          label: json['label'] as String,
        );

  final RectResult rectResult;
  final double score;
  final String label;
}

class RectResult {
  RectResult({
    required this.top,
    required this.right,
    required this.bottom,
    required this.left,
  });

  RectResult.fromJson(Map<String, dynamic> json)
      : this(
          top: json['top'] as double,
          right: json['right'] as double,
          bottom: json['bottom'] as double,
          left: json['left'] as double,
        );

  final double top;
  final double right;
  final double bottom;
  final double left;

  Rect toRect(int imageWidth, int imageHeight, {int offset = 5}) {
    final left = imageWidth * this.left - offset;
    final top = imageHeight * this.top - offset;
    final right = imageWidth * this.right + offset;
    final bottom = imageHeight * this.bottom + offset;
    return Rect.fromLTRB(left, top, right, bottom);
  }
}
