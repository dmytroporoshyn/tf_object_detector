#import "TfObjectDetectorPlugin.h"

#import "TensorFlowLiteC/TensorFlowLiteC.h"

#include <pthread.h>
#include <unistd.h>
#include <fstream>
#include <iostream>
#include <queue>
#include <sstream>
#include <string>

#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdio.h>
#include <vector>

typedef void (^TfLiteStatusCallback)(TfLiteStatus);
bool setup(NSDictionary* args);
void runDetection(NSDictionary* args, TfLiteStatusCallback cb);
void detect(NSDictionary* args, FlutterResult result);
void close();

@implementation TfObjectDetectorPlugin
+ (void)registerWithRegistrar:(NSObject <FlutterPluginRegistrar> *)registrar {
    FlutterMethodChannel *channel = [FlutterMethodChannel
                                     methodChannelWithName:@"tf_object_detector"
                                     binaryMessenger:[registrar messenger]];
    TfObjectDetectorPlugin *instance = [[TfObjectDetectorPlugin alloc] init];
    [registrar addMethodCallDelegate:instance channel:channel];
}

- (void)handleMethodCall:(FlutterMethodCall *)call result:(FlutterResult)result {
    if ([@"init" isEqualToString:call.method]) {
        bool result_value = false;
        result(@(result_value));
    } else if ([@"setup" isEqualToString:call.method]) {
        bool load_result = setup(call.arguments);
        result(@(load_result));
    } else if ([@"detect" isEqualToString:call.method]) {
        detect(call.arguments, result);
    } else if ([@"close" isEqualToString:call.method]) {
        close();
        result(@(true));
    } else {
        result(FlutterMethodNotImplemented);
    }
}

@end

NSArray* labels;
TfLiteInterpreter *interpreter = nullptr;
TfLiteModel *model = nullptr;
TfLiteDelegate *delegate = nullptr;
bool interpreter_busy = false;
float threshold = 0.4;

bool setup(NSDictionary* args) {
    NSString* model_path = args[@"model"];
    labels = args[@"labels"];
    threshold = [args[@"threshold"] floatValue];
    const int num_threads = [args[@"numThreads"] intValue];
    
    
    TfLiteInterpreterOptions *options = nullptr;
    model = TfLiteModelCreateFromFile(model_path.UTF8String);
    if (!model) {
        return false;
    }
    options = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreterOptionsSetNumThreads(options, num_threads);
    
    bool useGpuDelegate = [args[@"useGpuDelegate"] boolValue];
    
    interpreter = TfLiteInterpreterCreate(model, options);
    if (!interpreter) {
        return false;
    }
    
    if (TfLiteInterpreterAllocateTensors(interpreter) != kTfLiteOk) {
        return false;
    }
    return true;
}

void feedInputTensor(uint8_t* in, int* input_size, int image_height, int image_width, int image_channels, float input_mean, float input_std) {
    
    assert(TfLiteInterpreterGetInputTensorCount(interpreter) == 1);
    TfLiteTensor* input_tensor = TfLiteInterpreterGetInputTensor(interpreter, 0);
    
    const int input_channels = input_tensor->dims->data[3];
    const int width = input_tensor->dims->data[2];
    const int height = input_tensor->dims->data[1];
    *input_size = width;
    
    if (input_tensor->type == kTfLiteUInt8) {
        
        uint8_t* out = input_tensor->data.uint8;
        
        for (int y = 0; y < height; ++y) {
            const int in_y = (y * image_height) / height;
            uint8_t* in_row = in + (in_y * image_width * image_channels);
            uint8_t* out_row = out + (y * width * input_channels);
            for (int x = 0; x < width; ++x) {
                const int in_x = (x * image_width) / width;
                uint8_t* in_pixel = in_row + (in_x * image_channels);
                uint8_t* out_pixel = out_row + (x * input_channels);
                for (int c = 0; c < input_channels; ++c) {
                    out_pixel[c] = in_pixel[c];
                }
            }
        }
    } else { // kTfLiteFloat32
        
        float* out = input_tensor->data.f;
        
        for (int y = 0; y < height; ++y) {
            const int in_y = (y * image_height) / height;
            uint8_t* in_row = in + (in_y * image_width * image_channels);
            float* out_row = out + (y * width * input_channels);
            for (int x = 0; x < width; ++x) {
                const int in_x = (x * image_width) / width;
                uint8_t* in_pixel = in_row + (in_x * image_channels);
                float* out_pixel = out_row + (x * input_channels);
                for (int c = 0; c < input_channels; ++c) {
                    out_pixel[c] = (in_pixel[c] - input_mean) / input_std;
                }
            }
        }
    }
}

void feedInputTensorFrame(const FlutterStandardTypedData* typedData, int* input_size,
                          int image_height, int image_width, int image_channels, float input_mean, float input_std) {
    uint8_t* in = (uint8_t*)[[typedData data] bytes];
    feedInputTensor(in, input_size, image_height, image_width, image_channels, input_mean, input_std);
}

NSMutableArray* parseResults(int num_results_per_class) {

  assert(TfLiteInterpreterGetOutputTensorCount(interpreter) == 4);

  NSMutableArray* results = [NSMutableArray array];

  float* output_locations = TfLiteInterpreterGetOutputTensor(interpreter, 0)->data.f;
  float* output_classes = TfLiteInterpreterGetOutputTensor(interpreter, 1)->data.f;
  float* output_scores = TfLiteInterpreterGetOutputTensor(interpreter, 2)->data.f;
  float* num_detections = TfLiteInterpreterGetOutputTensor(interpreter, 3)->data.f;


  NSMutableDictionary* counters = [NSMutableDictionary dictionary];
  for (int d = 0; d < *num_detections; d++)
  {
    const int detected_class = output_classes[d];
    float score = output_scores[d];
    
    if (score < threshold) continue;
    
    NSMutableDictionary* res = [NSMutableDictionary dictionary];
    NSString* class_name = labels[detected_class];
    NSObject* counter = [counters objectForKey:class_name];
    
    if (counter) {
      int countValue = [(NSNumber*)counter intValue] + 1;
      if (countValue > num_results_per_class) {
        continue;
      }
      [counters setObject:@(countValue) forKey:class_name];
    } else {
      [counters setObject:@(1) forKey:class_name];
    }
    
    [res setObject:@(score) forKey:@"score"];
    [res setObject:class_name forKey:@"label"];
    
    const float ymin = fmax(0, output_locations[d * 4]);
    const float xmin = fmax(0, output_locations[d * 4 + 1]);
    const float ymax = output_locations[d * 4 + 2];
    const float xmax = output_locations[d * 4 + 3];

    NSMutableDictionary* rect = [NSMutableDictionary dictionary];
    [rect setObject:@(xmin) forKey:@"left"];
    [rect setObject:@(ymin) forKey:@"top"];
      
    [rect setObject:@(xmax) forKey:@"right"];
    [rect setObject:@(ymax) forKey:@"bottom"];
    
    [res setObject:rect forKey:@"rect"];
    [results addObject:res];
  }
  return results;
}

void detect(NSDictionary* args, FlutterResult result) {
    const FlutterStandardTypedData* typedData = args[@"bytesList"][0];
    const NSString* model = args[@"model"];
    const int image_height = [args[@"imageHeight"] intValue];
    const int image_width = [args[@"imageWidth"] intValue];
    const float input_mean = [args[@"imageMean"] floatValue];
    const float input_std = [args[@"imageStd"] floatValue];
    const float threshold = [args[@"threshold"] floatValue];
    const int num_results_per_class = [args[@"numResultsPerClass"] intValue];
    
    NSMutableArray* empty = [@[] mutableCopy];
    
    if (!interpreter || interpreter_busy) {
        NSLog(@"Failed to construct interpreter or busy.");
        return result(empty);
    }
    
    int input_size;
    int image_channels = 4;
    
    uint8_t* in = (uint8_t*)[[typedData data] bytes];
    feedInputTensor(in, &input_size, image_height, image_width, image_channels, input_mean, input_std);
    
    runDetection(args, ^(TfLiteStatus status) {
        if (status != kTfLiteOk) {
            NSLog(@"Failed to invoke!");
            return result(empty);
        }
        return result(parseResults(num_results_per_class));;
    });
}

void runDetection(NSDictionary* args, TfLiteStatusCallback cb) {
    const bool asynch = [args[@"asynch"] boolValue];
    if (asynch) {
        interpreter_busy = true;
        dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^(void){
            
            TfLiteStatus status = TfLiteInterpreterInvoke(interpreter);
            
            dispatch_async(dispatch_get_main_queue(), ^(void){
                interpreter_busy = false;
                cb(status);
            });
        });
    } else {
        
        TfLiteStatus status = TfLiteInterpreterInvoke(interpreter);
        
        cb(status);
    }
}

void close() {
  interpreter = nullptr;
  delegate = nullptr;
  model = NULL;
  labels = nullptr;
}
