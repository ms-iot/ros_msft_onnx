# Object Detection and Deep Pose Processing using ONNX Runtime
[Check out this section to learn more about object detection.](#object-detection-with-tiny\-YOLOv2/ONNX-Runtime) 

[Check out this section to learn about deep pose processing.](#deep-pose-processing-with-oNNX-runtime)

[Check out this section to learn about Parameters/Publishers/Subscriptions.](#Parameters-,Publishers,-and-Subscriptions)

# Object Detection with Tiny-YOLOv2/ONNX Runtime

[Tiny-YOLOv2](https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/tiny-yolov2) is a real-time neural network for object detection that detects 20 different classes. It is made up of 9 convolutional layers and 6 max-pooling layers and is a smaller version of the more complex full [YOLOv2](https://pjreddie.com/darknet/yolov2/) network.

This sample demonstrates using ONNX Runtime to run Tiny-YOLOv2 against a image stream, and outputing the images annotated with bounding boxes.

## Getting Started

To run this sample, a camera will be required to be installed and ready to use on your system.

You can begin with the below launch file. It will bring up RViz tool where you can observe the interaction between `object_detection` and `cv_camera` nodes.

```Batchfile
ros2 launch ros_msft_onnx tracker.launch.py
```

# Deep Pose Processing with ONNX Runtime

## Getting Started

To run this sample, a camera will be required to be installed and ready to use on your system.

You can begin with the below launch file.

```Batchfile
ros2 launch ros_msft_onnx pose.launch.py
```

# Parameters, Publishers, and Subscriptions

## Property Descriptions

| Property | Description |
|----------| ------------|
| onnx_model_path | Path to the model.onnx file | 
| confidence | Minimum confidence before publishing an event. 0 to 1 |
| tensor_width| The Width of the input to the model. |
| tensor_height| The Height of the input to the model. |
| tracker_type| Currently enabled - `yolo` or `pose`. |
| image_processing| `resize`, `scale` or `crop` |
| debug| `true` or `false` determines if a debug image is published |
| image_topic| The image topic to subscribe to |
| image_debug_topic | The topic name to publish the annotated image stream. |
| link_name | The frame to be associated with the image stream. |
| label | used to filter the found object to a specific label |
| mesh_rotation| The orientation of the mesh when debug rendering pose |
| mesh_scale| The scale of the mesh when debug rendering pose |
| mesh_resource| The mesh used for debug rendering pose |
| model_bounds| 9 coordinates used to perform the point in perspective caluclation for pose |
| calibration | Path to the OpenCV calibration file for point in persective |
| link_name | The frame to be associated with the image stream. |

## Subscriptions
Onnx subscribes to the topic listed in the `image_topic` property, or `/camera/image_raw`

## Publishing
Onnx Publishes the following topics:

### /image_debug_raw
The image stream annotated with the bounding boxes to the detected objects.

### /visual_markers
An array of `visualization_msgs::Marker` for found objects

### /detected_object
A single instance of the DetectedObjectPose message, which is output when tracker_type is set to pose.


