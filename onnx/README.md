# Object Detection and Deep Pose Processing using ONNX Runtime
[Check out this section to learn more about object detection.](#object-detection-with-tiny\-YOLOv2/ONNX-Runtime) 
[Check out this section to learn about deep pose processing.](#deep-pose-processing-with-oNNX-runtime)

# Object Detection with Tiny-YOLOv2/ONNX Runtime

[Tiny-YOLOv2](https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/tiny-yolov2) is a real-time neural network for object detection that detects 20 different classes. It is made up of 9 convolutional layers and 6 max-pooling layers and is a smaller version of the more complex full [YOLOv2](https://pjreddie.com/darknet/yolov2/) network.

This sample demonstrates using ONNX Runtime to run Tiny-YOLOv2 against a image stream, and outputing the images annotated with bounding boxes.

## Getting Started

To run this sample, a camera will be required to be installed and ready to use on your system.

You can begin with the below launch file. It will bring up RViz tool where you can observe the interaction between `object_detection` and `cv_camera` nodes.

```Batchfile
ros2 launch onnx tracker.launch.py
```

## Subscribed Topics

  * `/image_raw` (sensor_msgs/Image)
    > The image stream that runs aginst object detection.

## Published Topics

  * `/image_debug_raw` (sensor_msgs/Image)
    > The image stream annotated with the bounding boxes to the detected objects.

  * `/visual_marker` (visualization_msgs/Marker)
    > The topic to publish the markers corresponding to the detected objected.

## Parameters

  * `~image_topic` (string, default: `image_raw`)
    > The topic name where the image stream comes from.

  * `~confidence` (float, default: `0.7`)
    > A threshold value to decide when to report an object detected.

  * `~label` (string, default: `person`)
    > The label to match when running object detection. 

  * `~link_name` (string, default: `camera`)
    > The frame to be associated with the image stream. 

  * `~visual_marker_topic` (string, default: `visual_marker`)
    > The topic to publish the visualization markers corresponding to the detected objects.

  * `~image_debug_topic` (string, default: `image_debug_raw`)
    > The topic name to publish the annotated image stream.

# Deep Pose Processing with ONNX Runtime

## Getting Started

To run this sample, a camera will be required to be installed and ready to use on your system.

You can begin with the below launch file.

```Batchfile
ros2 launch onnx pose.launch.py
```
TODO: Fill in.
