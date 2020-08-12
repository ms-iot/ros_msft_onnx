# Object Detection with Tiny-YOLOv2/ONNX Runtime

Tiny-YOLOv2 is a real-time neural network for object detection that detects 20 different classes. It is made up of 9 convolutional layers and 6 max-pooling layers and is a smaller version of the more complex full YOLOv2 network.

This package demonstrates using ONNX Runtime to run Tiny-YOLOv2 against a image stream, and outputing the images annotated with bounding boxes.

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

  * `~confidence_threshold` (float, default: `0.7`)
    > A threshold value to decide when to report an object detected.

  * `~label` (string, default: `person`)
    > The label to match when running object detection. 

  * `~frame_id` (string, default: `camera`)
    > The frame to be associated with the image stream. 

  * `~visual_marker_topic` (string, default: `visual_marker`)
    > The topic to publish the visualization markers corresponding to the detected objects.

  * `~image_debug_topic` (string, default: `image_debug_raw`)
    > The topic name to publish the annotated image stream.