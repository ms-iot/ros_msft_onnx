# ONNX (Open Neural Network Exchange) ROS Node

## Consuming Onnx
Requirements:

* Install Visual Studio 2019 with UWP development
* ROS Noetic for Windows

The Onnx ROS Node is distrubted as source. To consume it in your robot, clone the ros_msft_onnx sources into your workspace.

For example:

```Batchfile
mkdir c:\workspace\onnx_demo\src
cd c:\workspace\onnx_demo\src
git clone https://github.com/ms-iot/ros_msft_onnx -b noetic-devel

#For running the samples, clone cv_camera as well
git clone https://github.com/ms-iot/cv_camera
```

There are two launch files included as samples in the launch folder. `tracker.launch` demonstrates tracking people in images/video and `pose.launch` demonstrates estimating the position and rotation of an engine block from images\video. To run the engine pose demo, copy the [Engine pose ONNX model](https://github.com/ms-iot/ros_msft_onnx_demo/releases/download/0.0/engine.onnx) to `ros_msft_onnx/testdata/`.

To use hardware accelleration, install [CUDA Toolkit 10.1](https://developer.nvidia.com/cuda-10.1-download-archive-base) and [cuDNN v7 for CUDA 10.1](https://developer.nvidia.com/rdp/cudnn-archive). 

For your own project, you can create a launch file in the following format:

```xml
<launch>
  <arg name="onnx_model_path_arg" default="$(find ros_msft_onnx)/testdata/model.onnx"/>
  <node pkg="ros_msft_onnx" type="ros_msft_onnx_node" name="ros_msft_onnx" output="screen">
    <param name="onnx_model_path" value="$(arg onnx_model_path_arg)"/>
    <param name="confidence" value="0.5"/>
    <param name="tensor_width" value="416"/>
    <param name="tensor_height" value="416"/>
    <param name="tracker_type" value="yolo"/>
    <param name="image_processing" value="resize"/>
    <param name="debug" value="true"/>
    <param name="image_topic" value="/cv_camera/image_raw" />
  </node>
  
  <!-- NOTE: The image properties need to be valid for the camera, or the node will auto select the closest values -->
  <node pkg="cv_camera" type="cv_camera_node" name="cv_camera" output="screen">
    <param name="rate" type="double" value="5.0"/>
    <param name="image_width" type="double" value="640"/>
    <param name="image_height" type="double" value="480"/>
  </node>

  <node pkg="tf" type="static_transform_publisher" name="onnx_link"
    args="0 -0.02  0 0 0 0 map base_link 100" />  

</launch>
```

> While 'Pose' processing is enabled, the service required to generate the model has not been published as of October 2020

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
| label | used to filter the found object to a specific label |
| mesh_rotation| The orientation of the mesh when debug rendering pose |
| mesh_scale| The scale of the mesh when debug rendering pose |
| mesh_resource| The mesh used for debug rendering pose |
| model_bounds| 9 coordinates used to perform the point in perspective caluclation for pose |
| calibration | Path to the OpenCV calibration file for point in persective |

## Building
Make sure to source your ROS version before building. Then use catkin_make_isolated to build.
```Batchfile
cd c:\workspace\onnx_demo
catkin_make_isolated
```

## Running the samples
To run the samples, first source the workspace:
```Batchfile
install_isolated\setup.bat
```

Then, for the tracker sample run:
```Batchfile
roslaunch ros_msft_onnx tracker.launch
```

For the engine pose sample run:
```Batchfile
roslaunch ros_msft_onnx pose.launch
```

## Subscriptions
Onnx subscribes to the topic listed in the `image_topic` property, or `/cv_camera/image_raw`

## Publishing
Onnx Publishes the following topics:

### /tracked_objects/image
Outputs an image with highlighing if the debug property is set

### /tracked_objects/
An array of `visualization_msgs::Marker` for found objects

### /detected_object
A single instance of the DetectedObjectPose message, which is output when tracker_type is set to pose.

# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
