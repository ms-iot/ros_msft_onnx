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

::For running the samples, clone `ros_msft_camera` as well
git clone https://github.com/ms-iot/ros_msft_camera --recursive
```

There are two launch files included as samples in the launch folder. `tracker.launch` demonstrates tracking upto 20 classes icluding people in images/video and `pose.launch` demonstrates estimating the position and rotation of an engine block from images\video. To run the engine pose demo, copy the [Engine pose ONNX model](https://github.com/ms-iot/ros_msft_onnx_demo/releases/download/0.0/engine.onnx) to `ros_msft_onnx/testdata/`.

To use hardware accelleration, install [CUDA Toolkit 10.1](https://developer.nvidia.com/cuda-10.1-download-archive-base) and [cuDNN v7 for CUDA 10.1](https://developer.nvidia.com/rdp/cudnn-archive). 

For a project trained using customvision.ai at runtime you can change the parameters using rqt_reconfigure.

```Batchfile
rosrun rqt_reconfigure rqt_reconfigure
```
Place the relevant extracted onnx zip folder downloaded from customvision.ai to `ros_msft_onnx/testdata/` or a known location. Change the anchor values anch0, anch1,...., anch9 from the default values to 0.573, 0.677, 1.87, 2.06, 3.34, 5.47, 7.88, 3.53, 9.77, 9.17 and update any other relevant parameters below:

![Rqt Reconfigure](./ros_msft_onnx/testdata/rqt_reconfigure.PNG)

> While 'Pose' processing is enabled, the service required to generate the model has not been published as of October 2020

## Property Descriptions

| Property         | Description                                                                 |
| ---------------- | --------------------------------------------------------------------------- |
| onnx_model_path  | Path to the model.onnx file                                                 |
| confidence       | Minimum confidence before publishing an event. 0 to 1                       |
| tensor_width     | The Width of the input to the model.                                        |
| tensor_height    | The Height of the input to the model.                                       |
| tracker_type     | Currently enabled - `yolo` or `pose`.                                       |
| image_processing | `resize`, `scale` or `crop`                                                 |
| debug            | `true` or `false` determines if a debug image is published                  |
| image_topic      | The image topic to subscribe to                                             |
| label            | used to filter the found object to a specific label                         |
| mesh_rotation    | The orientation of the mesh when debug rendering pose                       |
| mesh_scale       | The scale of the mesh when debug rendering pose                             |
| mesh_resource    | The mesh used for debug rendering pose                                      |
| model_bounds     | 9 coordinates used to perform the point in perspective caluclation for pose |
| calibration      | Path to the OpenCV calibration file for point in persective                 |

## Building
Make sure to source your ROS version before building. Then use catkin_make to build.
```Batchfile
cd c:\workspace\src
catkin_make
```

## Running the samples
To run the samples, first source the workspace:
```Batchfile
cd c:\workspace\devel\
setup.bat
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
Onnx subscribes to the topic listed in the `image_topic` property, or `/camera/image_raw`

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
