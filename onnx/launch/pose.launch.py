# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import launch
import launch.actions
import launch.substitutions
import launch_ros.actions
from launch.actions import DeclareLaunchArgument
from launch import conditions
import platform 

from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    share_dir = get_package_share_directory('ros_msft_onnx')
    engine_path = os.path.join(share_dir,
                             "data",
                             "Engine_Block.stl")    

    onnx_path = os.path.join(share_dir,
                             "data",
                             "engine.onnx")

    os_name = platform.system()
    os_flag = "false" if os_name == 'Windows' else "true" # default to Linux

    return launch.LaunchDescription([
        DeclareLaunchArgument(
            'onnx_model_path_arg', 
            default_value=onnx_path, 
            description="Onnx model path"),
        DeclareLaunchArgument(
            'model_bounds_arg', 
            default_value="", 
            description="Model bounds"),
        DeclareLaunchArgument(
            'tracker_type_arg', 
            default_value="pose", 
            description="Tracker type: choose between yolo or pose"),
        DeclareLaunchArgument(
            'mesh_resource_arg', 
            default_value=engine_path,
            description="Mesh resource arg"),
        launch_ros.actions.Node(
            package='ros_msft_onnx', executable='ros_msft_onnx', output='screen',
            name=['ros_msft_onnx'],
            parameters=[
                {'onnx_model_path': launch.substitutions.LaunchConfiguration('onnx_model_path_arg')},
                {'model_bounds': [-4.57, -10.50, -13.33, 5.54, -10.50, -13.33, 5.54, 0.00, -13.33, -4.57, 0.00, -13.33, -4.57, -10.50, 13.73, 5.54, -10.50, 13.73, 5.54, 0.00, 13.73, -4.57, 0.00, 13.73, 0.48, -5.25, 0.20]},
                {'tracker_type': launch.substitutions.LaunchConfiguration('tracker_type_arg')},
                {'mesh_resource': launch.substitutions.LaunchConfiguration('mesh_resource_arg')}, 
                {'debug': True},
                {'link_name': 'camera'},
                {'tracker_type': 'pose'}
            ]),
        launch_ros.actions.Node(
            package='cv_camera', executable='cv_camera_node', output='screen',
            name=['cv_camera'],
            parameters=[
                {'rate': 5.0},
                {'frame_id': 'camera'},
            ], 
            condition=conditions.IfCondition(os_flag)),
        launch_ros.actions.Node(
            package='win_camera', executable='win_camera_node', output='screen',
            name=['win_camera'],
            parameters=[
                {'frame_rate': 5.0},
                {'frame_id': 'camera'},
                {'camera_info_url': 'package://win_camera/camera_info/camera.yaml'},
            ], 
            condition=conditions.UnlessCondition(os_flag)),
        launch_ros.actions.Node(
            package='tf2_ros', executable='static_transform_publisher', output='screen',
            name=['static_transform_publisher'],
            arguments=[
                '0.1', '0.2', '0.3', '0.4', '.5', '.6', 'map', 'camera'
            ]),
    ])
