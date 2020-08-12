import os
import launch
import launch.actions
import launch.substitutions
import launch_ros.actions

from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    share_dir = get_package_share_directory('onnx_object_detection')
    rviz_default_view = os.path.join(share_dir, 'rviz', 'default_view.rviz')

    return launch.LaunchDescription([
        launch_ros.actions.Node(
            package='onnx_object_detection', executable='object_detection', output='screen',
            name=['object_detection'],
            parameters=[
                {'frame_id': 'camera'},
            ]),
        launch_ros.actions.Node(
            package='cv_camera', executable='cv_camera_node', output='screen',
            name=['cv_camera'],
            parameters=[
                {'rate': 5.0},
                {'frame_id': 'camera'},
            ]),
        launch_ros.actions.Node(
            package='tf2_ros', executable='static_transform_publisher', output='screen',
            name=['static_transform_publisher'],
            arguments=[
                '0.1', '0.2', '0.3', '0.4', '.5', '.6', 'map', 'camera'
            ]),
        launch_ros.actions.Node(
            package='rviz2', executable='rviz2', output='screen',
            name=['rviz2'],
            arguments=[
                '-d', rviz_default_view
            ]),
    ])
