import launch
import launch.actions
import launch.substitutions
import launch_ros.actions


def generate_launch_description():
    return launch.LaunchDescription([
        launch_ros.actions.Node(
            package='cv_camera', executable='cv_camera_node', output='screen',
            name=['cv_camera'],
            parameters=[
                {"rate": 5.0},
            ]),
        launch_ros.actions.Node(
            package='object_tracker', executable='object_tracker', output='screen',
            name=['object_tracker']),
    ])
