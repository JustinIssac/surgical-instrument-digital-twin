from setuptools import setup

package_name = 'surgical_perception'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Justin Varghese Issac',
    maintainer_email='justin@todo.com',
    description='Surgical instrument perception node using YOLOv11',
    license='MIT',
    entry_points={
        'console_scripts': [
            'perception_node  = surgical_perception.perception_node:main',
            'video_publisher  = surgical_perception.video_publisher:main',
            'pose_estimator   = surgical_perception.pose_estimator:main',
            'twin_sync_node   = surgical_perception.twin_sync_node:main',
            'rviz_markers     = surgical_perception.rviz_markers:main',
            'stereo_depth_node = surgical_perception.stereo_depth_node:main',
        ],
    },
)
