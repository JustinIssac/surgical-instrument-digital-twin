import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import glob


class VideoPublisher(Node):
    def __init__(self):
        super().__init__('video_publisher')
        self.declare_parameter(
            'frames_path',
            '/home/inoruske/surgical_twin_ws/demo_data'
        )
        self.declare_parameter(
            'right_frames_path',
            '/home/inoruske/surgical_twin_ws/demo_data_right'
        )
        self.declare_parameter('fps', 10.0)

        frames_path       = self.get_parameter('frames_path').value
        right_frames_path = self.get_parameter('right_frames_path').value
        fps               = self.get_parameter('fps').value

        self.left_pub  = self.create_publisher(
            Image, '/camera/image_raw', 10
        )
        self.right_pub = self.create_publisher(
            Image, '/camera/right/image_raw', 10
        )

        self.bridge        = CvBridge()
        self.left_frames   = sorted(glob.glob(
            os.path.join(frames_path, '*.png')
        ))
        self.right_frames  = sorted(glob.glob(
            os.path.join(right_frames_path, '*.png')
        ))
        self.index         = 0
        self.timer         = self.create_timer(1.0 / fps, self.publish_frame)

        self.get_logger().info(
            f'Video publisher ready: '
            f'{len(self.left_frames)} left, '
            f'{len(self.right_frames)} right frames at {fps} FPS'
        )

    def publish_frame(self):
        stamp = self.get_clock().now().to_msg()
        if not self.left_frames:
            self.get_logger().error('No frames found!')
            return

        # Publish left frame
        left_path  = self.left_frames[self.index]
        left_frame = cv2.imread(left_path)
        if left_frame is not None:
            msg                 = self.bridge.cv2_to_imgmsg(
                left_frame, encoding='bgr8'
            )
            msg.header.stamp    = stamp
            msg.header.frame_id = 'camera_left'
            self.left_pub.publish(msg)

        # Publish right frame if available
        if self.index < len(self.right_frames):
            right_path  = self.right_frames[self.index]
            right_frame = cv2.imread(right_path)
            if right_frame is not None:
                msg                 = self.bridge.cv2_to_imgmsg(
                    right_frame, encoding='bgr8'
                )
                msg.header.stamp    = stamp
                msg.header.frame_id = 'camera_right'
                self.right_pub.publish(msg)

        self.get_logger().info(
            f'Published frame {self.index + 1}/'
            f'{len(self.left_frames)}: '
            f'{os.path.basename(left_path)}'
        )

        self.index = (self.index + 1) % len(self.left_frames)


def main(args=None):
    rclpy.init(args=args)
    node = VideoPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Video publisher shutting down...')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
