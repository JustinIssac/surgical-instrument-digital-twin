import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import glob
import re
from std_msgs.msg import String as StringMsg
import json

# filenames look like: instrument_dataset_2_frame169.png
FRAME_RE = re.compile(r'(instrument_dataset_\d+)_frame(\d+)')


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
        # R6: looping is right for a demo but corrupts evaluation, which
        # must see each frame exactly once.
        self.declare_parameter('loop', True)
        # Restrict playback to one surgical sequence. Tracking across
        # concatenated procedures is meaningless: the entire scene changes.
        self.declare_parameter('sequence_filter', 'instrument_dataset_8')

        frames_path       = self.get_parameter('frames_path').value
        right_frames_path = self.get_parameter('right_frames_path').value
        fps               = self.get_parameter('fps').value
        self.loop         = bool(self.get_parameter('loop').value)
        seqf              = self.get_parameter('sequence_filter').value
        self.passes       = 0

        self.left_pub  = self.create_publisher(
            Image, '/camera/image_raw', 10
        )
        self.right_pub = self.create_publisher(
            Image, '/camera/right/image_raw', 10
        )

        # source identity: lets downstream nodes detect sequence changes,
        # which a monotonic message counter cannot express
        self.src_pub = self.create_publisher(StringMsg, '/frame_source', 10)
        self.bridge        = CvBridge()
        self.left_frames   = sorted(glob.glob(
            os.path.join(frames_path, '*.png')
        ))
        self.right_frames  = sorted(glob.glob(
            os.path.join(right_frames_path, '*.png')
        ))
        if seqf:
            self.left_frames  = [f for f in self.left_frames  if seqf in f]
            self.right_frames = [f for f in self.right_frames if seqf in f]
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

        m = FRAME_RE.match(os.path.basename(left_path))
        src = StringMsg()
        src.data = json.dumps({
            'seq':        m.group(1) if m else 'unknown',
            'src_frame':  int(m.group(2)) if m else self.index,
            'index':      self.index,
            'total':      len(self.left_frames),
            'stamp_sec':  stamp.sec,
            'stamp_ns':   stamp.nanosec})
        self.src_pub.publish(src)

        self.get_logger().info(
            f'Published frame {self.index + 1}/'
            f'{len(self.left_frames)}: '
            f'{os.path.basename(left_path)}'
        )

        self.index += 1
        if self.index >= len(self.left_frames):
            self.passes += 1
            if self.loop:
                self.index = 0
                self.get_logger().info(f'--- loop {self.passes} complete ---')
            else:
                self.index = len(self.left_frames) - 1
                self.timer.cancel()
                self.get_logger().info(
                    f'single pass complete: {len(self.left_frames)} frames. '
                    f'Publisher idle.')


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
