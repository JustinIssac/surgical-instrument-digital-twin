"""
Frame source for the surgical perception pipeline.

Accepts, and auto-detects between:
  * stereo PNG folders  (left + right directories)   -> full 3D
  * mono PNG folder                                   -> depth from mono model
  * stereo video        (side-by-side in one file)    -> full 3D
  * mono video          (mp4/avi/mov/mkv)             -> depth from mono model

Publishes /input_capability (transient-local, so late subscribers still
receive it) describing what the input actually supports. Downstream nodes
use this rather than assuming stereo and silently substituting a constant
depth, which produces a confident but fabricated 3D twin.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import os, re, glob, json
from pathlib import Path

FRAME_RE   = re.compile(r'(instrument_dataset_\d+)_frame(\d+)')
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.m4v', '.webm'}
IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

# Calibration is valid only for the EndoVis rig at this resolution.
CALIB_RESOLUTION = (1920, 1080)


class VideoPublisher(Node):
    def __init__(self):
        super().__init__('video_publisher')

        self.declare_parameter('source', '/home/inoruske/surgical_twin_ws/demo_data')
        self.declare_parameter('source_right', '/home/inoruske/surgical_twin_ws/demo_data_right')
        self.declare_parameter('fps', 10.0)
        self.declare_parameter('loop', True)
        self.declare_parameter('sequence_filter', 'instrument_dataset_8')
        # 'auto' | 'stereo_pair' | 'mono' | 'sbs'  (sbs = side-by-side video)
        self.declare_parameter('stereo_mode', 'auto')
        self.declare_parameter('start_frame', 0)

        src        = str(self.get_parameter('source').value)
        src_r      = str(self.get_parameter('source_right').value)
        fps        = float(self.get_parameter('fps').value)
        self.loop  = bool(self.get_parameter('loop').value)
        seqf       = self.get_parameter('sequence_filter').value
        mode       = str(self.get_parameter('stereo_mode').value)

        self.bridge = CvBridge()
        self.cap = self.cap_r = None
        self.frames = self.frames_r = []
        self.sbs = False
        self.index = int(self.get_parameter('start_frame').value)

        self.kind = self._open_source(src, src_r, seqf, mode)
        if self.kind is None:
            self.get_logger().error(f'Could not read any frames from: {src}')
            raise SystemExit(1)

        probe = self._probe_first_frame()
        if probe is None:
            self.get_logger().error('Source opened but first frame unreadable')
            raise SystemExit(1)
        h, w = probe.shape[:2]
        self.width, self.height = w, h

        self.has_stereo   = self.kind in ('stereo_pair', 'sbs')
        self.calib_valid  = (w, h) == CALIB_RESOLUTION
        self.depth_mode   = ('stereo' if (self.has_stereo and self.calib_valid)
                             else 'monocular')

        qos_latched = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE)

        self.left_pub  = self.create_publisher(Image, '/camera/image_raw', 10)
        self.right_pub = self.create_publisher(Image, '/camera/right/image_raw', 10)
        self.src_pub   = self.create_publisher(String, '/frame_source', 10)
        self.cap_pub   = self.create_publisher(String, '/input_capability', qos_latched)

        cap_msg = String()
        cap_msg.data = json.dumps({
            'source':         src,
            'kind':           self.kind,
            'width':          w,
            'height':         h,
            'has_stereo':     self.has_stereo,
            'calib_valid':    self.calib_valid,
            'depth_mode':     self.depth_mode,
            'total_frames':   self.total,
            'fps':            fps,
        })
        self.cap_pub.publish(cap_msg)

        self.get_logger().info(
            f'\n  source      : {src}'
            f'\n  kind        : {self.kind}'
            f'\n  resolution  : {w}x{h}'
            f'\n  frames      : {self.total}'
            f'\n  stereo      : {self.has_stereo}'
            f'\n  calibration : {"matches" if self.calib_valid else "MISMATCH -> monocular"}'
            f'\n  depth mode  : {self.depth_mode.upper()}'
            f'\n  loop        : {self.loop}')

        self.timer = self.create_timer(1.0 / fps, self.publish_frame)

    # ---------------------------------------------------------------
    def _open_source(self, src, src_r, seqf, mode):
        p = Path(src)

        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            self.cap = cv2.VideoCapture(str(p))
            if not self.cap.isOpened():
                return None
            self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # a 2:1 or wider frame is very likely side-by-side stereo
            if mode == 'sbs' or (mode == 'auto' and h and w / h >= 1.9):
                self.sbs = True
                return 'sbs'
            # a second video file may sit alongside
            pr = Path(src_r)
            if pr.is_file() and pr.suffix.lower() in VIDEO_EXTS:
                self.cap_r = cv2.VideoCapture(str(pr))
                if self.cap_r.isOpened():
                    return 'stereo_pair'
            return 'mono'

        if p.is_dir():
            self.frames = sorted(
                f for f in glob.glob(str(p / '*'))
                if Path(f).suffix.lower() in IMAGE_EXTS)
            if seqf:
                self.frames = [f for f in self.frames if seqf in f]
            if not self.frames:
                return None
            pr = Path(src_r)
            if mode != 'mono' and pr.is_dir():
                self.frames_r = sorted(
                    f for f in glob.glob(str(pr / '*'))
                    if Path(f).suffix.lower() in IMAGE_EXTS)
                if seqf:
                    self.frames_r = [f for f in self.frames_r if seqf in f]
            self.total = len(self.frames)
            if self.frames_r and len(self.frames_r) == len(self.frames):
                return 'stereo_pair'
            if self.frames_r:
                self.get_logger().warn(
                    f'left/right frame counts differ '
                    f'({len(self.frames)} vs {len(self.frames_r)}) '
                    f'-> treating as monocular')
                self.frames_r = []
            return 'mono'

        return None

    def _probe_first_frame(self):
        if self.cap is not None:
            ok, f = self.cap.read()
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            if not ok:
                return None
            return f[:, :f.shape[1] // 2] if self.sbs else f
        return cv2.imread(self.frames[0]) if self.frames else None

    # ---------------------------------------------------------------
    def _next_pair(self):
        """Return (left, right|None, label) or None when the source ends."""
        if self.cap is not None:
            ok, f = self.cap.read()
            if not ok:
                return None
            if self.sbs:
                m = f.shape[1] // 2
                return f[:, :m], f[:, m:], f'frame{self.index:06d}'
            r = None
            if self.cap_r is not None:
                ok_r, r = self.cap_r.read()
                if not ok_r:
                    r = None
            return f, r, f'frame{self.index:06d}'

        if self.index >= len(self.frames):
            return None
        lp = self.frames[self.index]
        l  = cv2.imread(lp)
        r  = (cv2.imread(self.frames_r[self.index])
              if self.index < len(self.frames_r) else None)
        return l, r, os.path.basename(lp)

    def _rewind(self):
        self.index = 0
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            if self.cap_r is not None:
                self.cap_r.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # ---------------------------------------------------------------
    def publish_frame(self):
        got = self._next_pair()
        if got is None:
            if self.loop:
                self._rewind()
                got = self._next_pair()
                if got is None:
                    self.timer.cancel(); return
            else:
                self.timer.cancel()
                self.get_logger().info(
                    f'single pass complete: {self.index} frames. Publisher idle.')
                return

        left, right, label = got
        if left is None:
            self.index += 1
            return

        stamp = self.get_clock().now().to_msg()

        m = self.bridge.cv2_to_imgmsg(left, encoding='bgr8')
        m.header.stamp = stamp; m.header.frame_id = 'camera_left'
        self.left_pub.publish(m)

        if right is not None:
            mr = self.bridge.cv2_to_imgmsg(right, encoding='bgr8')
            mr.header.stamp = stamp; mr.header.frame_id = 'camera_right'
            self.right_pub.publish(mr)

        fm  = FRAME_RE.match(label)
        src = String()
        src.data = json.dumps({
            'seq':       fm.group(1) if fm else Path(str(self.get_parameter('source').value)).stem,
            'src_frame': int(fm.group(2)) if fm else self.index,
            'index':     self.index,
            'total':     self.total,
            'label':     label,
        })
        self.src_pub.publish(src)

        if self.index % 25 == 0:
            self.get_logger().info(
                f'frame {self.index + 1}/{self.total or "?"}: {label}')
        self.index += 1


def main(args=None):
    rclpy.init(args=args)
    try:
        node = VideoPublisher()
    except SystemExit:
        rclpy.shutdown(); return
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.cap:   node.cap.release()
        if node.cap_r: node.cap_r.release()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
