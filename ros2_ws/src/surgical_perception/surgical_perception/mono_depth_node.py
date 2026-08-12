"""
Monocular metric depth for inputs without usable stereo.

Depth Anything V2 Small predicts INVERSE RELATIVE depth (unitless). Metric
depth is recovered by a two-parameter fit against SGBM stereo:

        1/Z_metres = a * d_rel + b

fitted on 298 paired samples from EndoVis dataset 8.

Validity is range-limited by measurement, not by assumption. Seven model
families (linear, quadratic, cubic, power-law, piecewise, isotonic) were
compared under 5-fold cross-validation; all underestimated by 23-32 mm
beyond 95 mm, indicating the depth network does not resolve that range in
these scenes rather than the calibration being misspecified. Estimates
beyond MAX_VALID_M are therefore reported as unreliable and excluded.

Measured accuracy within range: 3.6 mm median, +4.1 mm bias (50-65 mm),
+1.5 mm bias (65-80 mm).
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2, json, yaml
import numpy as np
import torch
from pathlib import Path

CALIB = "/home/inoruske/surgical_twin_ws/config/mono_depth_calib.yaml"
MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
INPUT_SIZE   = 518
MIN_VALID_M  = 0.030
MAX_VALID_M  = 0.085     # beyond this all fitted models fail (see docstring)


class MonoDepthNode(Node):
    def __init__(self):
        super().__init__('mono_depth_node')

        self.declare_parameter('calib_path', CALIB)
        self.declare_parameter('enabled_when', 'monocular')  # monocular|always
        self.declare_parameter('max_valid_m', MAX_VALID_M)
        self.declare_parameter('device', 'cuda')

        self.max_valid = float(self.get_parameter('max_valid_m').value)
        self.mode_req  = str(self.get_parameter('enabled_when').value)
        device         = str(self.get_parameter('device').value)

        c = yaml.safe_load(open(self.get_parameter('calib_path').value))
        self.a, self.b = float(c['a']), float(c['b'])

        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        self.proc  = AutoImageProcessor.from_pretrained(MODEL)
        self.model = AutoModelForDepthEstimation.from_pretrained(MODEL).to(device).eval()
        self.device = device

        self.bridge     = CvBridge()
        self.depth_map  = None       # metric, full frame
        self.rel_map    = None
        self.active     = False      # only runs when stereo is unavailable
        self.capability = None
        self.stats      = {'ok': 0, 'far': 0, 'near': 0, 'nomap': 0}
        self.last_ms    = 0.0

        qos = QoSProfile(depth=1,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(String, '/input_capability',
                                 self.capability_callback, qos)
        self.create_subscription(Image, '/camera/image_raw',
                                 self.image_callback, 5)
        self.create_subscription(String, '/instrument_detections',
                                 self.detection_callback, 10)

        self.pub      = self.create_publisher(String, '/instrument_detections_3d', 10)
        self.dbg_pub  = self.create_publisher(Image, '/mono_depth_image', 5)

        self.get_logger().info(
            f'Mono depth ready (idle until needed)\n'
            f'  1/Z = {self.a:.5f}*d_rel + {self.b:.5f}\n'
            f'  valid range {MIN_VALID_M*1000:.0f}-{self.max_valid*1000:.0f} mm '
            f'(3.6 mm median error within range)')

    # ----------------------------------------------------------------
    def capability_callback(self, msg):
        try:
            cap = json.loads(msg.data)
        except Exception:
            return
        self.capability = cap
        want = (cap.get('depth_mode') != 'stereo')
        self.active = True if self.mode_req == 'always' else want
        self.get_logger().info(
            f"input depth_mode={cap.get('depth_mode')} -> "
            f"mono depth {'ACTIVE' if self.active else 'idle (stereo in use)'}")

    def image_callback(self, msg):
        if not self.active:
            return
        import time
        t0 = time.monotonic()
        img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(cv2.resize(img, (INPUT_SIZE, INPUT_SIZE)),
                           cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            out = self.model(**self.proc(images=rgb, return_tensors='pt')
                             .to(self.device))
        rel = out.predicted_depth[0].cpu().numpy().astype(np.float32)
        rel = cv2.resize(rel, (w, h), interpolation=cv2.INTER_LINEAR)

        inv = self.a * rel + self.b
        with np.errstate(divide='ignore', invalid='ignore'):
            Z = np.where(inv > 1e-6, 1.0 / inv, np.nan)
        self.rel_map, self.depth_map = rel, Z
        self.last_ms = (time.monotonic() - t0) * 1000

        vis = np.clip((Z - MIN_VALID_M) / (self.max_valid - MIN_VALID_M), 0, 1)
        vis = (255 * (1 - np.nan_to_num(vis))).astype(np.uint8)
        self.dbg_pub.publish(self.bridge.cv2_to_imgmsg(
            cv2.applyColorMap(vis, cv2.COLORMAP_TURBO), 'bgr8'))

    # ----------------------------------------------------------------
    def detection_callback(self, msg):
        if not self.active:
            return                       # stereo node owns this topic
        data = json.loads(msg.data)
        out  = []

        for det in data.get('detections', []):
            u, v = det.get('tip_px', det['centroid_px'])
            z, method, reason = None, 'unavailable', 'no_depth_map'

            if self.depth_map is not None:
                h, w = self.depth_map.shape
                ui, vi = int(round(u)), int(round(v))
                if 0 <= ui < w and 0 <= vi < h:
                    patch = self.depth_map[max(0, vi-7):vi+8, max(0, ui-7):ui+8]
                    patch = patch[np.isfinite(patch)]
                    if patch.size >= 12:
                        cand = float(np.median(patch))
                        if cand < MIN_VALID_M:
                            reason = 'below_range'; self.stats['near'] += 1
                        elif cand > self.max_valid:
                            # beyond validated range: all models underestimate
                            # by 23-32 mm, so this cannot be trusted
                            reason = 'beyond_validated_range'; self.stats['far'] += 1
                        else:
                            z, method, reason = cand, 'monocular', None
                            self.stats['ok'] += 1
                    else:
                        self.stats['nomap'] += 1
                else:
                    self.stats['nomap'] += 1
            else:
                self.stats['nomap'] += 1

            if z is None:
                out.append({**det, 'position_3d': None,
                            'depth_method': 'unavailable',
                            'reject_reason': reason, 'depth_m': None})
                continue

            # back-project with the EndoVis intrinsics if the resolution
            # matches; otherwise report depth without 3D, since we have no
            # camera model for this input
            cap = self.capability or {}
            if cap.get('calib_valid'):
                fx = fy = 1125.55
                cx, cy = 960.22, 540.0
                X = (u - cx) * z / fx
                Y = (v - cy) * z / fy
                pos = {'x': round(X, 5), 'y': round(Y, 5), 'z': round(z, 5)}
            else:
                pos = None

            out.append({**det, 'position_3d': pos,
                        'depth_method': method,
                        'reject_reason': None if pos else 'no_camera_model',
                        'depth_m': round(z, 5)})

        m = String()
        m.data = json.dumps({
            'frame_id':   data['frame_id'],
            'timestamp':  data['timestamp'],
            'image_size': data.get('image_size', [1920, 1080]),
            'detections': out,
            'depth_source': 'monocular',
            'stereo_available': False})
        self.pub.publish(m)

        if data['frame_id'] % 50 == 0:
            t = sum(self.stats.values()) or 1
            self.get_logger().info(
                f"mono depth {self.last_ms:.0f}ms  "
                f"ok={self.stats['ok']} ({self.stats['ok']/t*100:.0f}%)  "
                f"far={self.stats['far']} near={self.stats['near']} "
                f"nomap={self.stats['nomap']}")


def main(args=None):
    rclpy.init(args=args)
    node = MonoDepthNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
