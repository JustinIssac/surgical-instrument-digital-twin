"""
Stereo depth estimation with proper rectification.

Fixes over previous version:
  2.1  loads corrected intrinsics from config (fx == fy, square pixels)
  2.2  performs stereoRectify + remap before SGBM
  2.3  distortion coefficients actually applied
  2.4  physiologically-motivated depth gate + disparity quality checks
  2.5  time-synchronised left/right via message_filters
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import message_filters
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
import cv2
import numpy as np
import yaml
import json
from pathlib import Path

CALIB = "/home/inoruske/surgical_twin_ws/config/camera_calib.yaml"


class StereoDepthNode(Node):
    def __init__(self):
        super().__init__('stereo_depth_node')

        self.declare_parameter('calib_path', CALIB)
        self.declare_parameter('depth_min_m', 0.02)   # 3 cm
        self.declare_parameter('depth_max_m', 0.15)   # 20 cm
        self.declare_parameter('max_depth_jump_m', 0.015)
        # how many frames a stale depth may be held. At ~6.4Hz, 5 frames
        # is under a second; the previous 30 was nearly 5s, long enough
        # for an instrument to be anywhere.
        self.declare_parameter('hold_frames', 5)
        # R2: SGBM at 1920x1080 with 192 disparities is far too slow for
        # 10 fps. Compute at reduced scale and upsample; disparity scales
        # linearly with image width so values are corrected accordingly.
        self.declare_parameter('disp_scale', 0.5)

        calib_path = self.get_parameter('calib_path').value
        self.dmin  = self.get_parameter('depth_min_m').value
        self.dmax  = self.get_parameter('depth_max_m').value
        self.djump = self.get_parameter('max_depth_jump_m').value
        self.hold_frames = int(self.get_parameter('hold_frames').value)
        self.dscale = float(self.get_parameter('disp_scale').value)

        with open(calib_path) as f:
            cfg = yaml.safe_load(f)

        W, H = cfg['image_width'], cfg['image_height']
        self.size = (W, H)

        L, R_ = cfg['left'], cfg['right']
        self.K1 = np.array([[L['fx'], 0, L['cx']],
                            [0, L['fy'], L['cy']],
                            [0, 0, 1]], dtype=np.float64)
        self.D1 = np.array(L['dist'], dtype=np.float64)
        self.K2 = np.array([[R_['fx'], 0, R_['cx']],
                            [0, R_['fy'], R_['cy']],
                            [0, 0, 1]], dtype=np.float64)
        self.D2 = np.array(R_['dist'], dtype=np.float64)

        Rmat = np.array(cfg['stereo']['R'], dtype=np.float64)
        Tvec = np.array(cfg['stereo']['T_m'], dtype=np.float64).reshape(3, 1)

        # ---- Rectification (Finding 2.2) ----
        self.R1, self.R2, self.P1, self.P2, self.Q, _, _ = cv2.stereoRectify(
            self.K1, self.D1, self.K2, self.D2, self.size,
            Rmat, Tvec, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
        )
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            self.K1, self.D1, self.R1, self.P1, self.size, cv2.CV_32FC1)
        self.map2x, self.map2y = cv2.initUndistortRectifyMap(
            self.K2, self.D2, self.R2, self.P2, self.size, cv2.CV_32FC1)

        # Rectified focal length & baseline drive the depth equation
        self.fx_rect   = float(self.P1[0, 0])
        self.baseline  = float(abs(self.P2[0, 3] / self.P2[0, 0]))

        # Valid-pixel mask: exclude synthetic padding, warped into rect frame
        ar = cfg['active_region']
        raw_mask = np.zeros((H, W), np.uint8)
        raw_mask[ar['y']:ar['y']+ar['h'], ar['x']:ar['x']+ar['w']] = 255
        self.valid_mask = cv2.remap(raw_mask, self.map1x, self.map1y,
                                    cv2.INTER_NEAREST) > 127

        self.stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=int(192*self.dscale)//16*16, blockSize=7,
            P1=8*3*7**2, P2=32*3*7**2,
            disp12MaxDiff=1, uniquenessRatio=12,
            speckleWindowSize=150, speckleRange=2,
            preFilterCap=63, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

        # Input capability governs whether metric depth is even possible.
        # Without this the node silently substitutes a constant when stereo
        # is unavailable, producing a confident but fabricated 3D twin.
        self.depth_mode = 'stereo'      # assume stereo until told otherwise
        self.capability = None
        self.create_subscription(
            String, '/input_capability', self.capability_callback,
            QoSProfile(depth=1,
                       durability=DurabilityPolicy.TRANSIENT_LOCAL,
                       reliability=ReliabilityPolicy.RELIABLE))

        self.bridge   = CvBridge()
        self.disp     = None
        self.last_depth = {}    # cls -> (depth, frame_id)
        self.stats = {'stereo': 0, 'gate': 0, 'nodisp': 0, 'jump': 0}

        # ---- Time-synchronised stereo subscription (Finding 2.5) ----
        ls = message_filters.Subscriber(self, Image, '/camera/image_raw')
        rs = message_filters.Subscriber(self, Image, '/camera/right/image_raw')
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [ls, rs], queue_size=5, slop=0.02)
        self.sync.registerCallback(self.stereo_callback)

        self.create_subscription(String, '/instrument_detections',
                                 self.detection_callback, 10)

        self.depth_pub = self.create_publisher(
            String, '/instrument_detections_3d', 10)
        self.disp_pub  = self.create_publisher(Image, '/disparity_image', 10)

        self.get_logger().info(
            f"Stereo depth node ready (rectified)\n"
            f"  fx_rect  = {self.fx_rect:.2f}\n"
            f"  baseline = {self.baseline*1000:.4f} mm\n"
            f"  gate     = [{self.dmin:.3f}, {self.dmax:.3f}] m\n"
            f"  valid px = {self.valid_mask.sum()/self.valid_mask.size*100:.1f}%")

    # ------------------------------------------------------------------
    def capability_callback(self, msg):
        try:
            cap = json.loads(msg.data)
        except Exception:
            return
        self.capability = cap
        self.depth_mode = cap.get('depth_mode', 'stereo')
        if self.depth_mode != 'stereo':
            self.get_logger().warn(
                f"input is {cap.get('kind')} at "
                f"{cap.get('width')}x{cap.get('height')} "
                f"(calib_valid={cap.get('calib_valid')}) -- "
                f"metric stereo depth UNAVAILABLE. Reporting 2D only; "
                f"no depth will be fabricated.")
        else:
            self.get_logger().info('input supports metric stereo depth')

    def stereo_callback(self, lmsg, rmsg):
        import time as _t; _t0 = _t.monotonic()
        left  = self.bridge.imgmsg_to_cv2(lmsg, 'bgr8')
        right = self.bridge.imgmsg_to_cv2(rmsg, 'bgr8')

        lr = cv2.remap(left,  self.map1x, self.map1y, cv2.INTER_LINEAR)
        rr = cv2.remap(right, self.map2x, self.map2y, cv2.INTER_LINEAR)

        lg = cv2.cvtColor(lr, cv2.COLOR_BGR2GRAY)
        rg = cv2.cvtColor(rr, cv2.COLOR_BGR2GRAY)

        if self.dscale != 1.0:
            sh = (int(lg.shape[1]*self.dscale), int(lg.shape[0]*self.dscale))
            lg_s = cv2.resize(lg, sh, interpolation=cv2.INTER_AREA)
            rg_s = cv2.resize(rg, sh, interpolation=cv2.INTER_AREA)
            d_s  = self.stereo.compute(lg_s, rg_s).astype(np.float32) / 16.0
            # disparity is in pixels -> rescale with image width
            d = cv2.resize(d_s, (lg.shape[1], lg.shape[0]),
                           interpolation=cv2.INTER_NEAREST) / self.dscale
        else:
            d = self.stereo.compute(lg, rg).astype(np.float32) / 16.0
        d[~self.valid_mask] = np.nan
        d[d <= 0.5] = np.nan          # sub-pixel noise floor
        self.disp = d
        self.last_disp_ms = (_t.monotonic() - _t0) * 1000.0

        vis = cv2.normalize(np.nan_to_num(d), None, 0, 255,
                            cv2.NORM_MINMAX, cv2.CV_8U)
        self.disp_pub.publish(self.bridge.cv2_to_imgmsg(
            cv2.applyColorMap(vis, cv2.COLORMAP_PLASMA), 'bgr8'))

    # ------------------------------------------------------------------
    def rectify_point(self, u, v):
        """Map a raw-image pixel into rectified image coordinates."""
        pt = np.array([[[float(u), float(v)]]], dtype=np.float64)
        out = cv2.undistortPoints(pt, self.K1, self.D1,
                                  R=self.R1, P=self.P1)
        return float(out[0, 0, 0]), float(out[0, 0, 1])

    def sample_depth(self, u, v, win=7):
        if self.disp is None:
            return None, 'nodisp'
        h, w = self.disp.shape
        ur, vr = self.rectify_point(u, v)
        ui, vi = int(round(ur)), int(round(vr))
        if not (0 <= ui < w and 0 <= vi < h):
            return None, 'nodisp'

        patch = self.disp[max(0, vi-win):min(h, vi+win+1),
                          max(0, ui-win):min(w, ui+win+1)]
        vals = patch[~np.isnan(patch)]
        if vals.size < 12:
            return None, 'nodisp'

        # Reject multi-modal patches (instrument edge straddling background)
        if float(np.std(vals)) > 6.0:
            return None, 'nodisp'

        return self.fx_rect * self.baseline / float(np.median(vals)), 'stereo'

    # ------------------------------------------------------------------
    def detection_callback(self, msg):
        # Exactly one node may own /instrument_detections_3d. When stereo is
        # unavailable, mono_depth_node takes over; publishing empty results
        # here as well makes twin_sync see alternating null/valid messages,
        # which destroys track continuity.
        if self.depth_mode != 'stereo':
            return

        data = json.loads(msg.data)
        out  = []

        for det in data.get('detections', []):
            u, v = det.get('tip_px', det['centroid_px'])
            cls  = det['class_name']

            z, why = self.sample_depth(u, v)

            if z is not None and not (self.dmin <= z <= self.dmax):
                z, why = None, 'gate'

            # Temporal plausibility, with two corrections:
            #  (a) the reference must expire, else one rejection cascades
            #      into permanent rejection (the reference goes stale and
            #      every later sample looks like a jump)
            #  (b) the allowance scales with elapsed frames, since a gap
            #      of N frames permits N times the movement
            if z is not None and cls in self.last_depth:
                z_prev, f_prev = self.last_depth[cls]
                gap = max(1, data['frame_id'] - f_prev)
                if gap <= 15 and abs(z - z_prev) > self.djump * gap:
                    z, why = None, 'jump'


            age = None
            if z is None:
                # Hold this instrument's last good stereo depth for a short
                # window. An instrument cannot teleport, so a 1-5 frame old
                # measurement is a defensible estimate -- and holding avoids
                # the ~26mm jump that a global constant caused, which broke
                # data association downstream.
                #
                # Beyond that window there is NO estimate. Previously a
                # 0.055m constant was substituted and labelled 'assumed',
                # which is fabrication: the reported 3D position had no
                # relationship to the instrument's actual depth. Those
                # positions fed the twin and the path-length metric.
                # Now the detection is reported in 2D with depth marked
                # unavailable, matching the monocular path.
                if cls in self.last_depth:
                    z_prev, f_prev = self.last_depth[cls]
                    age = data['frame_id'] - f_prev
                    if age <= self.hold_frames:
                        z, method = z_prev, 'held'
                        self.stats['held'] = self.stats.get('held', 0) + 1
                self.stats[why] = self.stats.get(why, 0) + 1

            if z is None:
                out.append({**det,
                            'position_3d':   None,
                            'depth_method':  'unavailable',
                            'reject_reason': why,
                            'depth_m':       None})
                self.stats['unavailable'] = self.stats.get('unavailable', 0) + 1
                continue
            else:
                method = 'stereo'
                self.last_depth[cls] = (z, data['frame_id'])
                self.stats['stereo'] += 1

            # Back-project using RECTIFIED intrinsics, then report in
            # rectified-left frame (consistent with the disparity source)
            ur, vr = self.rectify_point(u, v)
            X = (ur - self.P1[0, 2]) * z / self.P1[0, 0]
            Y = (vr - self.P1[1, 2]) * z / self.P1[1, 1]

            out.append({**det,
                        'position_3d':  {'x': round(X, 5),
                                         'y': round(Y, 5),
                                         'z': round(z, 5)},
                        'depth_method': method,
                        'depth_age_frames': age if method == 'held' else 0,
                        'reject_reason': None if method == 'stereo' else why,
                        'centroid_rect': [round(ur, 1), round(vr, 1)],
                        'depth_m': round(z, 5)})

        m = String()
        m.data = json.dumps({
            'frame_id':  data['frame_id'],
            'timestamp': data['timestamp'],
            'image_size': data.get('image_size', [1920, 1080]),
            'detections': out,
            'stereo_available': self.disp is not None})
        self.depth_pub.publish(m)

        if data['frame_id'] % 50 == 0:
            tot = sum(self.stats.values()) or 1
            self.get_logger().info(
                f"depth stats  stereo={self.stats['stereo']} "
                f"({self.stats['stereo']/tot*100:.0f}%)  "
                f"gate={self.stats.get('gate',0)} "
                f"nodisp={self.stats.get('nodisp',0)} "
                f"jump={self.stats.get('jump',0)} "
                f"held={self.stats.get('held',0)} "
                f"unavail={self.stats.get('unavailable',0)}  "
                f"sgbm={getattr(self, 'last_disp_ms', 0):.0f}ms")


def main(args=None):
    rclpy.init(args=args)
    node = StereoDepthNode()
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
