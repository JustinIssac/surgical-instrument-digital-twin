"""
Coordinate frame transformation node.

REWRITTEN. Previously this node re-derived 3D positions from pixel
coordinates using its own (incorrect) intrinsics, silently overriding
the stereo node's output whenever stereo depth was unavailable.

It now performs exactly one job: convert 3D points from the camera
OPTICAL frame into the Gazebo WORLD frame, and forward orientation.
No camera intrinsics live here any more.

  B1  wrong intrinsics deleted (not patched -- removed entirely)
  B2  shaft_yaw_rad from mask PCA is forwarded, replacing the old
      binary bbox aspect-ratio guess
  B3  operates on tip_px / tip-derived 3D, never the shaft centroid
  B5  optical -> world frame conversion (REP-103)

Frame conventions
  camera optical : X right,   Y down, Z forward   (OpenCV)
  ROS / Gazebo   : X forward, Y left, Z up        (REP-103)
      x_w = z_o ,  y_w = -x_o ,  z_w = -y_o
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import numpy as np
import json
import math

# Rotation: camera optical frame -> ROS body/world frame (REP-103)
R_OPT2ROS = np.array([[0.0,  0.0, 1.0],
                      [-1.0, 0.0, 0.0],
                      [0.0, -1.0, 0.0]], dtype=np.float64)


class PoseEstimatorNode(Node):
    def __init__(self):
        super().__init__('pose_estimator')

        # World origin IS the camera optical centre.
        #
        # This was previously offset to (0, 0, 0.50) to lift the twin to a
        # visible height above the Gazebo ground plane -- harmless while
        # nothing else in the scene was derived from measurement. Once the
        # tissue surface was back-projected from the same disparity field,
        # the two paths disagreed by exactly that constant (477 mm), since
        # the cloud is camera-relative and the instruments were not.
        #
        # With the offset removed, a reported coordinate is a true distance
        # from the endoscope, and the two independent paths are directly
        # comparable -- which is what makes tip-to-surface distance a
        # meaningful measurement rather than one carrying a display constant.
        self.declare_parameter('camera_origin_x', 0.0)
        self.declare_parameter('camera_origin_y', 0.0)
        self.declare_parameter('camera_origin_z', 0.0)
        self.declare_parameter('scene_scale', 1.0)

        self.cam_origin = np.array([
            self.get_parameter('camera_origin_x').value,
            self.get_parameter('camera_origin_y').value,
            self.get_parameter('camera_origin_z').value], dtype=np.float64)
        self.scale = float(self.get_parameter('scene_scale').value)

        self.sub = self.create_subscription(
            String, '/instrument_detections_3d', self.callback, 10)
        self.pub = self.create_publisher(
            String, '/instrument_poses_3d', 10)

        self.get_logger().info(
            'Pose estimator ready (frame transform only)\n'
            '  optical -> world : x_w=z_o, y_w=-x_o, z_w=-y_o\n'
            f'  camera origin    : {self.cam_origin.tolist()}\n'
            '  no intrinsics used in this node')

    # ------------------------------------------------------------------
    def optical_to_world(self, p_opt):
        """3D point, camera optical frame -> Gazebo world frame."""
        return self.cam_origin + self.scale * (R_OPT2ROS @ np.asarray(p_opt))

    def shaft_direction_world(self, yaw_img):
        """
        Convert the in-image shaft direction into a world-frame unit vector.

        yaw_img = atan2(dy, dx) in image pixel coords (y increases downward),
        pointing from the instrument body toward the jaws. In the optical
        frame that is (cos y, sin y, 0); the out-of-plane component is not
        observable from a single view, so it is taken as zero and this is
        recorded as a known limitation.
        """
        d_opt = np.array([math.cos(yaw_img), math.sin(yaw_img), 0.0])
        d_w   = R_OPT2ROS @ d_opt
        n     = np.linalg.norm(d_w)
        return (d_w / n) if n > 1e-9 else np.array([1.0, 0.0, 0.0])

    # ------------------------------------------------------------------
    def callback(self, msg):
        data  = json.loads(msg.data)
        poses = []

        for det in data.get('detections', []):
            p3 = det.get('position_3d')
            if p3 is None:
                # No metric depth available. Forward the 2D result so the
                # detector's output is still usable and visible, but do not
                # synthesise a 3D position.
                poses.append({
                    'class_id':      det['class_id'],
                    'class_name':    det['class_name'],
                    'confidence':    det['confidence'],
                    'position_3d':   None,
                    'depth_method':  det.get('depth_method', 'unavailable'),
                    'identity':      det.get('identity', 'known'),
                    'tip_px':        det.get('tip_px'),
                    'centroid_px':   det.get('centroid_px'),
                    'shaft_yaw_rad': det.get('shaft_yaw_rad'),
                    'frame_id':      data.get('frame_id', 0)})
                continue

            p_opt = [p3['x'], p3['y'], p3['z']]
            p_w   = self.optical_to_world(p_opt)

            yaw_img = det.get('shaft_yaw_rad')
            if yaw_img is None:
                d_w, yaw_known = np.array([1.0, 0.0, 0.0]), False
            else:
                d_w, yaw_known = self.shaft_direction_world(float(yaw_img)), True

            poses.append({
                'class_id':      det['class_id'],
                'class_name':    det['class_name'],
                'confidence':    det['confidence'],
                # world frame, metres -- what the twin consumes
                'position_3d':   {'x': round(float(p_w[0]), 5),
                                  'y': round(float(p_w[1]), 5),
                                  'z': round(float(p_w[2]), 5)},
                # camera frame retained for evaluation / reprojection
                'position_cam':  {'x': p3['x'], 'y': p3['y'], 'z': p3['z']},
                'shaft_dir_world': [round(float(v), 5) for v in d_w],
                'shaft_yaw_rad': det.get('shaft_yaw_rad'),
                'yaw_observed':  yaw_known,
                'depth_method':  det.get('depth_method', 'unknown'),
                # R3: identity must survive to the twin, else the UNKNOWN
                # safety feature is silently discarded downstream
                'identity':      det.get('identity', 'known'),
                'predicted_class': det.get('predicted_class'),
                'tip_px':        det.get('tip_px'),
                'axis_len_px':   det.get('axis_len_px'),
                'centroid_px':   det.get('centroid_px'),
                'frame_id':      data.get('frame_id', 0),
            })

        out = String()
        out.data = json.dumps({
            'frame_id':  data.get('frame_id', 0),
            'timestamp': data.get('timestamp', 0),
            'frame':     'gazebo_world',
            'poses':     poses})
        self.pub.publish(out)

        if poses and data.get('frame_id', 0) % 50 == 0:
            ns = sum(1 for p in poses if p['depth_method'] == 'stereo')
            self.get_logger().info(
                f"frame {data.get('frame_id')}: {len(poses)} poses "
                f"({ns} stereo depth)")


def main(args=None):
    rclpy.init(args=args)
    node = PoseEstimatorNode()
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
