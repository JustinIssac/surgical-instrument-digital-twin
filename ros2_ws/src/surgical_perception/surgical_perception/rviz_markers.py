"""
RViz visualisation + TF tree.

  B16          MarkerArray visualisation (previously no rviz support at all)
  Finding 3.1  publishes a real TF tree: world -> camera_optical
               Until now 3D points were used with no frame management,
               despite this being a robotics project.

Markers per track:
  sphere      estimated jaws position (the tracked point)
  arrow       shaft direction, drawn proximal -> distal
  text        class name, track id, speed
  line strip  recent trajectory (Economy of Motion made visible)
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, TransformStamped
from tf2_ros import StaticTransformBroadcaster
from rclpy.duration import Duration
import numpy as np
import json
import math
from collections import defaultdict, deque

# camera optical (X right, Y down, Z fwd) -> ROS world (X fwd, Y left, Z up)
R_OPT2ROS = np.array([[0.0,  0.0, 1.0],
                      [-1.0, 0.0, 0.0],
                      [0.0, -1.0, 0.0]], dtype=np.float64)

PALETTE = [(0.20,0.60,1.00), (1.00,0.40,0.20), (0.20,1.00,0.40),
           (1.00,0.20,0.80), (0.80,0.80,0.20), (0.60,0.20,1.00),
           (0.20,0.80,0.80), (1.00,0.60,0.20)]

CLASS_NAMES = ['Large_Needle_Driver_Left','Large_Needle_Driver_Right',
               'Prograsp_Forceps_Left','Prograsp_Forceps_Right',
               'Maryland_Bipolar_Forceps','Bipolar_Forceps',
               'Monopolar_Curved_Scissors','Grasping_Retractor_Right']


def rot_to_quat(R):
    """Rotation matrix -> (x, y, z, w)."""
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2,1] - R[1,2]) / s
        y = (R[0,2] - R[2,0]) / s
        z = (R[1,0] - R[0,1]) / s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
        w = (R[2,1] - R[1,2]) / s; x = 0.25 * s
        y = (R[0,1] + R[1,0]) / s; z = (R[0,2] + R[2,0]) / s
    elif R[1,1] > R[2,2]:
        s = math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
        w = (R[0,2] - R[2,0]) / s; x = (R[0,1] + R[1,0]) / s
        y = 0.25 * s;               z = (R[1,2] + R[2,1]) / s
    else:
        s = math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
        w = (R[1,0] - R[0,1]) / s; x = (R[0,2] + R[2,0]) / s
        y = (R[1,2] + R[2,1]) / s; z = 0.25 * s
    return x, y, z, w


class RvizMarkerNode(Node):
    def __init__(self):
        super().__init__('rviz_markers')

        self.declare_parameter('trail_len', 40)
        self.declare_parameter('camera_z', 0.50)
        self.trail_len = int(self.get_parameter('trail_len').value)
        cam_z          = float(self.get_parameter('camera_z').value)

        # ---- static TF: world -> camera_optical (Finding 3.1) ----
        self.tf_static = StaticTransformBroadcaster(self)
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = 'world'
        tf.child_frame_id  = 'camera_optical'
        tf.transform.translation.z = cam_z
        qx, qy, qz, qw = rot_to_quat(R_OPT2ROS)
        tf.transform.rotation.x = qx; tf.transform.rotation.y = qy
        tf.transform.rotation.z = qz; tf.transform.rotation.w = qw
        self.tf_static.sendTransform(tf)

        self.trails = defaultdict(lambda: deque(maxlen=self.trail_len))

        self.pub = self.create_publisher(MarkerArray, '/instrument_markers', 10)
        self.create_subscription(
            String, '/instrument_poses_filtered', self.callback, 10)

        self.get_logger().info(
            'RViz markers ready\n'
            '  fixed frame : world\n'
            '  TF          : world -> camera_optical\n'
            '  topic       : /instrument_markers')

    # ------------------------------------------------------------------
    def _base(self, mid, mtype, stamp):
        m = Marker()
        m.header.frame_id = 'world'
        m.header.stamp = stamp
        m.ns = 'instruments'
        m.id = mid
        m.type = mtype
        m.action = Marker.ADD
        m.lifetime = Duration(seconds=0.6).to_msg()   # dead tracks expire
        m.pose.orientation.w = 1.0
        return m

    def callback(self, msg):
        data  = json.loads(msg.data)
        stamp = self.get_clock().now().to_msg()
        arr   = MarkerArray()

        live = set()
        for tr in data.get('tracks', []):
            tid  = int(tr['track_id'])
            live.add(tid)
            pos  = np.array(tr['position'], dtype=float)
            name = tr['class_name']
            unknown = tr.get('identity') == 'unknown'
            cidx = CLASS_NAMES.index(name) if name in CLASS_NAMES else 0
            r, g, b = ((0.55, 0.55, 0.55) if unknown
                       else PALETTE[cidx % len(PALETTE)])
            d = np.array(tr.get('shaft_dir', [1.0, 0.0, 0.0]), dtype=float)
            n = np.linalg.norm(d)
            d = d / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])

            # --- jaws sphere ---
            s = self._base(tid*10 + 0, Marker.SPHERE, stamp)
            s.pose.position.x, s.pose.position.y, s.pose.position.z = pos
            s.scale.x = s.scale.y = s.scale.z = 0.010
            s.color = (ColorRGBA(r=1.0, g=0.65, b=0.0, a=1.0) if unknown
                       else ColorRGBA(r=0.1, g=1.0, b=0.2, a=1.0))
            arr.markers.append(s)

            # --- shaft arrow, proximal -> distal ---
            a = self._base(tid*10 + 1, Marker.ARROW, stamp)
            # B20: arrow length from the observed pixel extent,
            # converted to metres via the pinhole relation, so
            # rviz and Gazebo agree on instrument geometry
            apx = tr.get('axis_len_px')
            depth = abs(float(pos[0]))
            L = (float(apx) * depth / 1125.55) if apx else 0.06
            L = min(max(L, 0.02), 0.35)
            tail = pos - d * L
            a.points = [Point(x=float(tail[0]), y=float(tail[1]), z=float(tail[2])),
                        Point(x=float(pos[0]),  y=float(pos[1]),  z=float(pos[2]))]
            a.scale.x, a.scale.y, a.scale.z = 0.004, 0.009, 0.010
            a.color = ColorRGBA(r=float(r), g=float(g), b=float(b), a=0.95)
            arr.markers.append(a)

            # --- label ---
            t = self._base(tid*10 + 2, Marker.TEXT_VIEW_FACING, stamp)
            t.pose.position.x = float(pos[0])
            t.pose.position.y = float(pos[1])
            t.pose.position.z = float(pos[2]) + 0.022
            t.scale.z = 0.007
            t.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.9)
            label = ("UNKNOWN (pred: "
                     + str(tr.get("predicted_class", "?")) + ")") if unknown else name
            t.text = f"{label}  {tr.get('path_len_m',0)*1000:.0f}mm"
            arr.markers.append(t)

            # --- trajectory trail ---
            self.trails[tid].append(pos.copy())
            if len(self.trails[tid]) > 1:
                l = self._base(tid*10 + 3, Marker.LINE_STRIP, stamp)
                l.scale.x = 0.0018
                l.color = ColorRGBA(r=float(r), g=float(g), b=float(b), a=0.55)
                l.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2]))
                            for p in self.trails[tid]]
                arr.markers.append(l)

        for tid in list(self.trails):
            if tid not in live:
                del self.trails[tid]

        self.pub.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = RvizMarkerNode()
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
