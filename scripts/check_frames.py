"""Compare the coordinate ranges of the two world-frame paths.

Both the tissue cloud and the instrument poses claim to be in 'world' via
the same REP-103 convention. If they disagree, they cannot both be right,
and the twin's absolute placement is unvalidated.
"""
import rclpy, json, numpy as np
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String

class Check(Node):
    def __init__(self):
        super().__init__('check_frames')
        self.cloud = None; self.pose = None
        self.create_subscription(PointCloud2, '/tissue_cloud', self.cb_cloud, 1)
        self.create_subscription(String, '/instrument_poses_filtered', self.cb_pose, 10)

    def cb_cloud(self, m):
        a = np.frombuffer(m.data, dtype=np.dtype([
            ('x','<f4'),('y','<f4'),('z','<f4'),('rgb','<u4')]), count=m.width)
        self.cloud = {k: (float(a[k].min()), float(a[k].max())) for k in 'xyz'}
        self.cloud['n'] = m.width
        self.report()

    def cb_pose(self, m):
        try:
            tr = json.loads(m.data).get('tracks') or json.loads(m.data).get('instruments')
        except Exception:
            return
        if tr:
            self.pose = tr[0]
        self.report()

    def report(self):
        if self.cloud and self.pose:
            print("\n--- TISSUE CLOUD (world) ---")
            print(f"  n = {self.cloud['n']}")
            for k in 'xyz':
                lo, hi = self.cloud[k]
                print(f"  {k}: {lo*1000:8.1f} .. {hi*1000:8.1f} mm")
            print("--- INSTRUMENT (world) ---")
            print(f"  {json.dumps(self.pose, indent=2)[:600]}")
            raise SystemExit

rclpy.init()
try:
    rclpy.spin(Check())
except SystemExit:
    pass
