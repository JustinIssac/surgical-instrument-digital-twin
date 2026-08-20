"""
Signed tip-to-surface clearance.

Two independent paths derive from the same rectified disparity field: the
instrument tip (segmentation -> PCA tip estimate -> patch-median depth ->
back-projection -> Kalman) and the tissue surface (dense back-projection of
the same field). They share only the disparity. Their geometric relationship
is therefore a check on everything downstream of it.

The quantity has a predicted sign. Instruments occlude tissue, so the tip
lies between the camera and the surface and the clearance must be positive.
Negative values are physically impossible and measure the accumulated error
of depth estimation, tip estimation and back-projection together.

WHAT THIS IS NOT
  This is agreement between two estimates, not accuracy against truth. The
  surface is measured, not ground truth. A depth bias affecting both paths
  equally cancels and is invisible here. EndoVis provides no 3D ground
  truth, so no stronger claim is available from this dataset.

  Cloud points under the instrument are removed by the bbox exclusion, so
  the local plane is fitted to an annulus of surrounding tissue and
  interpolated beneath the tip. Where real tissue curves sharply under the
  instrument the plane is an approximation, and the residual column records
  how well it held.
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
import numpy as np
import json, csv, time
from pathlib import Path

CLOUD_DTYPE = np.dtype([('x', '<f4'), ('y', '<f4'),
                        ('z', '<f4'), ('rgb', '<u4')])


class ClearanceNode(Node):
    def __init__(self):
        super().__init__('clearance_node')

        self.declare_parameter('radius_m', 0.020)      # neighbourhood radius
        self.declare_parameter('min_points', 25)       # below this, no estimate
        self.declare_parameter('max_age_s', 0.25)      # cloud staleness limit
        self.declare_parameter('csv_path',
                               '/home/inoruske/surgical_twin_ws/results/clearance.csv')

        self.radius = float(self.get_parameter('radius_m').value)
        self.minpts = int(self.get_parameter('min_points').value)
        self.maxage = float(self.get_parameter('max_age_s').value)

        self.cloud = None
        self.cloud_t = 0.0
        self.rows = 0
        self.skipped = {'nocloud': 0, 'stale': 0, 'sparse': 0}

        # Append rather than truncate: restarting the node during a session
        # would otherwise silently destroy the run being collected.
        path = Path(self.get_parameter('csv_path').value)
        path.parent.mkdir(parents=True, exist_ok=True)
        fresh = not path.exists() or path.stat().st_size == 0
        self.fh = open(path, 'a', newline='')
        self.csv = csv.writer(self.fh)
        if fresh:
            self.csv.writerow(['t', 'track_id', 'class_name',
                               'tip_x_mm', 'tip_y_mm', 'tip_z_mm',
                               'n_neighbours', 'plane_rms_mm',
                               'clearance_mm'])

        self.create_subscription(PointCloud2, '/tissue_cloud', self.cb_cloud, 1)
        self.create_subscription(String, '/instrument_poses_filtered',
                                 self.cb_poses, 10)
        self.create_timer(10.0, self.report)
        self.get_logger().info(
            f'Clearance node ready\n'
            f'  radius {self.radius*1000:.0f}mm, min {self.minpts} points\n'
            f'  -> {path}')

    def cb_cloud(self, msg):
        a = np.frombuffer(msg.data, dtype=CLOUD_DTYPE, count=msg.width)
        self.cloud = np.stack([a['x'], a['y'], a['z']], axis=1).astype(np.float64)
        self.cloud_t = time.monotonic()

    def clearance(self, tip):
        """Signed distance from tip to the local tissue plane, in metres."""
        if self.cloud is None:
            self.skipped['nocloud'] += 1
            return None
        if time.monotonic() - self.cloud_t > self.maxage:
            self.skipped['stale'] += 1
            return None

        d = self.cloud - tip
        near = self.cloud[(d * d).sum(axis=1) <= self.radius ** 2]
        if near.shape[0] < self.minpts:
            self.skipped['sparse'] += 1
            return None

        # Local plane by PCA: the normal is the direction of least variance.
        c = near.mean(axis=0)
        _, sv, vt = np.linalg.svd(near - c, full_matrices=False)
        n = vt[2]

        # Orient the normal toward the camera, which sits at the world origin
        # since camera_origin_z was removed. Without this the sign would flip
        # arbitrarily with the SVD's internal convention.
        if np.dot(n, -c) < 0:
            n = -n

        # RMS distance of the neighbourhood to its own fitted plane: how
        # planar the patch actually is. A large value means the clearance
        # figure rests on a poor local model and should be read with care.
        rms = float(np.sqrt(np.mean(((near - c) @ n) ** 2)))
        return float(np.dot(tip - c, n)), near.shape[0], rms

    def cb_poses(self, msg):
        try:
            data = json.loads(msg.data)
        except Exception:
            return
        tracks = data.get('tracks') or data.get('instruments') or []
        now = time.time()

        for tr in tracks:
            p = tr.get('position')
            if not p or len(p) != 3:
                continue
            res = self.clearance(np.asarray(p, dtype=np.float64))
            if res is None:
                continue
            dist, npts, rms = res
            self.csv.writerow([f'{now:.3f}', tr.get('track_id'),
                               tr.get('class_name'),
                               f'{p[0]*1000:.2f}', f'{p[1]*1000:.2f}',
                               f'{p[2]*1000:.2f}',
                               npts, f'{rms*1000:.2f}', f'{dist*1000:.2f}'])
            self.rows += 1
        self.fh.flush()

    def report(self):
        s = self.skipped
        self.get_logger().info(
            f'clearance: {self.rows} samples | skipped '
            f'nocloud={s["nocloud"]} stale={s["stale"]} sparse={s["sparse"]}')

    def destroy_node(self):
        self.fh.close()
        super().destroy_node()


def main():
    rclpy.init()
    n = ClearanceNode()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        if rclpy.ok():          # Ctrl-C already shuts the context down
            rclpy.shutdown()


if __name__ == '__main__':
    main()
