"""
External pipeline verification.

Subscribes to every topic, checks rate AND payload schema, and reports
which of the recently-changed behaviours are actually working.
Run with the pipeline live. Takes ~20 s.
"""
import rclpy, json, time, sys
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from visualization_msgs.msg import MarkerArray
from collections import defaultdict

DURATION = 20.0

class Smoke(Node):
    def __init__(self):
        super().__init__('smoke_test')
        self.t0 = time.monotonic()
        self.n  = defaultdict(int)
        self.samples = {}
        subs = [
            ('/camera/image_raw', Image), ('/camera/right/image_raw', Image),
            ('/annotated_frame', Image),  ('/disparity_image', Image),
            ('/instrument_detections', String),
            ('/instrument_detections_3d', String),
            ('/instrument_poses_3d', String),
            ('/instrument_poses_filtered', String),
            ('/instrument_markers', MarkerArray),
        ]
        for topic, typ in subs:
            self.create_subscription(
                typ, topic, lambda m, t=topic: self.cb(t, m), 10)
        self.create_timer(DURATION, self.report)

    def cb(self, topic, msg):
        self.n[topic] += 1
        if topic not in self.samples and isinstance(msg, String):
            try: self.samples[topic] = json.loads(msg.data)
            except Exception: pass
        if topic == '/instrument_markers' and topic not in self.samples:
            self.samples[topic] = {'n_markers': len(msg.markers)}

    # --------------------------------------------------------------
    def report(self):
        el = time.monotonic() - self.t0
        print(f"\n{'='*64}\nPIPELINE SMOKE TEST  ({el:.0f}s)\n{'='*64}")
        print(f"{'topic':34s} {'msgs':>6s} {'Hz':>7s}")
        print("-"*52)
        for t in ['/camera/image_raw','/camera/right/image_raw',
                  '/instrument_detections','/disparity_image',
                  '/instrument_detections_3d','/instrument_poses_3d',
                  '/instrument_poses_filtered','/instrument_markers',
                  '/annotated_frame']:
            c = self.n[t]
            print(f"{t:34s} {c:6d} {c/el:7.2f}" + ("" if c else "   <-- SILENT"))

        print(f"\n{'-'*64}\nBEHAVIOUR CHECKS\n{'-'*64}")
        ok = fail = 0
        def check(label, cond, detail=""):
            nonlocal ok, fail
            if cond: ok += 1;  print(f"  PASS  {label} {detail}")
            else:    fail += 1; print(f"  FAIL  {label} {detail}")

        d = self.samples.get('/instrument_detections')
        if d and d.get('detections'):
            x = d['detections'][0]
            check("B11 identity field", 'identity' in x, f"= {x.get('identity')}")
            check("4.1 tip_px published", 'tip_px' in x, f"= {x.get('tip_px')}")
            check("3.2 continuous yaw", isinstance(x.get('shaft_yaw_rad'), float),
                  f"= {x.get('shaft_yaw_rad')}")
            check("tip != centroid", x.get('tip_px') != x.get('centroid_px'),
                  f"centroid {x.get('centroid_px')}")
        else:
            print("  SKIP  no detections seen")

        d3 = self.samples.get('/instrument_detections_3d')
        if d3 and d3.get('detections'):
            x = d3['detections'][0]
            z = x.get('position_3d', {}).get('z')
            check("2.x depth present", z is not None, f"z = {z} m")
            check("depth physiological", z is not None and 0.02 <= z <= 0.15)
            check("depth_method reported", 'depth_method' in x,
                  f"= {x.get('depth_method')}")

        pw = self.samples.get('/instrument_poses_3d')
        if pw and pw.get('poses'):
            x = pw['poses'][0]
            check("B5 world frame declared", pw.get('frame') == 'gazebo_world')
            check("B5 optical!=world", x.get('position_3d') != x.get('position_cam'),
                  f"world {x.get('position_3d')}")
            check("B2 shaft_dir_world", 'shaft_dir_world' in x)

        f = self.samples.get('/instrument_poses_filtered')
        if f is not None:
            check("B6 track_id keying",
                  'tracks' in f and all('track_id' in t for t in f['tracks']),
                  f"{len(f.get('tracks', []))} tracks")
            if f.get('tracks'):
                pl = f['tracks'][0].get('path_len_m', 0)
                check("B7 path length sane", pl < 1.0, f"= {pl*1000:.1f} mm")

        m = self.samples.get('/instrument_markers')
        check("B16 rviz markers", m is not None and m.get('n_markers', 0) > 0,
              f"{m.get('n_markers') if m else 0} markers")

        print(f"\n  {ok} passed, {fail} failed")
        print(f"{'='*64}")
        rclpy.shutdown()

def main():
    rclpy.init(); n = Smoke()
    print(f"listening for {DURATION:.0f}s ...")
    try: rclpy.spin(n)
    except Exception: pass

if __name__ == '__main__':
    main()
