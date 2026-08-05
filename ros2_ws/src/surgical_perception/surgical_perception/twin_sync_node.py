"""
Digital twin synchronisation with multi-object tracking.

  B6  track-ID based data association (greedy NN with a distance gate)
      replaces per-class Kalman filters. Handles duplicate classes and
      survives single-frame classification flips.
  B7  Economy of Motion resets on sequence loop and on track death;
      only integrates while a track is confirmed.
  dt  derived from message timestamps, not hardcoded.
  Orientation from the world-frame shaft direction vector.
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import gz.transport13
import gz.msgs10.entity_factory_pb2 as ef_pb2
import gz.msgs10.boolean_pb2 as bool_pb2
import gz.msgs10.pose_pb2 as pose_pb2
import gz.msgs10.entity_pb2 as entity_pb2
import numpy as np
import json, math, time

GATE_M       = 0.035    # max association distance (m)
CONFIRM_HITS = 3        # detections before a track is spawned
MAX_MISSES   = 8        # frames coasting before a track dies


class Track:
    """Constant-velocity Kalman track: state [x,y,z,vx,vy,vz]."""
    _next_id = 0

    def __init__(self, pos, class_id, class_name, stamp, identity='known'):
        self.id = Track._next_id; Track._next_id += 1
        self.class_id, self.class_name = class_id, class_name
        self.class_votes = {class_id: 1}
        self.identity = identity          # 'known' | 'unknown'
        self.id_votes = {identity: 1}

        self.x = np.array([*pos, 0.0, 0.0, 0.0], dtype=np.float64)
        self.P = np.diag([1e-3, 1e-3, 1e-3, 1e-2, 1e-2, 1e-2])
        self.H = np.hstack([np.eye(3), np.zeros((3, 3))])
        self.q, self.r = 5e-3, 2e-4      # process / measurement noise

        self.hits, self.misses = 1, 0
        self.confirmed = False
        self.path_len  = 0.0
        self.last_pos  = np.array(pos, dtype=np.float64)
        self.stamp     = stamp
        self.dir_world = np.array([1.0, 0.0, 0.0])

    def predict(self, stamp):
        dt = max(1e-3, min(0.5, stamp - self.stamp))   # clamp against stalls
        self.stamp = stamp
        F = np.eye(6); F[0, 3] = F[1, 4] = F[2, 5] = dt
        self.x = F @ self.x
        Q = np.eye(6) * self.q * dt
        self.P = F @ self.P @ F.T + Q
        return self.x[:3]

    def update(self, pos, class_id, dir_world, identity='known'):
        z = np.array(pos, dtype=np.float64)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + np.eye(3) * self.r
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

        # B7: integrate path length only while confirmed
        if self.confirmed:
            self.path_len += float(np.linalg.norm(self.x[:3] - self.last_pos))
        self.last_pos = self.x[:3].copy()

        self.id_votes[identity] = self.id_votes.get(identity, 0) + 1
        # majority vote: a track is UNKNOWN only if persistently unknown
        self.identity = max(self.id_votes, key=self.id_votes.get)
        self.class_votes[class_id] = self.class_votes.get(class_id, 0) + 1
        # majority vote -> immune to single-frame label flips
        self.class_id = max(self.class_votes, key=self.class_votes.get)

        if dir_world is not None:
            self.dir_world = np.asarray(dir_world, dtype=np.float64)

        self.hits += 1; self.misses = 0
        if self.hits >= CONFIRM_HITS:
            self.confirmed = True

    def mark_missed(self):
        self.misses += 1
        return self.misses > MAX_MISSES

    @property
    def position(self): return self.x[:3]
    @property
    def speed(self):    return float(np.linalg.norm(self.x[3:]))


def dir_to_quat(d):
    """Quaternion rotating model +Z onto world direction d."""
    d = np.asarray(d, dtype=np.float64)
    n = np.linalg.norm(d)
    if n < 1e-9: return (1.0, 0.0, 0.0, 0.0)
    d = d / n
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, d); c = float(np.dot(z, d))
    if c < -0.999999: return (0.0, 1.0, 0.0, 0.0)   # 180-degree flip
    s = math.sqrt((1.0 + c) * 2.0)
    return (s * 0.5, v[0] / s, v[1] / s, v[2] / s)  # w,x,y,z


class TwinSyncNode(Node):
    def __init__(self):
        super().__init__('twin_sync_node')
        self.gz = gz.transport13.Node()

        self.class_names = [
            'Large_Needle_Driver_Left','Large_Needle_Driver_Right',
            'Prograsp_Forceps_Left','Prograsp_Forceps_Right',
            'Maryland_Bipolar_Forceps','Bipolar_Forceps',
            'Monopolar_Curved_Scissors','Grasping_Retractor_Right']
        self.colors = [(0.2,0.6,1.0),(1.0,0.4,0.2),(0.2,1.0,0.4),(1.0,0.2,0.8),
                       (0.8,0.8,0.2),(0.6,0.2,1.0),(0.2,0.8,0.8),(1.0,0.6,0.2)]

        self.sdf_tpl = """<?xml version="1.0" ?>
<sdf version="1.8">
  <model name="{name}">
    <static>true</static>
    <link name="shaft">
      <pose>0 0 -0.18 0 0 0</pose>
      <visual name="v">
        <geometry><cylinder><radius>0.005</radius><length>0.35</length></cylinder></geometry>
        <material><ambient>{r} {g} {b} 1</ambient><diffuse>{r} {g} {b} 1</diffuse></material>
      </visual>
    </link>
    <link name="jaws">
      <pose>0 0 0 0 0 0</pose>
      <visual name="v">
        <geometry><sphere><radius>0.008</radius></sphere></geometry>
        <material><ambient>1 1 0 1</ambient><diffuse>1 1 0 1</diffuse></material>
      </visual>
    </link>
    <joint name="j" type="fixed"><parent>shaft</parent><child>jaws</child></joint>
  </model>
</sdf>"""

        self.tracks, self.spawned = [], {}
        self.spawn_failed = {}        # track_id -> consecutive failures
        self.gazebo_ok    = True      # degrade gracefully if Gazebo is absent
        self.last_frame_id = -1

        self.filtered_pub = self.create_publisher(
            String, '/instrument_poses_filtered', 10)
        self.create_subscription(
            String, '/instrument_poses_3d', self.callback, 10)

        self.get_logger().info(
            f'Twin sync ready — track-based association\n'
            f'  gate {GATE_M*1000:.0f} mm, confirm {CONFIRM_HITS}, '
            f'max misses {MAX_MISSES}')

    # ------------------------------------------------------------------
    def display_name(self, track):
        """R4: an UNKNOWN track must not be labelled as a specific
        instrument type in the twin or in rviz."""
        return ('UNKNOWN' if track.identity == 'unknown'
                else self.class_names[track.class_id])

    def spawn(self, track):
        name = f"instrument_{track.id}_{self.display_name(track)}"
        r, g, b = ((0.55, 0.55, 0.55) if track.identity == 'unknown'
                   else self.colors[track.class_id % len(self.colors)])
        req = ef_pb2.EntityFactory()
        req.sdf  = self.sdf_tpl.format(name=name, r=r, g=g, b=b)
        req.name = name
        req.pose.position.z = 0.5
        if not self.gazebo_ok:
            return                      # stop hammering a dead service
        try:
            ok = self.gz.request('/world/empty/create', req,
                                 ef_pb2.EntityFactory, bool_pb2.Boolean, 400)
        except Exception:
            ok = False
        if ok:
            self.spawned[track.id] = name
            self.spawn_failed.pop(track.id, None)
            self.get_logger().info(f'spawned {name}')
        else:
            n = self.spawn_failed.get(track.id, 0) + 1
            self.spawn_failed[track.id] = n
            if n == 1:
                self.get_logger().warn(f'spawn failed for {name}')
            if sum(self.spawn_failed.values()) >= 6:
                self.gazebo_ok = False
                self.get_logger().warn(
                    'Gazebo unreachable - continuing without the 3D twin. '
                    'Tracking and rviz markers are unaffected.')

    def move(self, track):
        name = self.spawned.get(track.id)
        if not name: return
        p = pose_pb2.Pose(); p.name = name
        pos = track.position
        p.position.x, p.position.y, p.position.z = map(float, pos)
        w, x, y, z = dir_to_quat(track.dir_world)
        p.orientation.w, p.orientation.x = w, x
        p.orientation.y, p.orientation.z = y, z
        if not self.gazebo_ok:
            return
        try:
            self.gz.request('/world/empty/set_pose', p,
                            pose_pb2.Pose, bool_pb2.Boolean, 150)
        except Exception:
            pass

    def remove(self, tid):
        """B23: delete the Gazebo model too, not just the bookkeeping entry.
        Otherwise dead tracks leave ghost instruments frozen in the scene."""
        name = self.spawned.pop(tid, None)
        self.spawn_failed.pop(tid, None)
        if name and self.gazebo_ok:
            try:
                ent = entity_pb2.Entity()
                ent.name = name
                ent.type = entity_pb2.Entity.MODEL
                self.gz.request('/world/empty/remove', ent,
                                entity_pb2.Entity, bool_pb2.Boolean, 150)
            except Exception:
                pass

    # ------------------------------------------------------------------
    def callback(self, msg):
        data  = json.loads(msg.data)
        fid   = data.get('frame_id', 0)
        stamp = time.monotonic()

        # B7: sequence looped -> reset all path lengths
        if fid < self.last_frame_id:
            for t in self.tracks: t.path_len = 0.0
            self.get_logger().info('sequence loop detected — EoM reset')
        self.last_frame_id = fid

        # R3: keep low-confidence detections; they are tracked and displayed
        # as UNKNOWN rather than dropped. Silently discarding an unrecognised
        # instrument is the worst failure mode for a safety-framed system.
        dets = [d for d in data.get('poses', []) if d.get('confidence', 0) >= 0.25]
        for t in self.tracks: t.predict(stamp)

        # B6: greedy nearest-neighbour association within the gate
        unmatched = list(range(len(dets)))
        used = set()
        pairs = []
        for ti, t in enumerate(self.tracks):
            for di in unmatched:
                p = dets[di]['position_3d']
                d = float(np.linalg.norm(
                    t.position - np.array([p['x'], p['y'], p['z']])))
                if d <= GATE_M:
                    # class agreement halves the effective cost
                    cost = d * (0.5 if dets[di]['class_id'] == t.class_id else 1.0)
                    pairs.append((cost, ti, di))
        pairs.sort()
        assigned_t, assigned_d = set(), set()
        for cost, ti, di in pairs:
            if ti in assigned_t or di in assigned_d: continue
            p = dets[di]['position_3d']
            self.tracks[ti].update([p['x'], p['y'], p['z']],
                                   dets[di]['class_id'],
                                   dets[di].get('shaft_dir_world'),
                                   dets[di].get('identity', 'known'))
            assigned_t.add(ti); assigned_d.add(di)

        # unmatched detections -> new tracks
        for di, d in enumerate(dets):
            if di in assigned_d: continue
            p = d['position_3d']
            self.tracks.append(Track([p['x'], p['y'], p['z']],
                                     d['class_id'], d['class_name'], stamp,
                                     d.get('identity', 'known')))

        # unmatched tracks -> coast, possibly die
        alive = []
        for ti, t in enumerate(self.tracks):
            if ti in assigned_t or t.hits == 1:
                alive.append(t); continue
            if t.mark_missed():
                self.remove(t.id)          # B7: path length dies with track
            else:
                alive.append(t)
        self.tracks = alive

        for t in self.tracks:
            if not t.confirmed: continue
            if t.id not in self.spawned: self.spawn(t)
            self.move(t)

        out = String()
        out.data = json.dumps({
            'frame_id': fid,
            'tracks': [{
                'track_id':   t.id,
                'class_name': self.display_name(t),
                'identity':   t.identity,
                'predicted_class': self.class_names[t.class_id],
                'position':   [round(float(v), 5) for v in t.position],
                'speed_mps':  round(t.speed, 4),
                'path_len_m': round(t.path_len, 5),
                'hits': t.hits, 'misses': t.misses,
                'shaft_dir':  [round(float(v), 5) for v in t.dir_world],
            } for t in self.tracks if t.confirmed]})
        self.filtered_pub.publish(out)

        if fid % 50 == 0 and self.tracks:
            for t in self.tracks:
                if t.confirmed:
                    self.get_logger().info(
                        f'  track {t.id} {self.display_name(t)}: '
                        f'path {t.path_len*1000:.1f} mm, '
                        f'speed {t.speed*1000:.1f} mm/s')


def main(args=None):
    rclpy.init(args=args)
    node = TwinSyncNode()
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
