"""
Digital twin synchronisation — multi-object tracking and Gazebo sync.

Rewritten from scratch. The previous version accumulated many incremental
patches and had a defective cleanup path: tracks could leave self.tracks
without their Gazebo model being deleted, orphaning models permanently.

Design:
  * Single continuous sequence assumed. Loop detection resets Economy of
    Motion; there is no cross-sequence track carrying.
  * Constant-velocity Kalman filter per track, dt from real timestamps.
  * Greedy nearest-neighbour association with an ANISOTROPIC gate (depth
    is an order of magnitude noisier than lateral position) that scales
    with elapsed time, since frames drop irregularly.
  * RECONCILIATION: at the end of every callback, any Gazebo model that
    does not correspond to a live track is removed. Ghost models are
    therefore impossible regardless of errors elsewhere.
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import gz.transport13
import gz.msgs10.entity_factory_pb2 as ef_pb2
import gz.msgs10.entity_pb2 as entity_pb2
import gz.msgs10.boolean_pb2 as bool_pb2
import gz.msgs10.pose_pb2 as pose_pb2
import numpy as np
import json, math, time

# ---- tracking parameters -------------------------------------------------
GATE_LATERAL_M = 0.020   # lateral: mask centroid, precise
GATE_DEPTH_M   = 0.060   # depth: stereo, noisy
GATE_MAX_SCALE = 3.0     # max gate expansion for long frame gaps
NOMINAL_DT     = 0.15    # s, expected interval between processed frames
CONFIRM_HITS   = 2       # matches before a KNOWN track is spawned
CONFIRM_UNKNOWN = 5      # unknowns need more evidence: at conf 0.25-0.55
                         # the detector also fires on tissue folds and
                         # specular highlights, and each would spawn a track
MAX_MISSES     = 10      # frames coasting before a track dies
# Duplicate suppression must be at least as permissive as the association
# gate, otherwise a detection can fail association (on depth, gate 60mm)
# yet still pass suppression (45mm) and spawn a redundant track.
DUP_GATE_LATERAL_M = 0.025
DUP_GATE_DEPTH_M   = 0.070
COAST_BLOCK_M      = 0.055   # no new track near a coasting track of same class
MIN_CONF       = 0.12    # matches perception_node's floor; see note there

CLASS_NAMES = ['Large_Needle_Driver_Left','Large_Needle_Driver_Right',
               'Prograsp_Forceps_Left','Prograsp_Forceps_Right',
               'Maryland_Bipolar_Forceps','Bipolar_Forceps',
               'Monopolar_Curved_Scissors','Grasping_Retractor_Right']
PALETTE = [(0.2,0.6,1.0),(1.0,0.4,0.2),(0.2,1.0,0.4),(1.0,0.2,0.8),
           (0.8,0.8,0.2),(0.6,0.2,1.0),(0.2,0.8,0.8),(1.0,0.6,0.2)]


class Track:
    """Constant-velocity Kalman track. State = [x,y,z,vx,vy,vz] (world m)."""
    _next_id = 0

    def __init__(self, pos, class_id, identity, stamp):
        self.id = Track._next_id; Track._next_id += 1
        self.class_id = class_id
        self.class_votes = {class_id: 1}
        self.identity = identity
        self.id_votes = {identity: 1}

        self.x = np.array([*pos, 0.0, 0.0, 0.0], dtype=np.float64)
        self.P = np.diag([1e-3]*3 + [1e-2]*3)
        self.H = np.hstack([np.eye(3), np.zeros((3,3))])
        # Tuned by ablation over 737 live pose samples across 5 instruments.
        # Previous values (q=5e-3, r=2e-4) gave process noise 25x measurement
        # noise, so the filter almost passed measurements through: 15% jitter
        # reduction only. These give 34% at 6.2mm lag. Heavier settings reach
        # 73% but at 14.5mm lag -- a quarter of the working distance.
        # Note a 5-frame moving average performs equivalently to the heavy
        # Kalman configuration (4.69 vs 4.24mm jitter, 14.59 vs 14.54mm lag),
        # so the Kalman formulation is not itself the source of the benefit.
        self.q, self.r = 5e-3, 1e-3

        self.hits, self.misses = 1, 0
        self.confirmed = False
        self.path_len  = 0.0
        self.last_pos  = np.array(pos, dtype=np.float64)
        self.stamp     = stamp
        self.stamp_prev = stamp
        self.dir_world = np.array([1.0, 0.0, 0.0])
        self.axis_len_px = None

    def predict(self, stamp):
        dt = max(1e-3, min(0.5, stamp - self.stamp))
        self.stamp_prev, self.stamp = self.stamp, stamp
        F = np.eye(6); F[0,3] = F[1,4] = F[2,5] = dt
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + np.eye(6) * self.q * dt
        return self.x[:3]

    def update(self, pos, class_id, identity, dir_world, axis_len):
        z = np.asarray(pos, dtype=np.float64)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + np.eye(3) * self.r
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x += K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

        if self.confirmed:
            self.path_len += float(np.linalg.norm(self.x[:3] - self.last_pos))
        self.last_pos = self.x[:3].copy()

        # majority votes: immune to single-frame label or confidence flips
        self.class_votes[class_id] = self.class_votes.get(class_id, 0) + 1
        self.class_id = max(self.class_votes, key=self.class_votes.get)
        self.id_votes[identity] = self.id_votes.get(identity, 0) + 1
        self.identity = max(self.id_votes, key=self.id_votes.get)

        if dir_world is not None:
            self.dir_world = np.asarray(dir_world, dtype=np.float64)
        if axis_len is not None:
            self.axis_len_px = axis_len

        self.hits += 1
        self.misses = 0
        need = CONFIRM_UNKNOWN if self.identity == 'unknown' else CONFIRM_HITS
        if self.hits >= need:
            self.confirmed = True

    def mark_missed(self):
        self.misses += 1
        return self.misses > MAX_MISSES

    @property
    def position(self): return self.x[:3]
    @property
    def speed(self):    return float(np.linalg.norm(self.x[3:]))
    @property
    def display_name(self):
        return 'UNKNOWN' if self.identity == 'unknown' else CLASS_NAMES[self.class_id]
    @property
    def model_name(self):
        return f'instrument_{self.id}_{self.display_name}'


def dir_to_quat(d):
    """Quaternion (w,x,y,z) rotating model +Z onto world direction d."""
    d = np.asarray(d, dtype=np.float64)
    n = np.linalg.norm(d)
    if n < 1e-9: return (1.0, 0.0, 0.0, 0.0)
    d = d / n
    c = float(d[2])                      # dot([0,0,1], d)
    if c < -0.999999: return (0.0, 1.0, 0.0, 0.0)
    v = np.cross([0.0, 0.0, 1.0], d)
    s = math.sqrt((1.0 + c) * 2.0)
    return (s * 0.5, v[0]/s, v[1]/s, v[2]/s)


# Geometry per class. Classes 0 and 1 (Large Needle Driver) use the dVRK
# CAD mesh, cropped to the usable head region. Remaining classes use
# parametric geometry: borrowing a needle-driver mesh for a different
# instrument type would misrepresent what the system detected.
MESH_CLASSES = {0, 1}

SDF_MESH = """<?xml version="1.0" ?>
<sdf version="1.8">
  <model name="{name}">
    <static>true</static>
    <link name="head">
      <visual name="v">
        <geometry><mesh><uri>model://instruments/meshes/shaft_head.obj</uri></mesh></geometry>
        <material>
          <ambient>0.50 0.52 0.55 1</ambient><diffuse>0.72 0.75 0.79 1</diffuse>
          <specular>0.95 0.95 0.95 1</specular>
        </material>
      </visual>
    </link>
    <link name="jaw_l"><visual name="v">
      <geometry><mesh><uri>model://instruments/meshes/gripper_left.OBJ</uri></mesh></geometry>
      <material><ambient>0.60 0.62 0.65 1</ambient><diffuse>0.82 0.84 0.88 1</diffuse>
      <specular>1 1 1 1</specular></material>
    </visual></link>
    <link name="jaw_r"><visual name="v">
      <geometry><mesh><uri>model://instruments/meshes/gripper_right.OBJ</uri></mesh></geometry>
      <material><ambient>0.60 0.62 0.65 1</ambient><diffuse>0.82 0.84 0.88 1</diffuse>
      <specular>1 1 1 1</specular></material>
    </visual></link>
    <link name="shaft">
      <pose>0 0 -0.055 0 0 0</pose>
      <visual name="v">
        <geometry><cylinder><radius>0.0032</radius><length>0.075</length></cylinder></geometry>
        <material>
          <ambient>0.09 0.09 0.10 1</ambient><diffuse>0.15 0.15 0.17 1</diffuse>
          <specular>0.6 0.6 0.6 1</specular>
        </material>
      </visual>
    </link>
    <link name="band">
      <pose>0 0 -0.021 0 0 0</pose>
      <visual name="v">
        <geometry><cylinder><radius>0.0034</radius><length>0.004</length></cylinder></geometry>
        <material><ambient>{r} {g} {b} 1</ambient><diffuse>{r} {g} {b} 1</diffuse></material>
      </visual>
    </link>
  </model>
</sdf>"""

SDF_PRIM = """<?xml version="1.0" ?>
<sdf version="1.8">
  <model name="{name}">
    <static>true</static>
    <link name="jaws">
      <visual name="v">
        <geometry><sphere><radius>0.0045</radius></sphere></geometry>
        <material>
          <ambient>0.65 0.66 0.68 1</ambient><diffuse>0.85 0.86 0.90 1</diffuse>
          <specular>1 1 1 1</specular>
        </material>
      </visual>
    </link>
    <link name="wrist">
      <pose>0 0 -0.008 0 0 0</pose>
      <visual name="v">
        <geometry><cylinder><radius>0.0035</radius><length>0.012</length></cylinder></geometry>
        <material>
          <ambient>0.55 0.56 0.58 1</ambient><diffuse>0.75 0.76 0.80 1</diffuse>
          <specular>0.9 0.9 0.9 1</specular>
        </material>
      </visual>
    </link>
    <link name="shaft">
      <pose>0 0 -0.055 0 0 0</pose>
      <visual name="v">
        <geometry><cylinder><radius>0.0030</radius><length>0.075</length></cylinder></geometry>
        <material>
          <ambient>0.09 0.09 0.10 1</ambient><diffuse>0.15 0.15 0.17 1</diffuse>
          <specular>0.6 0.6 0.6 1</specular>
        </material>
      </visual>
    </link>
    <link name="band">
      <pose>0 0 -0.021 0 0 0</pose>
      <visual name="v">
        <geometry><cylinder><radius>0.0033</radius><length>0.004</length></cylinder></geometry>
        <material><ambient>{r} {g} {b} 1</ambient><diffuse>{r} {g} {b} 1</diffuse></material>
      </visual>
    </link>
  </model>
</sdf>"""


class TwinSyncNode(Node):
    def __init__(self):
        super().__init__('twin_sync_node')
        self.declare_parameter('world', 'empty')
        self.world = self.get_parameter('world').value

        self.gz = gz.transport13.Node()
        self.gazebo_ok  = True
        self.fail_count = 0

        self.tracks  = []      # live tracks
        self.spawned = {}      # track_id -> gazebo model name  (INVARIANT:
                               # keys must always be a subset of live ids)
        self.last_src_frame = None
        self.pending_removal = set()   # models Gazebo did not confirm

        self.pub = self.create_publisher(String, '/instrument_poses_filtered', 10)
        self.create_subscription(String, '/instrument_poses_3d', self.callback, 10)
        self.create_subscription(String, '/frame_source', self.source_callback, 10)

        self.get_logger().info(
            f'Twin sync ready (world={self.world})\n'
            f'  gate lateral {GATE_LATERAL_M*1000:.0f}mm / '
            f'depth {GATE_DEPTH_M*1000:.0f}mm, scaled up to {GATE_MAX_SCALE:.0f}x\n'
            f'  confirm {CONFIRM_HITS} hits, die after {MAX_MISSES} misses')

    # ---- Gazebo -----------------------------------------------------
    def _gz_request(self, service, req, req_t, timeout=400):
        if not self.gazebo_ok:
            return False
        try:
            ok = self.gz.request(f'/world/{self.world}/{service}',
                                 req, req_t, bool_pb2.Boolean, timeout)
        except Exception:
            ok = False
        if not ok:
            self.fail_count += 1
            if self.fail_count == 8:
                self.gazebo_ok = False
                self.get_logger().warn(
                    'Gazebo unreachable — continuing without the 3D twin. '
                    'Tracking and rviz markers are unaffected.')
        return ok

    def _spawn(self, track):
        r, g, b = PALETTE[track.class_id % len(PALETTE)] \
                  if track.identity != 'unknown' else (0.55, 0.55, 0.55)
        req = ef_pb2.EntityFactory()
        tpl = SDF_MESH if (track.class_id in MESH_CLASSES
                           and track.identity != 'unknown') else SDF_PRIM
        req.sdf  = tpl.format(name=track.model_name, r=r, g=g, b=b)
        req.name = track.model_name
        req.pose.position.z = 0.5
        if self._gz_request('create', req, ef_pb2.EntityFactory, 800):
            self.spawned[track.id] = track.model_name

    def _remove_model(self, name):
        ent = entity_pb2.Entity()
        ent.name = name
        ent.type = entity_pb2.Entity.MODEL
        return self._gz_request('remove', ent, entity_pb2.Entity, 500)

    def _despawn(self, tid):
        """Remove a track's Gazebo model.

        The bookkeeping entry is only dropped once Gazebo confirms removal.
        Popping first meant a failed/timed-out request orphaned the model:
        it stayed in the scene but was no longer known to reconciliation,
        so it could never be cleaned up. Those are the frozen instruments.
        """
        name = self.spawned.get(tid)
        if not name:
            return
        if self._remove_model(name):
            self.spawned.pop(tid, None)
            self.pending_removal.discard(name)
        else:
            # keep retrying on subsequent frames
            self.spawned.pop(tid, None)
            self.pending_removal.add(name)

    def _move(self, track):
        if track.id not in self.spawned:
            return
        p = pose_pb2.Pose()
        p.name = self.spawned[track.id]
        p.position.x, p.position.y, p.position.z = map(float, track.position)
        w, x, y, z = dir_to_quat(track.dir_world)
        p.orientation.w, p.orientation.x = w, x
        p.orientation.y, p.orientation.z = y, z
        self._gz_request('set_pose', p, pose_pb2.Pose, 150)

    # ---- sequence / loop --------------------------------------------
    def source_callback(self, msg):
        """Reset Economy of Motion when the sequence loops."""
        try:
            f = json.loads(msg.data).get('src_frame')
        except Exception:
            return
        if f is None:
            return
        if self.last_src_frame is not None and f < self.last_src_frame:
            # A loop is a hard discontinuity: instruments jump from their
            # end-of-sequence positions back to their start positions.
            # Carrying tracks across it guarantees association failure and
            # spurious track creation, so clear them.
            n = len(self.tracks)
            for t in list(self.tracks):
                self._despawn(t.id)
            self.tracks = []
            self.get_logger().info(
                f'sequence looped — cleared {n} tracks, EoM reset')
        self.last_src_frame = f

    # ---- main -------------------------------------------------------
    def callback(self, msg):
        data  = json.loads(msg.data)
        stamp = time.monotonic()

        # a pose without metric depth cannot be placed in the 3D twin
        dets = [d for d in data.get('poses', [])
                if d.get('confidence', 0) >= MIN_CONF
                and d.get('position_3d') is not None]

        for t in self.tracks:
            t.predict(stamp)

        # ---- association: greedy nearest neighbour within gate ----
        pairs = []
        for ti, t in enumerate(self.tracks):
            k = min(GATE_MAX_SCALE,
                    max(1.0, (stamp - t.stamp_prev) / NOMINAL_DT))
            for di, d in enumerate(dets):
                p  = d['position_3d']
                dv = t.position - np.array([p['x'], p['y'], p['z']])
                lat   = float(np.hypot(dv[1], dv[2]))
                depth = abs(float(dv[0]))          # world X == optical Z
                if lat <= GATE_LATERAL_M * k and depth <= GATE_DEPTH_M * k:
                    # Association is purely geometric. A class-match
                    # discount was previously applied, but with two
                    # physically identical instruments present (both Large
                    # Needle Drivers in this sequence) it biases assignment
                    # toward whichever track shares the label rather than
                    # whichever is actually closer.
                    cost = lat
                    pairs.append((cost, ti, di))
        pairs.sort()

        used_t, used_d = set(), set()
        for _, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            d = dets[di]; p = d['position_3d']
            self.tracks[ti].update(
                [p['x'], p['y'], p['z']], d['class_id'],
                d.get('identity', 'known'), d.get('shaft_dir_world'),
                d.get('axis_len_px'))
            used_t.add(ti); used_d.add(di)

        # ---- unmatched detections -> new tracks ----
        for di, d in enumerate(dets):
            if di in used_d:
                continue
            p  = d['position_3d']
            pv = np.array([p['x'], p['y'], p['z']])
            # suppress duplicates: association can fail transiently on depth
            # noise; without this, each failure spawns a permanent duplicate
            blocked = False
            for t in self.tracks:
                dv  = t.position - pv
                lat = float(np.hypot(dv[1], dv[2]))
                dep = abs(float(dv[0]))
                if lat < DUP_GATE_LATERAL_M and dep < DUP_GATE_DEPTH_M:
                    blocked = True; break
                # a track that just lost its detection will usually reacquire
                # it; spawning a sibling in the meantime creates a permanent
                # duplicate once both are confirmed
                if (t.misses > 0 and t.class_id == d['class_id']
                        and lat < COAST_BLOCK_M):
                    blocked = True; break
            if blocked:
                continue
            self.tracks.append(Track(pv, d['class_id'],
                                     d.get('identity', 'known'), stamp))

        # ---- unmatched tracks coast, then die ----
        survivors = []
        for ti, t in enumerate(self.tracks):
            if ti in used_t or t.hits == 1:
                survivors.append(t)
            elif t.mark_missed():
                self._despawn(t.id)
            else:
                survivors.append(t)
        self.tracks = survivors

        # ---- spawn / move confirmed tracks ----
        for t in self.tracks:
            if not t.confirmed:
                continue
            if t.id not in self.spawned:
                self._spawn(t)
            self._move(t)

        # ---- RECONCILE: no Gazebo model may outlive its track ----
        live = {t.id for t in self.tracks}
        for tid in [k for k in self.spawned if k not in live]:
            self._despawn(tid)

        # retry removals Gazebo did not confirm, so a dropped request
        # cannot leave a permanently frozen model in the scene
        if self.pending_removal:
            for name in list(self.pending_removal):
                if self._remove_model(name):
                    self.pending_removal.discard(name)

        out = String()
        out.data = json.dumps({
            'frame_id': data.get('frame_id', 0),
            'n_tracks': len(self.tracks),
            'tracks': [{
                'track_id':   t.id,
                'class_name': t.display_name,
                'identity':   t.identity,
                'predicted_class': CLASS_NAMES[t.class_id],
                'position':   [round(float(v), 5) for v in t.position],
                'shaft_dir':  [round(float(v), 5) for v in t.dir_world],
                'axis_len_px': t.axis_len_px,
                'speed_mps':  round(t.speed, 4),
                'path_len_m': round(t.path_len, 5),
                'hits': t.hits, 'misses': t.misses,
            } for t in self.tracks if t.confirmed]})
        self.pub.publish(out)

        if data.get('frame_id', 0) % 50 == 0:
            self.get_logger().info(
                f'{len(self.tracks)} tracks '
                f'({sum(1 for t in self.tracks if t.confirmed)} confirmed), '
                f'{len(self.spawned)} models, next id {Track._next_id}')


def main(args=None):
    rclpy.init(args=args)
    node = TwinSyncNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        for tid in list(node.spawned):
            node._despawn(tid)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
