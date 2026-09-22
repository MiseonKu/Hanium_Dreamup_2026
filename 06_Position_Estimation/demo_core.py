"""Five-point static correction experiment; no hardware imports or side effects."""
from __future__ import annotations
import hashlib
import json
from itertools import product
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
AXES = ('shoulder_pan', 'shoulder_lift', 'wrist_flex')
SENSORS = ('imu0_base', 'imu1_link_root', 'imu2_link_mid', 'imu3_tip')
REF = np.array([2055., 2093., 1970.])
RAD_PER_TICK = 2 * np.pi / 4096

class DemoError(RuntimeError):
    pass

def finite(value, shape):
    a = np.asarray(value, dtype=float)
    if a.shape != shape or not np.isfinite(a).all():
        raise DemoError(f'Expected finite shape {shape}, got {a.shape}')
    return a

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

class Model:
    """Exact exported three-seed MLP; inference needs NumPy only, no pickle."""
    def __init__(self, directory=ROOT / 'data'):
        directory = Path(directory)
        self.meta = read_json(directory / 'model.json')
        payload = (directory / 'weights.npz').read_bytes()
        if hashlib.sha256(payload).hexdigest() != self.meta['weights_sha256']:
            raise DemoError('Model weight checksum mismatch')
        with np.load(directory / 'weights.npz', allow_pickle=False) as z:
            self.w = {name: z[name].copy() for name in z.files}
        self.names = self.meta['feature_names']
        if len(self.names) != 43 or any(n.startswith(('vive_', 'planned_')) for n in self.names):
            raise DemoError('Unexpected feature contract')

    def predict(self, features, check_envelope=False):
        if set(features) != set(self.names):
            raise DemoError('Exactly 43 named features required')
        x = finite([features[n] for n in self.names], (43,))
        if check_envelope:
            # Coarse historical envelope, NOT calibrated uncertainty or OOD detection.
            lo, hi = self.w['feature_min'], self.w['feature_max']
            tolerance = np.maximum((hi - lo) * .05, 1e-8)
            outside = (x < lo - tolerance) | (x > hi + tolerance)
            if outside.any():
                raise DemoError('Outside historical feature envelope: ' + ', '.join(np.array(self.names)[outside]))
        predictions = []
        for i in range(3):
            a = (x - self.w[f'mean_{i}']) / self.w[f'scale_{i}']
            for j in range(3):
                a = a @ self.w[f'w_{i}_{j}'] + self.w[f'b_{i}_{j}']
                if j < 2:
                    a = np.maximum(a, 0)
            predictions.append(a * self.w[f'yscale_{i}'] + self.w[f'ymean_{i}'])
        return finite(np.mean(predictions, axis=0), (3,))

def fk(raw, geometry):
    """3-axis FK; optional effective joint offsets do not change ML inputs."""
    q = (finite(raw, (3,)) - REF) * RAD_PER_TICK + [0, np.pi / 2, 0]
    q = q + finite(geometry.get('joint_angle_offsets_rad', [0, 0, 0]), (3,))
    a, b, c = q
    l1, l2 = finite(geometry['link_lengths_mm'], (2,))
    r = l1 * np.cos(b) + l2 * np.cos(b + c)
    p = np.array([np.cos(a) * r, np.sin(a) * r,
                  l1 * np.sin(b) + l2 * np.sin(b + c)])
    # Tip local X along link; Y lateral; Z = X cross Y.
    x = np.array([np.cos(a)*np.cos(b+c), np.sin(a)*np.cos(b+c), np.sin(b+c)])
    y = np.array([-np.sin(a), np.cos(a), 0.])
    orientation = np.column_stack((x, y, np.cross(x, y)))
    return p + orientation @ finite(geometry['marker_offset_tip_mm'], (3,))

def jacobian(raw, geometry):
    raw = finite(raw, (3,))
    return np.column_stack([(fk(raw + np.eye(3)[j]*.1, geometry) -
                             fk(raw - np.eye(3)[j]*.1, geometry))/.2 for j in range(3)])

def transform(config):
    f = config['frame']
    r = finite(f['world_from_model_rotation'], (3, 3))
    t = finite(f['world_from_model_translation_mm'], (3,))
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-6) or abs(np.linalg.det(r)-1) > 1e-6:
        raise DemoError('Frame requires a proper rigid rotation (no scale/reflection)')
    return r, t

def validate_config(c, targets, live=False):
    mode = c.get('mode', 'independent_calibration')
    if mode not in ('independent_calibration', 'approximate_dataset_demo'):
        raise DemoError('Unknown reference mode')
    if mode == 'approximate_dataset_demo':
        if c.get('approximation', {}).get('accepted_for_demo') is not True:
            raise DemoError('Approximate dataset reference must be explicitly accepted for this demo')
        transform(c)
        lengths = finite(c['geometry']['link_lengths_mm'], (2,))
        offsets = finite(c['geometry']['joint_angle_offsets_rad'], (3,))
        if np.any(lengths <= 0) or np.any(np.abs(offsets) > np.pi):
            raise DemoError('Invalid effective kinematics')
        finite(c['geometry']['marker_offset_tip_mm'], (3,))
    if len(targets) != 5 or len({t['target_id'] for t in targets}) != 5:
        raise DemoError('Exactly five distinct targets required')
    limits = finite(c['hardware']['raw_limits'], (3, 2))
    if (limits[:, 0] >= limits[:, 1]).any() or (limits < 0).any() or (limits > 4095).any():
        raise DemoError('Invalid raw limits')
    settings = c['control']
    if any(k in settings for k in ('gain','max_correction_ticks','max_cartesian_step_mm','deadband_mm')):
        raise DemoError('Old partial-correction config: use the full-error configuration')
    bounds = {'damping_mm_per_tick': (0.001, 10),
              'max_total_correction_ticks': (1, 500), 'ik_step_ticks': (1, 50),
              'ik_tolerance_mm': (.1, 1), 'ik_max_iterations': (10, 300),
              'max_initial_error_mm': (1, 150),
              'pause_s': (2, 30), 'summary_s': (2.9, 3.1),
              'settle_timeout_s': (5, 60), 'axis_gap_s': (.6, 5),
              'max_transit_step_ticks': (1, 50), 'arrival_tolerance_ticks': (1, 10),
              'max_gyro_rms_rad_s': (.001, .05), 'max_encoder_span_ticks': (1, 10)}
    for name, (lo, hi) in bounds.items():
        v = float(settings[name])
        if not np.isfinite(v) or not lo <= v <= hi:
            raise DemoError(f'{name} must be within [{lo}, {hi}]')
    for t in targets:
        raw = finite(t['raw_ticks'], (3,))
        if not np.equal(raw, np.round(raw)).all():
            raise DemoError('Target ticks must be integers')
        if np.any(raw < limits[:, 0] + 40) or np.any(raw > limits[:, 1] - 40):
            raise DemoError('Target too close to active range edge')
    if not live:
        return
    checks = c['commissioning']
    for name in ('same_training_installation_verified', 'imu_mapping_and_scales_verified',
                 'three_axis_rig_verified',
                 'path_clearance_and_external_stop_verified'):
        if checks.get(name) is not True:
            raise DemoError(f'Live mode blocked: commissioning.{name} is not true')
    if mode == 'independent_calibration' and checks.get('frame_and_marker_calibration_verified') is not True:
        raise DemoError('Live mode blocked: commissioning.frame_and_marker_calibration_verified is not true')
    if not checks.get('calibration_record') or not checks.get('operator'):
        raise DemoError('Record calibration evidence and operator before live use')
    transform(c)
    lengths = finite(c['geometry']['link_lengths_mm'], (2,))
    if np.any(lengths <= 0):
        raise DemoError('Invalid link lengths')
    finite(c['geometry']['marker_offset_tip_mm'], (3,))
    ports = [c['hardware']['esp32_port'], *c['hardware']['imu_ports'].values()]
    if set(c['hardware']['imu_ports']) != set(SENSORS) or len(set(ports)) != 5 or any(not p for p in ports):
        raise DemoError('Five unique verified serial ports required')
    if c['hardware']['servo_ids'] != [1, 2, 4]:
        raise DemoError('This package targets historical 3-axis IDs 1,2,4 only')
    if not 1 <= c['hardware']['speed'] <= 120 or not 1 <= c['hardware']['acceleration'] <= 10:
        raise DemoError('Speed/acceleration exceed demo limits')

def target_world(target, c):
    r, translation = transform(c)
    if c.get('mode') == 'approximate_dataset_demo':
        # The fitted R/t maps effective FK, NOT the old nominal planned XYZ.
        # This is a dataset-derived reference, not an independently known sag-free goal.
        return r @ fk(target['raw_ticks'], c['geometry']) + translation
    return r @ (finite(target['p_xyz_mm'], (3,)) +
                   fk(target['raw_ticks'], c['geometry']) -
                   fk(target['raw_ticks'], {**c['geometry'], 'marker_offset_tip_mm': [0,0,0]})) + translation


def correction(raw, prediction_world, target, c):
    r, translation = transform(c)
    raw = finite(raw, (3,))
    desired = target_world(target, c)
    error = desired - finite(prediction_world, (3,))
    norm = float(np.linalg.norm(error))
    s = c['control']
    if norm > s['max_initial_error_mm']:
        raise DemoError(f'Position discrepancy {norm:.1f} mm exceeds configured bound')
    if norm == 0:
        return raw.round().astype(int), {'status':'already_at_target', 'desired_world_mm':desired.tolist(),
            'error_mm':error.tolist(), 'requested_correction_world_mm':error.tolist()}
    # Preserve 100% of the Cartesian error. Solver increments and physical
    # transit steps limit individual steps, NEVER shrink the final request.
    rigid_start = r @ fk(raw, c['geometry']) + translation
    rigid_goal = rigid_start + error
    limits = np.array(c['hardware']['raw_limits'],float)
    lower = np.maximum(limits[:,0]+40, raw-s['max_total_correction_ticks'])
    upper = np.minimum(limits[:,1]-40, raw+s['max_total_correction_ticks'])
    q = raw.copy()
    for iteration in range(int(s['ik_max_iterations'])):
        residual = rigid_goal - (r @ fk(q,c['geometry']) + translation)
        if np.linalg.norm(residual) <= s['ik_tolerance_mm']:
            break
        j = r @ jacobian(q,c['geometry'])
        singular = np.linalg.svd(j,compute_uv=False)
        if singular[-1]<.015 or singular[0]/singular[-1]>40:
            raise DemoError('Full-error IK encountered near-singular geometry; no partial correction sent')
        delta = j.T @ np.linalg.solve(j@j.T + np.eye(3)*s['damping_mm_per_tick']**2,residual)
        delta *= min(1.,s['ik_step_ticks']/max(np.max(np.abs(delta)),1e-12))
        # Backtracking occurs offline. No motor command until the entire goal is solved.
        accepted = False
        for fraction in (1.,.5,.25,.125,.0625):
            candidate = q + fraction*delta
            if np.any(candidate<lower) or np.any(candidate>upper):
                continue
            new_residual = rigid_goal-(r@fk(candidate,c['geometry'])+translation)
            if np.linalg.norm(new_residual)<np.linalg.norm(residual)-1e-8:
                q = candidate
                accepted = True
                break
        if not accepted:
            raise DemoError('Full-error IK cannot progress within joint/correction limits; no partial correction sent')
    # Choose the best nearby integer-tick solution, even for sub-tolerance
    # requests. Solver tolerance must not become a hidden movement deadband.
    center = np.rint(q).astype(int)
    choices = []
    for offset in product((-1,0,1),repeat=3):
        candidate = center + offset
        if np.any(candidate<lower) or np.any(candidate>upper):
            continue
        difference = r@fk(candidate,c['geometry'])+translation-rigid_start-error
        choices.append((float(np.linalg.norm(difference)),float(np.linalg.norm(candidate-raw)),candidate))
    if not choices:
        raise DemoError('No integer-tick solution within limits; no partial correction sent')
    remaining,_,command = min(choices,key=lambda item:(item[0],item[1]))
    if np.array_equal(command,raw) and norm <= s['ik_tolerance_mm']:
        return command, {'status':'below_tick_resolution', 'desired_world_mm':desired.tolist(),
            'error_mm':error.tolist(), 'requested_correction_world_mm':error.tolist(),
            'rigid_predicted_displacement_world_mm':[0,0,0], 'rigid_ik_residual_mm':norm,
            'delta_ticks':[0,0,0]}
    realized_rigid_delta = r@fk(command,c['geometry'])+translation-rigid_start
    remaining = float(np.linalg.norm(error-realized_rigid_delta))
    if (remaining>s['ik_tolerance_mm'] or np.any(command<lower) or np.any(command>upper)):
        raise DemoError(f'Full-error IK unsolved after tick quantization: residual {remaining:.3f} mm; no partial correction sent')
    return command, {'status':'full_error_correction', 'desired_world_mm':desired.tolist(),
                     'error_mm':error.tolist(), 'requested_correction_world_mm':error.tolist(),
                     'rigid_predicted_displacement_world_mm':realized_rigid_delta.tolist(),
                     'rigid_ik_residual_mm':remaining, 'ik_iterations':iteration+1,
                     'delta_ticks':(command-raw).tolist()}

def summarise(imu, encoder, c):
    """Arithmetic Euler mean THEN sin/cos, matching historical build_trials.py."""
    enc = np.asarray(encoder, float)
    if enc.ndim != 2 or enc.shape[1] != 3 or len(enc) < 4 or not np.isfinite(enc).all():
        raise DemoError('Insufficient encoder samples')
    if np.max(np.ptp(enc, axis=0)) > c['control']['max_encoder_span_ticks']:
        raise DemoError('Encoder moved during summary window')
    features = {f'encoder_delta_rad_j{j}': float((enc[:,j].mean()-REF[j])*RAD_PER_TICK) for j in range(3)}
    counts = {}
    for sensor in SENSORS:
        samples = imu[sensor]
        arrays = {}
        for key in ('angle', 'acceleration', 'angular_velocity'):
            a = np.asarray(samples[key], float)
            if a.ndim != 2 or a.shape[1] != 3 or len(a) < 28 or not np.isfinite(a).all():
                raise DemoError(f'{sensor}/{key}: at least 28 fresh samples needed')
            arrays[key] = a
        if np.any(np.ptp(arrays['angle'], axis=0) > .15):
            raise DemoError(f'{sensor}: Euler wrap or motion within static window')
        rms = float(np.sqrt(np.mean(np.sum(arrays['angular_velocity']**2, axis=1))))
        if rms > c['control']['max_gyro_rms_rad_s']:
            raise DemoError(f'{sensor}: gyro is not settled')
        for j, axis in enumerate('xyz'):
            angle = arrays['angle'][:,j].mean()
            features[f'{sensor}_euler_{axis}_rad_sin'] = float(np.sin(angle))
            features[f'{sensor}_euler_{axis}_rad_cos'] = float(np.cos(angle))
            features[f'{sensor}_accel_{axis}_mps2'] = float(arrays['acceleration'][:,j].mean())
        features[f'{sensor}_gyro_rms_rad_s'] = rms
        counts[sensor] = {k: len(a) for k,a in arrays.items()}
    return features, {'encoder_n': len(enc), 'imu_n': counts}
