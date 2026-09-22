import copy
import csv
import io
import json
from pathlib import Path
import struct
import tempfile
import threading
import time
from collections import deque
import unittest
from unittest.mock import patch
import numpy as np
from demo_core import (ROOT, Model, DemoError, read_json, correction, fk, jacobian,
                       validate_config, summarise, SENSORS, AXES, RAD_PER_TICK, REF)
from hardware import parse_state, extract_packets, Motor, IMUs

class DemoTests(unittest.TestCase):
    def setUp(self):
        self.cfg=read_json(ROOT/'config.json')
        self.targets=read_json(ROOT/'data/targets.json')

    def calibrated(self):
        c=copy.deepcopy(self.cfg)
        c['frame']={'world_from_model_rotation':np.eye(3).tolist(),'world_from_model_translation_mm':[10,20,30]}
        c['geometry']['marker_offset_tip_mm']=[0,0,0]
        for k,v in c['commissioning'].items():
            c['commissioning'][k]=True if isinstance(v,bool) else 'TEST FIXTURE ONLY'
        return c

    def state_text(self):
        text=(ROOT/'data/historical_state.txt').read_text(encoding='utf-8')
        # Recorded REST lay below the active lower bound, which this demo rejects.
        return text.replace('raw_pos=1184','raw_pos=1300')

    def test_original_model_fixture_parity(self):
        model=Model()
        with (ROOT/'data/inference_fixture.csv').open(newline='') as f:
            features=list(csv.DictReader(f))
        with (ROOT/'data/inference_fixture_predictions.csv').open(newline='') as f:
            predictions=list(csv.DictReader(f))
        self.assertEqual(len(features),32)
        for x,y in zip(features,predictions):
            actual=model.predict({n:float(x[n]) for n in model.names})
            np.testing.assert_allclose(actual,[float(y[f'pred_vive_world_{a}_mm']) for a in 'xyz'],atol=1e-7,rtol=0)

    def test_nominal_fk_reproduces_archive(self):
        c=self.calibrated()
        for t in self.targets:
            self.assertLess(np.linalg.norm(fk(t['raw_ticks'],c['geometry'])-t['p_xyz_mm']),1.)

    def test_jacobian_local_motion(self):
        c=self.calibrated(); t=self.targets[0]; q=np.array(t['raw_ticks'],float)
        change=np.array([.01,-.02,.03])
        np.testing.assert_allclose(jacobian(q,c['geometry'])@change,
            fk(q+change,c['geometry'])-fk(q,c['geometry']),atol=2e-6)

    def test_full_40mm_request_solved_or_explicitly_rejected(self):
        c=self.calibrated(); translation=np.array([10,20,30])
        solved=0; rejected=0
        for t in self.targets:
            raw=np.array(t['raw_ticks']); goal=np.array(t['p_xyz_mm'])+translation
            before=goal-[0,0,40]
            try:
                command,detail=correction(raw,before,t,c)
            except DemoError as exc:
                self.assertIn('no partial correction sent',str(exc));rejected+=1
                continue
            solved+=1
            np.testing.assert_allclose(detail['requested_correction_world_mm'],[0,0,40],atol=1e-10)
            actual_delta=fk(command,c['geometry'])-fk(raw,c['geometry'])
            self.assertLessEqual(np.linalg.norm(actual_delta-[0,0,40]),c['control']['ik_tolerance_mm'])
            self.assertLessEqual(np.max(np.abs(command-raw)),c['control']['max_total_correction_ticks'])
        self.assertGreater(solved,0);self.assertGreater(rejected,0)

    def test_limit_does_not_shrink_requested_correction(self):
        c=self.calibrated();c['control']['max_total_correction_ticks']=1
        t=self.targets[0];goal=np.array(t['p_xyz_mm'])+[10,20,30]
        with self.assertRaisesRegex(DemoError,'no partial correction sent'):
            correction(t['raw_ticks'],goal-[0,0,40],t,c)

    def test_frame_rotates_error_and_jacobian_consistently(self):
        c=self.calibrated(); t=self.targets[0]; raw=t['raw_ticks']
        p=fk(raw,c['geometry'])+[0,0,-35]
        a,_=correction(raw,p+[10,20,30],t,c)
        r=np.array([[0,-1,0],[1,0,0],[0,0,1]])
        c['frame']['world_from_model_rotation']=r.tolist()
        b,_=correction(raw,r@p+[10,20,30],t,c)
        np.testing.assert_equal(a,b)

    def test_unconfigured_live_blocked_before_hardware(self):
        import run_demo
        with patch('builtins.input',side_effect=AssertionError('must not prompt')), patch('hardware.Motor',side_effect=AssertionError('must not open port')):
            with self.assertRaises(DemoError):
                run_demo.live(Model(),self.targets,self.cfg,lambda *a,**k:None)

    def test_bad_rotation_nan_and_excessive_error(self):
        c=self.calibrated(); t=self.targets[0]
        for pred in ([float('nan'),0,0],[1e6,0,0]):
            with self.assertRaises(DemoError):correction(t['raw_ticks'],pred,t,c)
        c['frame']['world_from_model_rotation'][0][0]=-1
        with self.assertRaises(DemoError):validate_config(c,self.targets,live=True)

    def test_provisional_limits_fail_closed(self):
        for name,value in [('gain',1),('max_total_correction_ticks',1000),('summary_s',.1),('damping_mm_per_tick',float('nan'))]:
            c=self.calibrated();c['control'][name]=value
            with self.assertRaises(DemoError):validate_config(c,self.targets,live=True)

    def test_target_count_distinct_and_range(self):
        with self.assertRaises(DemoError):validate_config(self.cfg,self.targets[:4])
        t=copy.deepcopy(self.targets);t[1]=t[0]
        with self.assertRaises(DemoError):validate_config(self.cfg,t)
        t=copy.deepcopy(self.targets);t[0]['raw_ticks'][0]=10
        with self.assertRaises(DemoError):validate_config(self.cfg,t)

    def test_state_guards_historical_protocol(self):
        valid=self.state_text()
        self.assertEqual(set(parse_state(valid,self.cfg)),set(AXES))
        for bad in [valid.replace('fault=false','fault=true'),
                    valid.replace('ping=true','ping=false'),
                    valid.replace('mapping_ready=true','mapping_ready=false'),
                    valid.replace('raw_active_min=1160','raw_active_min=1161'),
                    valid.replace('id=4','id=3'),valid.replace('torque=true','torque=false'),
                    valid+'\naxis=elbow_flex id=3 raw_pos=2000 mapping_ready=true',
                    valid.replace('raw_pos=1300','raw_pos=1184')]:
            with self.assertRaises(DemoError):parse_state(bad,self.cfg)

    def test_serial_fault_never_sends_next_axis(self):
        m=Motor.__new__(Motor);m.cfg=self.cfg;m.last_updates={};m.before_move=lambda:None
        sent=[]
        m.request=lambda command:sent.append(command) or self.state_text().replace('fault=false','fault=true')
        with self.assertRaises(DemoError):m.move(self.targets[0]['raw_ticks'])
        self.assertEqual(sent,['STATE'])

    def test_clamped_ack_is_failure(self):
        m=Motor.__new__(Motor);m.cfg=self.cfg;m.before_move=lambda:None
        m.request=lambda command:'[MOVE_RAW] axis=shoulder_pan id=1 raw=2001 speed=90 acc=5'
        with self.assertRaises(DemoError):m.command_axis('shoulder_pan',2000)

    def test_stale_state_is_failure(self):
        m=Motor.__new__(Motor);m.cfg=self.cfg;m.last_updates={}
        m.request=lambda command:self.state_text()
        m.state()
        with self.assertRaises(DemoError):m.state()

    def test_packet_checksum_partial_and_units(self):
        def packet(kind,v):
            body=bytes([0x55,kind])+struct.pack('<hhhh',*v,0)
            return body+bytes([sum(body)&255])
        frame=packet(0x51,[0,2048,0])
        buf=bytearray(frame[:5]); self.assertEqual(extract_packets(buf)[0],[])
        buf.extend(frame[5:]); rows,discard=extract_packets(buf)
        self.assertEqual(discard,0);np.testing.assert_allclose(rows[0][1],[0,9.80665,0])
        broken=bytearray(frame);broken[-1]^=1
        rows,discard=extract_packets(broken+bytearray(frame))
        self.assertEqual(len(rows),1);self.assertGreater(discard,0)
        rows,_=extract_packets(bytearray(packet(0x53,[16384,0,0])))
        self.assertAlmostEqual(rows[0][1][0],np.pi/2)

    def samples(self):
        imu={s:{'angle':[[.2,.1,-.5]]*30,'acceleration':[[0,9.80665,0]]*30,
                'angular_velocity':[[.001,.002,0]]*30} for s in SENSORS}
        return imu,[REF.tolist()]*10

    def test_summary_training_convention(self):
        imu,enc=self.samples(); features,quality=summarise(imu,enc,self.cfg)
        self.assertEqual(set(features),set(Model().names))
        self.assertAlmostEqual(features['imu0_base_euler_x_rad_sin'],np.sin(.2))
        self.assertAlmostEqual(features['imu0_base_gyro_rms_rad_s'],np.sqrt(.000005))
        self.assertEqual(features['encoder_delta_rad_j0'],0)

    def test_missing_samples_wrap_motion_rejected(self):
        imu,enc=self.samples();imu[SENSORS[0]]['angle']=[]
        with self.assertRaises(DemoError):summarise(imu,enc,self.cfg)
        imu,enc=self.samples();imu[SENSORS[0]]['angle'][0]=[3.14,0,0]
        with self.assertRaises(DemoError):summarise(imu,enc,self.cfg)
        imu,enc=self.samples();enc[0]=[2070,2093,1970]
        with self.assertRaises(DemoError):summarise(imu,enc,self.cfg)

    def test_model_forbids_label_input_and_out_of_envelope(self):
        model=Model(); row=read_json(ROOT/'data/replay.json')[0]['features']
        with self.assertRaises(DemoError):model.predict({**row,'vive_x':0})
        row={**row,'encoder_delta_rad_j0':100}
        with self.assertRaises(DemoError):model.predict(row,check_envelope=True)

    def test_tampered_weights_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'model.json').write_bytes((ROOT/'data/model.json').read_bytes())
            (p/'weights.npz').write_bytes(b'bad')
            with self.assertRaises(DemoError):Model(p)

    def test_independent_frame_registration_and_bad_pairs(self):
        from calibrate_frame import fit_frame
        a=np.array([[0,0,0],[100,0,0],[0,100,0],[0,0,100]],float)
        r=np.array([[0,-1,0],[1,0,0],[0,0,1]])
        b=a@r.T+[10,20,30]
        result=fit_frame(a,b)
        np.testing.assert_allclose(result['world_from_model_rotation'],r,atol=1e-10)
        np.testing.assert_allclose(result['world_from_model_translation_mm'],[10,20,30],atol=1e-10)
        b[0]+=100
        with self.assertRaises(DemoError):fit_frame(a,b)
        with self.assertRaises(DemoError):fit_frame([[0,0,0]]*4,[[1,1,1]]*4)

    def test_stale_imu_and_interior_gap_rejected(self):
        imus=IMUs.__new__(IMUs);imus.lock=threading.Lock();imus.failures=[]
        now=time.monotonic()
        imus.samples={s:{k:deque([(now-1,[0,0,0])]) for k in ('angle','acceleration','angular_velocity')} for s in SENSORS}
        with self.assertRaises(DemoError):imus.ensure_alive()
        for s in SENSORS:
            for values in imus.samples[s].values():values.append((now,[0,0,0]))
        imus.ensure_alive()
        with self.assertRaises(DemoError):imus.snapshot(now-1,now)

    def test_live_orchestration_five_points_one_correction_with_fake_devices(self):
        import run_demo
        c=self.calibrated();targets=self.targets;geometry=c['geometry']
        calls=[];device={};events=[]
        class FakeMotor:
            def __init__(self,cfg,log):self.raw=np.array(targets[0]['raw_ticks']);device['m']=self
            def safe_engage(self):calls.append('engage')
            def move(self,raw,force_goal=False):
                self.raw=np.array(raw);calls.append(('move',self.raw.tolist()))
                calls.append(('force_goal',force_goal))
                for t in targets:
                    if self.raw.tolist()==t['raw_ticks']:
                        self.target=t
            def state(self):return self.raw
            def positions(self,state):return state
            def wait(self,seconds):calls.append(('wait',seconds))
            def release_supported(self):calls.append('release')
            def close(self):calls.append('motor_close')
            def hold_after_abort(self):calls.append('abort_hold')
        class FakeIMUs:
            def __init__(self,*a):pass
            def ensure_alive(self):pass
            def close(self):calls.append('imu_close')
        class FakeModel:
            def predict(self,features,check_envelope=False):
                calls.append(('predict',check_envelope))
                return np.array(features['synthetic_position'])
        def fake_measure(*args):
            # Known-reachable synthetic error from +60 ticks on each joint.
            t=device['m'].target
            np.testing.assert_array_equal(device['m'].raw,t['raw_ticks'])
            calls.append(('measure',t['target_id']))
            goal=np.array(t['p_xyz_mm'])
            displacement=fk(np.array(t['raw_ticks'])+60,geometry)-fk(t['raw_ticks'],geometry)
            offset=goal-fk(t['raw_ticks'],geometry)-displacement
            p=fk(device['m'].raw,geometry)+[10,20,30]+offset
            return {'synthetic_position':p.tolist()},{},None
        class FakeLog:
            def __init__(self,path):self.directory=path
            def __call__(self,event,**values):events.append(event)
        with tempfile.TemporaryDirectory() as directory, patch('hardware.Motor',FakeMotor),patch('hardware.IMUs',FakeIMUs),patch('hardware.measure',fake_measure),patch('time.sleep'),patch('builtins.input',side_effect=['RUN FIVE TARGETS','RELEASE']),patch('builtins.print'):
            run_demo.live(FakeModel(),targets,c,FakeLog(Path(directory)))
            with (Path(directory)/'live_summary.csv').open(encoding='utf-8-sig') as f:
                rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),5)
            self.assertTrue(all(r['post_correction_measured']=='False' for r in rows))
            self.assertTrue(all('estimated_after_error_mm' not in r and 'vive_after_error_mm' not in r for r in rows))
            self.assertTrue(all(r['status']=='full_error_correction' for r in rows))
            self.assertTrue(all(float(r['rigid_ik_residual_mm'])<=c['control']['ik_tolerance_mm'] for r in rows))
        self.assertEqual(sum(isinstance(a,tuple) and a[0]=='move' for a in calls),10)
        self.assertEqual([a[1] for a in calls if isinstance(a,tuple) and a[0]=='force_goal'],[False,True]*5)
        self.assertEqual([a[1] for a in calls if isinstance(a,tuple) and a[0]=='measure'],[t['target_id'] for t in targets])
        self.assertEqual(calls.count(('predict',True)),5)
        self.assertEqual(calls.count(('wait',2.5)),5)
        self.assertEqual(events.count('target_motion_complete'),5)
        self.assertNotIn('after',events)
        self.assertIn('release',calls);self.assertNotIn('abort_hold',calls)

    def test_errors_below_former_15mm_threshold_are_requested_in_full(self):
        c=self.calibrated();t=self.targets[0];goal=np.array(t['p_xyz_mm'])+[10,20,30]
        for distance in (1.,5.,10.,14.,15.,20.):
            command,detail=correction(t['raw_ticks'],goal-[0,0,distance],t,c)
            self.assertEqual(detail['status'],'full_error_correction')
            np.testing.assert_allclose(detail['requested_correction_world_mm'],[0,0,distance],atol=1e-10)
            self.assertFalse(np.array_equal(command,t['raw_ticks']))

    def test_submillimeter_request_is_not_discarded_by_solver_tolerance(self):
        c=self.calibrated();t=self.targets[0];raw=np.array(t['raw_ticks'])
        delta=fk(raw+[1,0,0],c['geometry'])-fk(raw,c['geometry'])
        self.assertLess(np.linalg.norm(delta),c['control']['ik_tolerance_mm'])
        command,detail=correction(raw,np.array(t['p_xyz_mm'])+[10,20,30]-delta,t,c)
        self.assertEqual(detail['status'],'full_error_correction')
        np.testing.assert_equal(command,raw+[1,0,0])

    def test_unrepresentable_tiny_error_is_explicit(self):
        c=self.calibrated();t=self.targets[0];goal=np.array(t['p_xyz_mm'])+[10,20,30]
        command,detail=correction(t['raw_ticks'],goal-[0,0,.0001],t,c)
        self.assertEqual(detail['status'],'below_tick_resolution')
        np.testing.assert_equal(command,t['raw_ticks'])
        self.assertAlmostEqual(detail['requested_correction_world_mm'][2],.0001)

    def test_one_tick_motor_goal_is_sent_despite_arrival_tolerance(self):
        m=Motor.__new__(Motor);m.cfg=self.cfg;m.before_move=lambda:None
        raw=np.array(self.targets[0]['raw_ticks'])
        m.state=lambda:{a:{'raw_pos':str(v),'raw_speed':'0'} for a,v in zip(AXES,raw)}
        sent=[];m.command_axis=lambda a,v:sent.append((a,v));m.wait=lambda seconds:None
        m.move(raw+[1,0,0],force_goal=True)
        self.assertIn(('shoulder_pan',int(raw[0])+1),sent)

class ApproximateDemoTests(unittest.TestCase):
    def setUp(self):
        self.cfg=read_json(ROOT/'config.approximate.json')
        self.targets=read_json(ROOT/'data/targets.json')

    def ready_fixture(self):
        c=copy.deepcopy(self.cfg)
        for key in ('same_training_installation_verified','imu_mapping_and_scales_verified',
                    'three_axis_rig_verified','path_clearance_and_external_stop_verified'):
            c['commissioning'][key]=True
        c['commissioning']['operator']='OFFLINE TEST ONLY'
        return c

    def test_effective_fk_and_transform_match_fitted_candidate(self):
        from fit_coordinate_frame import effective_fk
        from demo_core import target_world
        model=read_json(ROOT/'coordinate_analysis/candidate_effective_fk.json')['model']
        np.testing.assert_allclose(self.cfg['frame']['world_from_model_rotation'],model['rotation'],atol=1e-12)
        np.testing.assert_allclose(self.cfg['frame']['world_from_model_translation_mm'],model['translation_mm'],atol=1e-12)
        for target in self.targets:
            p=effective_fk(np.array([target['raw_ticks']]),model['effective_parameters'][:2],model['effective_parameters'][2:])[0]
            np.testing.assert_allclose(fk(target['raw_ticks'],self.cfg['geometry']),p,atol=1e-10)
            expected=np.array(model['rotation'])@p+model['translation_mm']
            np.testing.assert_allclose(target_world(target,self.cfg),expected,atol=1e-10)

    def test_reference_does_not_mix_old_planned_xyz_with_effective_fk(self):
        from demo_core import target_world
        target=copy.deepcopy(self.targets[0]);before=target_world(target,self.cfg)
        target['p_xyz_mm']=[1e6,2e6,3e6]
        np.testing.assert_array_equal(target_world(target,self.cfg),before)

    def test_approximation_requires_consent_and_preserves_rig_checks(self):
        c=self.ready_fixture()
        self.assertFalse(c['commissioning']['frame_and_marker_calibration_verified'])
        validate_config(c,self.targets,live=True)  # Does not claim independent calibration.
        c['approximation']['accepted_for_demo']=False
        with self.assertRaises(DemoError):validate_config(c,self.targets,live=True)
        for key in ('same_training_installation_verified','imu_mapping_and_scales_verified',
                    'three_axis_rig_verified','path_clearance_and_external_stop_verified'):
            c=self.ready_fixture();c['commissioning'][key]=False
            with self.assertRaises(DemoError):validate_config(c,self.targets,live=True)

    def test_invalid_approximation_parameters_rejected(self):
        for group,key,value in [('geometry','joint_angle_offsets_rad',[0,float('nan'),0]),
                                ('geometry','link_lengths_mm',[-1,215]),
                                ('frame','world_from_model_rotation',np.diag([-1,1,1]).tolist())]:
            c=copy.deepcopy(self.cfg);c[group][key]=value
            with self.assertRaises(DemoError):validate_config(c,self.targets)

    def test_five_archived_inputs_request_full_errors(self):
        from demo_core import target_world
        model=Model()
        for target,row in zip(self.targets,read_json(ROOT/'data/replay.json')):
            prediction=model.predict(row['features'],check_envelope=True)
            command,detail=correction(row['raw_ticks'],prediction,target,self.cfg)
            error=target_world(target,self.cfg)-prediction
            np.testing.assert_allclose(detail['requested_correction_world_mm'],error,atol=1e-10)
            self.assertEqual(detail['status'],'full_error_correction')
            delta=np.array(self.cfg['frame']['world_from_model_rotation'])@(
                fk(command,self.cfg['geometry'])-fk(row['raw_ticks'],self.cfg['geometry']))
            self.assertLessEqual(np.linalg.norm(delta-error),self.cfg['control']['ik_tolerance_mm'])

    def test_approximate_live_five_points_with_mock_devices_no_remeasurement(self):
        from unittest.mock import Mock
        import run_demo
        c=self.ready_fixture();rows=iter(read_json(ROOT/'data/replay.json'))
        motor=Mock();imus=Mock();model=Mock(wraps=Model());events=[]
        motor.positions.side_effect=lambda state:state
        def measure_once(*args):
            row=next(rows)
            motor.state.return_value=np.array(row['raw_ticks'])
            return row['features'],{},None
        class FakeLog:
            def __init__(self,path):self.directory=path
            def __call__(self,event,**values):events.append(event)
        with tempfile.TemporaryDirectory() as directory,patch('hardware.Motor',return_value=motor),\
             patch('hardware.IMUs',return_value=imus),patch('hardware.measure',side_effect=measure_once) as measurement,\
             patch('time.sleep'),patch('builtins.input',side_effect=['RUN FIVE TARGETS','RELEASE']),patch('builtins.print'):
            run_demo.live(model,self.targets,c,FakeLog(Path(directory)))
            self.assertEqual(measurement.call_count,5)
            self.assertEqual(model.predict.call_count,5)
            self.assertEqual(motor.move.call_count,10)
            for call in motor.move.call_args_list[1::2]:self.assertTrue(call.kwargs['force_goal'])
            self.assertEqual(events.count('target_motion_complete'),5)
            self.assertNotIn('after',events)
            motor.hold_after_abort.assert_not_called()
            motor.release_supported.assert_called_once()

    def test_stock_approximate_profile_blocks_before_hardware(self):
        import run_demo
        with patch('builtins.input',side_effect=AssertionError('no prompt')),\
             patch('hardware.Motor',side_effect=AssertionError('no port')):
            with self.assertRaises(DemoError):
                run_demo.live(Model(),self.targets,self.cfg,lambda *a,**k:None)


if __name__=='__main__':unittest.main(verbosity=2)
