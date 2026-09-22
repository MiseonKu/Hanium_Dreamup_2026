"""Default: historical offline replay. --live explicitly opts into real motion."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time
import numpy as np
from demo_core import ROOT, Model, DemoError, read_json, validate_config, correction, fk, target_world

class Log:
    def __init__(self,mode):
        self.directory=ROOT/'results'/f'{datetime.now():%Y%m%d_%H%M%S_%f}_{mode}'
        self.directory.mkdir(parents=True)
        self.file=(self.directory/'events.jsonl').open('x',encoding='utf-8')
        self.lock=threading.Lock()

    def __call__(self,event,**values):
        with self.lock:
            self.file.write(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),
                'monotonic_s':time.monotonic(),'event':event,**values},ensure_ascii=False,allow_nan=False)+'\n')
            self.file.flush()

    def close(self):
        self.file.close()

def save_csv(path,rows):
    if rows:
        with path.open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)

def dry_run(model,targets,cfg,log):
    print('OFFLINE REPLAY: actual archived inputs, no serial ports, no physical corrections.')
    replay=read_json(ROOT/'data/replay.json')
    output=[]
    for target,row in zip(targets,replay):
        if target['target_id']!=row['target_id']:
            raise DemoError('Replay/target pairing mismatch')
        prediction=model.predict(row['features'])
        err=float(np.linalg.norm(prediction-np.asarray(row['measured_world_mm'])))
        record={'target_id':target['target_id'],'trial_uid':row['trial_uid'],
                'nominal_raw_ticks':json.dumps(target['raw_ticks']),
                'predicted_training_world_mm':json.dumps(prediction.tolist()),
                'historical_measured_world_mm':json.dumps(row['measured_world_mm']),
                'resubstitution_prediction_error_mm':err,
                'physical_correction_tested':False}
        output.append(record); log('replay',**record)
        print(f"{target['target_id']} raw={target['raw_ticks']} prediction={prediction.round(2)}")
    save_csv(log.directory/'replay.csv',output)
    print('These are previously trained rows; replay errors are NOT a new accuracy score.')
    print('Live calibration configured:',all(cfg['commissioning'].get(k) is True for k in (
        'same_training_installation_verified','frame_and_marker_calibration_verified')))

def simulate(targets,cfg,log):
    """Controller plumbing only, explicitly synthetic sag and frame, no trained ML."""
    c=json.loads(json.dumps(cfg))
    c['frame']={'world_from_model_rotation':np.eye(3).tolist(),'world_from_model_translation_mm':[0,0,0]}
    c['geometry']['marker_offset_tip_mm']=[0,0,0]
    rows=[]
    print('SYNTHETIC CONTROLLER TEST: full 40 mm upward request, identity frame, no ML/live evidence.')
    for target in targets:
        raw=np.array(target['raw_ticks'],float)
        goal=np.asarray(target['p_xyz_mm'])
        before=goal-[0,0,40]
        offset=before-fk(raw,c['geometry'])
        try:
            command,detail=correction(raw,before,target,c)
            after=fk(command,c['geometry'])+offset
            after_error=float(np.linalg.norm(goal-after))
            status='full_error_correction'
            reason=''
        except DemoError as exc:
            command=raw
            after_error=None
            status='rejected_no_motion'
            reason=str(exc)
        row={'target_id':target['target_id'],'before_ticks':json.dumps(raw.tolist()),
             'after_ticks':json.dumps(command.tolist()),'synthetic_before_error_mm':float(np.linalg.norm(goal-before)),
             'synthetic_after_error_mm':after_error,'status':status,'reason':reason,'physical_tested':False}
        rows.append(row); log('synthetic_controller',**row)
        print(f"{target['target_id']} full 40 mm request: {status}; synthetic remaining={after_error}; {reason}")
    save_csv(log.directory/'synthetic_controller.csv',rows)

def preview_correction(model,targets,cfg,log):
    print('OFFLINE CORRECTION PREVIEW: archived inputs only; no ports or motor motion.')
    print('Reference mode:',cfg.get('mode','independent_calibration'))
    replay={r['target_id']:r for r in read_json(ROOT/'data/replay.json')}
    rows=[]
    for target in targets:
        old=replay[target['target_id']]
        prediction=model.predict(old['features'],check_envelope=True)
        desired=target_world(target,cfg)
        row={'target_id':target['target_id'],'trial_uid':old['trial_uid'],
             'reference_mode':cfg.get('mode','independent_calibration'),
             'requested_world_xyz_mm':json.dumps((desired-prediction).tolist()),
             'requested_norm_mm':float(np.linalg.norm(desired-prediction)),
             'candidate_ticks':'','status':'','reason':'','rigid_ik_residual_mm':None,
             'hardware_executed':False,'post_correction_measured':False}
        try:
            command,detail=correction(old['raw_ticks'],prediction,target,cfg)
            row.update(candidate_ticks=json.dumps(command.tolist()),status=detail['status'],
                       rigid_ik_residual_mm=detail.get('rigid_ik_residual_mm'))
        except DemoError as exc:
            row.update(status='rejected_no_motion',reason=str(exc))
        rows.append(row);log('correction_preview',**row)
        print(f"{target['target_id']}: request={row['requested_norm_mm']:.2f} mm; {row['status']}; ticks={row['candidate_ticks']}; {row['reason']}")
    save_csv(log.directory/'correction_preview.csv',rows)
    print('These are in-sample archived inputs, not a live accuracy/control test.')


def live(model,targets,cfg,log,vive_audit=False):
    validate_config(cfg,targets,live=True)  # BEFORE importing/opening any hardware API.
    from hardware import IMUs, Motor, ViveAudit, measure
    print('LIVE: five targets, one full-error correction each, no post-correction position measurement. Support arm before startup.')
    print('At completion/abort torque may remain ON; attend the arm until supported release.')
    if cfg.get('mode')=='approximate_dataset_demo':
        print('APPROXIMATE REFERENCE: fitted effective FK, not a measured sag-free target. Correction can be in any XYZ direction.')
    if input('Type RUN FIVE TARGETS to operate the physical robot: ').strip()!='RUN FIVE TARGETS':
        raise DemoError('Live execution cancelled before opening ports')
    imus=motor=audit=None
    results=[]
    try:
        imus=IMUs(cfg,log)
        time.sleep(1.5); imus.ensure_alive()
        if vive_audit:
            audit=ViveAudit(cfg['hardware']['vive_audit_serial'],log)
        motor=Motor(cfg,log)
        motor.before_move=imus.ensure_alive
        motor.safe_engage()
        for target in targets:
            print(f"{target['target_id']}: moving to original target {target['raw_ticks']}")
            log('target_start',target=target)
            motor.move(target['raw_ticks'])
            before_features,before_quality,before_audit=measure(motor,imus,cfg,audit)
            before=model.predict(before_features,check_envelope=True)
            raw=motor.positions(motor.state())
            try:
                command,detail=correction(raw,before,target,cfg)
            except DemoError as exc:
                log('correction_rejected_no_motion',target_id=target['target_id'],
                    features=before_features,quality=before_quality,
                    prediction_world_mm=before.tolist(),audit=before_audit,reason=str(exc))
                raise
            log('before',target_id=target['target_id'],features=before_features,quality=before_quality,
                prediction_world_mm=before.tolist(),audit=before_audit,correction=detail,command_ticks=command.tolist())
            if detail['status']=='full_error_correction':
                print(f"{target['target_id']}: full error {detail['requested_correction_world_mm']} mm -> {command.tolist()}")
                motor.move(command,force_goal=True)
            else:
                print(f"{target['target_id']}: {detail['status']} (requested error is recorded)")
            # Motion-only demo: retain motor arrival/fault checks, but do not
            # collect another ML window or claim a measured correction result.
            goal=np.asarray(detail['desired_world_mm'])
            before_error=float(np.linalg.norm(goal-before))
            row={'target_id':target['target_id'],'status':detail['status'],
                 'reference_mode':cfg.get('mode','independent_calibration'),
                 'estimated_before_error_mm':before_error,'post_correction_measured':False,
                 'vive_before_error_mm':None,
                 'raw_before':json.dumps(raw.tolist()),'raw_command':json.dumps(command.tolist()),
                 'requested_correction_world_mm':json.dumps(detail.get('requested_correction_world_mm',[0,0,0])),
                 'rigid_ik_residual_mm':detail.get('rigid_ik_residual_mm'),
                 'desired_training_world_mm':json.dumps(goal.tolist())}
            if before_audit and before_audit['valid']:
                row['vive_before_error_mm']=float(np.linalg.norm(goal-np.asarray(before_audit['measured_world_mm'])))
            results.append(row); save_csv(log.directory/'live_summary.csv',results)
            log('target_motion_complete',target_id=target['target_id'],summary=row)
            print(f"{target['target_id']}: {detail['status']}; initial estimated error {before_error:.1f} mm; final position not remeasured.")
            motor.wait(2.5)
        log('five_targets_complete')
    except BaseException:
        if motor:
            motor.hold_after_abort()
        raise
    finally:
        if motor:
            # No blind park and no automatic torque release on a flexible unsupported arm.
            try:
                print('Support the arm NOW. Type RELEASE only after supporting it. External stop if needed.')
                if input('Supported release (RELEASE): ').strip()=='RELEASE':
                    motor.release_supported()
                else:
                    print('Torque state retained. Do not leave arm unattended; use external stop/support.')
            except (Exception,KeyboardInterrupt) as exc:
                print(f'Release not confirmed: {exc}. Torque may remain ON. Attend arm/use external stop.')
            motor.close()
        if audit:
            audit.close()
        if imus:
            imus.close()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--dry-run',action='store_true')
    mode.add_argument('--simulate',action='store_true')
    mode.add_argument('--live',action='store_true')
    mode.add_argument('--preview-correction',action='store_true')
    p.add_argument('--config',type=Path,default=ROOT/'config.json')
    p.add_argument('--vive-audit',action='store_true',help='Record Vive as evaluation only (live mode)')
    args=p.parse_args()
    cfg=read_json(args.config); targets=read_json(ROOT/'data/targets.json')
    validate_config(cfg,targets,live=args.live)
    model=Model()
    log=Log('live' if args.live else 'synthetic' if args.simulate else 'correction_preview' if args.preview_correction else 'replay')
    try:
        log('run_start',config=cfg,targets=targets,model=model.meta,
            mode='live' if args.live else 'synthetic' if args.simulate else 'correction_preview' if args.preview_correction else 'replay')
        if args.live:
            live(model,targets,cfg,log,args.vive_audit)
        elif args.simulate:
            simulate(targets,cfg,log)
        elif args.preview_correction:
            preview_correction(model,targets,cfg,log)
        else:
            dry_run(model,targets,cfg,log)
    except (Exception,KeyboardInterrupt) as exc:
        log('aborted',reason=str(exc)); print(f'STOP: {exc}'); return 2
    finally:
        log.close(); print(f'Results: {log.directory}')
    return 0

if __name__=='__main__':
    try:
        raise SystemExit(main())
    except DemoError as exc:
        print(f'PRECHECK BLOCKED (no ports opened): {exc}')
        raise SystemExit(2)
