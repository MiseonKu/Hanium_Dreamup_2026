"""Offline frame/kinematics diagnostics. Never imports hardware or edits live config.

Requires pandas, pyarrow and scipy in the analysis environment, not on the robot.
Fits only segments 36-38 and excludes all five demo targets. Target-group CV is
development evidence; this is NOT an independent calibration or control test.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from demo_core import ROOT, REF, RAD_PER_TICK, DemoError, read_json, Model, correction

SOURCE_SHA256 = '94deccb960b0554fb9a683875d36ef5726afc825d4b4ea244a1490f3353b1930'
MEMBER = 'ml_preparation_v1/canonical/canonical_real.parquet'
KINDS = ('planned_rigid', 'encoder_rigid', 'encoder_anchor_fit', 'encoder_effective_geometry_fit')
COLS = ([f'planned_p_{a}_mm' for a in 'xyz'] +
        [f'encoder_tick_j{j}' for j in range(3)] +
        [f'vive_pos_raw_{a}_mm' for a in 'xyz'])


def rigid_fit(source, destination):
    a, b = np.asarray(source, float), np.asarray(destination, float)
    if a.ndim != 2 or a.shape != b.shape or a.shape[1] != 3 or len(a) < 4:
        raise DemoError('At least four paired 3D points required')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise DemoError('Nonfinite registration data')
    aa, bb = a-a.mean(0), b-b.mean(0)
    if min(np.linalg.svd(aa, compute_uv=False)[1], np.linalg.svd(bb, compute_uv=False)[1]) < 1e-8:
        raise DemoError('Collinear registration data')
    u, _, vt = np.linalg.svd(aa.T @ bb)
    r = vt.T @ np.diag([1., 1., np.linalg.det(vt.T @ u.T)]) @ u.T
    return r, b.mean(0)-r @ a.mean(0)


def effective_fk(raw, offsets=(0., 0.), lengths=(233., 215.)):
    """Diagnostic effective geometry, NOT measured undeformed geometry."""
    raw = np.asarray(raw, float)
    if raw.ndim != 2 or raw.shape[1] != 3 or not np.isfinite(raw).all():
        raise DemoError('Finite N by 3 raw ticks required')
    a, b, c = ((raw-REF)*RAD_PER_TICK + [0., np.pi/2+offsets[0], offsets[1]]).T
    radius = lengths[0]*np.cos(b) + lengths[1]*np.cos(b+c)
    return np.column_stack((np.cos(a)*radius, np.sin(a)*radius,
                            lengths[0]*np.sin(b)+lengths[1]*np.sin(b+c)))


def model_points(table, kind, parameters):
    if kind == 'planned_rigid':
        return np.asarray(table, float)[:, :3]
    return effective_fk(np.asarray(table, float)[:, 3:6], parameters[:2], parameters[2:])


def fit_model(table, kind):
    if kind not in KINDS:
        raise DemoError('Unknown diagnostic model')
    table = np.asarray(table, float)
    parameters = np.array([0., 0., 233., 215.])
    optimizer = None
    if kind in ('encoder_anchor_fit', 'encoder_effective_geometry_fit'):
        from scipy.optimize import least_squares
        vary_lengths = kind == 'encoder_effective_geometry_fit'
        lower = [-np.pi, -np.pi] + ([116.5, 107.5] if vary_lengths else [])
        upper = [np.pi, np.pi] + ([349.5, 322.5] if vary_lengths else [])

        def unpack(x):
            return np.asarray(x) if vary_lengths else np.r_[x, 233., 215.]

        def residual(x):
            points = model_points(table, kind, unpack(x))
            r, t = rigid_fit(points, table[:, 6:])
            return (points @ r.T+t-table[:, 6:]).ravel()

        candidates = []
        # Fixed multi-start policy; no seed or limit selected from demo outcomes.
        for lift in (-2.5, -1.5, 0., 1.5, 2.5):
            for wrist in (-1.5, 0., 1.5):
                start = [lift, wrist] + ([233., 215.] if vary_lengths else [])
                result = least_squares(residual, start, bounds=(lower, upper), max_nfev=300,
                                       ftol=1e-10, xtol=1e-10, gtol=1e-10)
                if result.success and np.isfinite(result.fun).all():
                    candidates.append(result)
        if not candidates:
            raise DemoError('No converged diagnostic fit')
        best = min(candidates, key=lambda x: float(x.fun @ x.fun))
        parameters = unpack(best.x)
        optimizer = {'starts': 15, 'converged_starts': len(candidates),
                     'selected_nfev': int(best.nfev),
                     'at_bound': bool(np.any(np.minimum(best.x-lower, upper-best.x) < 1e-5))}
    r, t = rigid_fit(model_points(table, kind, parameters), table[:, 6:])
    return {'kind': kind, 'effective_parameters': parameters.tolist(),
            'additional_lift_wrist_offsets_deg': np.rad2deg(parameters[:2]).tolist(),
            'effective_link_lengths_mm': parameters[2:].tolist(),
            'rotation': r.tolist(), 'translation_mm': t.tolist(), 'optimizer': optimizer,
            'independently_calibrated': False, 'live_authorized': False,
            'transform_input_basis': ('historical_planned_xyz' if kind == 'planned_rigid'
                                     else 'FK_with_THIS_models_effective_parameters'),
            'warning': 'Effective parameters may absorb flex, mounting and other errors; not measured geometry.'}


def predict(table, model):
    return (model_points(table, model['kind'], model['effective_parameters']) @
            np.asarray(model['rotation']).T + model['translation_mm'])


def scores(residual):
    norm = np.linalg.norm(np.asarray(residual, float), axis=1)
    return {'n': len(norm), 'rmse_3d_mm': float(np.sqrt(np.mean(norm**2))),
            'median_3d_mm': float(np.median(norm)), 'p95_3d_mm': float(np.percentile(norm, 95)),
            'max_3d_mm': float(norm.max())}


def fitting_mask(segment_ids, target_ids, demo_ids):
    return np.isin(segment_ids, [36, 37, 38]) & ~np.isin(target_ids, demo_ids)


def preview_nominal(model, targets):
    """Use the unchanged existing solver, offline. This returns no live config."""
    c = read_json(ROOT/'config.json')
    c['frame'] = {'world_from_model_rotation': model['rotation'],
                  'world_from_model_translation_mm': model['translation_mm']}
    c['geometry']['marker_offset_tip_mm'] = [0, 0, 0]  # user-stated coincident points
    estimator = Model()
    replay = {row['target_id']: row for row in read_json(ROOT/'data/replay.json')}
    out = []
    for target in targets:
        row = replay[target['target_id']]
        position = estimator.predict(row['features'])
        goal = np.asarray(model['rotation']) @ target['p_xyz_mm'] + model['translation_mm']
        record = {'target_id': target['target_id'], 'trial_uid': row['trial_uid'],
                  'requested_world_xyz_mm': (goal-position).tolist(),
                  'requested_norm_mm': float(np.linalg.norm(goal-position)),
                  'actual_motor_command_sent': False}
        try:
            command, detail = correction(row['raw_ticks'], position, target, c)
            record.update(status=detail['status'], candidate_ticks=command.tolist(), detail=detail)
        except DemoError as exc:
            record.update(status='rejected_no_motion', reason=str(exc))
        out.append(record)
    return out


def main():
    import pandas as pd
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=Path(
        r'C:\Users\PC\Documents\VS Code\capstone-code\06_simulation_and_data_collect_preparation\imu_added_experiment\ml_preparation_v1.zip'))
    parser.add_argument('--output', type=Path, default=ROOT/'coordinate_analysis')
    args = parser.parse_args()
    with args.archive.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    if digest != SOURCE_SHA256:
        raise DemoError('Pinned dataset checksum mismatch; no fitting performed')
    with ZipFile(args.archive) as z:
        data = pd.read_parquet(io.BytesIO(z.read(MEMBER)))
    data = data.loc[data.train_eligible].copy()
    if len(data) != 7702:
        raise DemoError('Unexpected eligible row count')
    targets = read_json(ROOT/'data/targets.json')
    demo_ids = [t['target_id'] for t in targets]
    train = data.loc[fitting_mask(data.segment_id, data.target_id, demo_ids)]
    later = data.loc[data.segment_id.isin([39, 40]) & ~data.target_id.isin(demo_ids)]
    demo = data.loc[data.target_id.isin(demo_ids)]
    # Equal target/segment weighting, so repeats do not masquerade as new poses.
    means = train.groupby(['target_id', 'segment_id'])[COLS].mean().groupby('target_id').mean()
    table = means.to_numpy(float)
    fold_ids = np.array([int(t[1:]) % 5 for t in means.index])
    models = {}
    for kind in KINDS:
        print('Fitting', kind, flush=True)
        model = fit_model(table, kind)
        cv_residual = np.empty((len(table), 3))
        folds = []
        for fold in range(5):
            train_mask, test_mask = fold_ids != fold, fold_ids == fold
            local = fit_model(table[train_mask], kind)
            error = predict(table[test_mask], local)-table[test_mask, 6:]
            cv_residual[test_mask] = error
            folds.append({'fold': fold, 'fit_target_ids': means.index[train_mask].tolist(),
                          'check_target_ids': means.index[test_mask].tolist(),
                          'scores': scores(error), 'model': local})
        model['scores'] = {'fit_target_means': scores(predict(table, model)-table[:, 6:]),
                           'development_target_cv': scores(cv_residual)}
        for role, rows in (('fit_trials', train), ('later_trials', later), ('demo_trials', demo)):
            values = rows[COLS].to_numpy(float)
            model['scores'][role] = scores(predict(values, model)-values[:, 6:])
        model['demo_per_target'] = {
            name: scores(predict(rows[COLS].to_numpy(float), model)-rows[COLS[6:]].to_numpy(float))
            for name, rows in demo.groupby('target_id')}
        model['development_folds'] = folds
        models[kind] = model
        print('  target CV %.2f mm; later trials %.2f mm' % (
            model['scores']['development_target_cv']['rmse_3d_mm'],
            model['scores']['later_trials']['rmse_3d_mm']), flush=True)
    preferred = min(KINDS, key=lambda k: models[k]['scores']['development_target_cv']['rmse_3d_mm'])
    report = {
        'contract': 'offline_coordinate_and_effective_geometry_diagnostic.v1',
        'source_archive_sha256': digest, 'source_member': MEMBER,
        'eligible_rows': len(data), 'fit_rows': len(train), 'fit_targets': len(means),
        'fit_segments': [36, 37, 38], 'later_check_segments': [39, 40],
        'demo_target_ids_excluded_from_all_fits': demo_ids,
        'fit_trial_uid_sha256': hashlib.sha256('\n'.join(sorted(train.trial_uid)).encode()).hexdigest(),
        'fit_target_means': [{'target_id': t, 'values_in_column_order': row.tolist()}
                             for t, row in zip(means.index, table)], 'column_order': COLS,
        'models': models, 'lowest_development_cv_model': preferred,
        'nominal_demo_preview': preview_nominal(models['planned_rigid'], targets),
        'hardware_commands_sent': 0, 'live_configuration_changed': False,
        'source_dataset_and_ML_weights_changed': False,
        'claims': [
            'All checks are retrospective development diagnostics, not independent validation.',
            'Raw labels and official ML evaluation are unchanged; no all-data alignment was fitted.',
            'Point-mean residuals are NOT trial residuals, ML accuracy or physical correction accuracy.',
            'R/t for an effective FK cannot be pasted into the nominal demo geometry.',
            'Effective fitted geometry may absorb average sag; it does not identify the undeformed target.',
            'No reflection, spatial scaling or affine warp is permitted in the reported rigid transforms.',
            'The historical ML itself used all eligible rows; replay is in-sample, not an independent test.',
            'Anchor/sign identifiability is limited; fitted offsets are not verified motor calibrations.']}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'alignment_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                                               allow_nan=False), encoding='utf-8')
    candidate = {'contract': 'effective_fk_candidate.analysis_only.v1',
                 'model': models[preferred], 'source_archive_sha256': digest,
                 'joint_angle_rule': 'q=(raw-[2055,2093,1970])*2*pi/4096+[0,pi/2+offset_lift,offset_wrist]',
                 'compatible_with_existing_demo_geometry': False,
                 'is_undeformed_target_model': False, 'live_authorized': False,
                 'warning': 'Do not paste this transform into config.json; it uses different effective kinematics.'}
    (args.output/'candidate_effective_fk.json').write_text(
        json.dumps(candidate, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    names = {'planned_rigid': '기존 계획점 + 회전/이동만', 'encoder_rigid': '실측 엔코더 FK + 회전/이동만',
             'encoder_anchor_fit': '관절 기준각 추가 추정 (링크 길이 고정)',
             'encoder_effective_geometry_fit': '기준각 + 유효 링크 길이 추가 추정'}
    lines = ['# 좌표계 및 기구학 오프라인 분석', '',
        '실제 모터 구동·기존 데이터/모델/실구동 설정 변경 없음. 아래 값은 보정 후 오차가 아닙니다.', '',
        f'학습 적격 7,702행 중 {len(train):,}행, {len(means)}개 목표의 segment 36–38만 맞춤에 사용했습니다.',
        '기존 데모 5점은 모든 맞춤과 모델 선택에서 제외했습니다. 남은 목표의 5-fold 검사는 개발용 진단입니다.',
        '후반 segment 39–40도 같은 과거 세션이므로 독립 실험이라고 부르지 않습니다.', '',
        '| 방법 | 맞춤 목표 평균점 RMSE | 목표 제외 교차검사 RMSE | 후반 실제 행 RMSE | 데모 5점 실제 행 RMSE |',
        '|---|---:|---:|---:|---:|']
    for kind in KINDS:
        metrics = models[kind]['scores']
        values = [metrics[k]['rmse_3d_mm'] for k in
                  ('fit_target_means', 'development_target_cv', 'later_trials', 'demo_trials')]
        lines.append('| '+names[kind]+' | '+' | '.join(f'{v:.2f} mm' for v in values)+' |')
    lines += ['', '## 해석', '',
        '- 회전·원점 이동만으로 남는 차이와, 기구학 파라미터를 바꿔 줄인 차이는 서로 다릅니다.',
        '- 추가 추정한 기준각과 링크 길이는 데이터를 설명하는 유효값입니다. 실제 치수/영점 측정값이 아닙니다.',
        '- 처짐, 체결/장착, 모델 구조 오차를 함께 흡수할 수 있어 이 값을 처짐 없는 정답으로 사용할 수 없습니다.',
        '- 추정된 회전/이동은 반드시 해당 기구학과 함께 사용해야 합니다. 기존 config.json에 복사하면 안 됩니다.',
        '- 작은 평균점 잔차를 개별 시행 위치 정확도 또는 보정 성능으로 보고하면 안 됩니다.',
        '- 엔코더+IMU 위치 추정 모델은 재학습하지 않았고, 기존 약 14.3 mm 결과도 변경하지 않았습니다.', '',
        '## 기존 기구학 + 순수 좌표변환의 5점 오프라인 명령 검사', '',
        '과거 각 1행의 ML 입력을 재생한 결과이며 실제 센서 또는 모터를 사용하지 않았습니다.', '',
        '| 목표 | 요청 위치 차이 크기 | 결과 |', '|---|---:|---|']
    for row in report['nominal_demo_preview']:
        lines.append(f"| {row['target_id']} | {row['requested_norm_mm']:.2f} mm | {row['status']} |")
    lines += ['', '상세 변환행렬, 파라미터, 분할 목록과 거부 사유는 alignment_report.json에 있습니다.', '',
              '다음 실장비 확인: 모터별 실제 회전축/방향, 기준 자세, 회전축 사이 거리, 기준 자세의 말단 위치.',
              '사용자 확인에 따라 마커 오프셋은 0으로 가정했지만, 처짐 없는 목표 위치는 별도 근거가 필요합니다.', '']
    (args.output/'ALIGNMENT_REPORT_KO.md').write_text('\n'.join(lines), encoding='utf-8')
    print('Report:', args.output/'ALIGNMENT_REPORT_KO.md', flush=True)
    print('Live config unchanged; no hardware commands sent.', flush=True)


if __name__ == '__main__':
    main()
