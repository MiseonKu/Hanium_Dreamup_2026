"""Reproduce portable NumPy model and five-target data from local immutable archives."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
from zipfile import ZipFile
import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parent

def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo/'09_SO101_IMU_VIVE_LEARNING/src'))
    from demo_core import Model, jacobian
    base = args.repo/'06_simulation_and_data_collect_preparation/imu_added_experiment'
    out = ROOT/'data'
    out.mkdir(exist_ok=True)
    with ZipFile(base/'endpoint_regression_experiment_v1.zip') as z:
        prefix = 'endpoint_regression_experiment_v1/final/'
        payload = z.read(prefix+'model.joblib')
        expected = '14ddf90e009f112e85600f9346281dd91b125da10f37ca4b211c42c03ae45401'
        assert hashlib.sha256(payload).hexdigest() == expected
        metadata = json.loads(z.read(prefix+'model_metadata.json'))
        model = joblib.load(io.BytesIO(payload))
        fixture = pd.read_parquet(io.BytesIO(z.read(prefix+'inference_fixture.parquet')))
        predictions = pd.read_parquet(io.BytesIO(z.read(prefix+'inference_fixture_predictions.parquet')))
        fixture.to_csv(out/'inference_fixture.csv', index=False)
        predictions.to_csv(out/'inference_fixture_predictions.csv', index=False)
    names = list(model.feature_names)
    with ZipFile(base/'ml_preparation_v1.zip') as z:
        canonical = pd.read_parquet(io.BytesIO(z.read('ml_preparation_v1/canonical/canonical_real.parquet')))
    eligible = canonical[canonical.train_eligible].copy()
    assert len(eligible) == 7702
    weights = {'feature_min': eligible[names].min().to_numpy(float),
               'feature_max': eligible[names].max().to_numpy(float)}
    for i, member in enumerate(model.members):
        estimator = member.estimator
        assert estimator.activation == 'relu' and estimator.out_activation_ == 'identity'
        assert len(estimator.coefs_) == 3 and list(member.feature_names) == names
        prep = member.preprocessor
        for key, value in [('mean',prep.feature_mean_),('scale',prep.feature_scale_),
                           ('ymean',prep.target_mean_),('yscale',prep.target_scale_)]:
            weights[f'{key}_{i}'] = np.asarray(value)
        for j,(w,b) in enumerate(zip(estimator.coefs_,estimator.intercepts_)):
            weights[f'w_{i}_{j}'],weights[f'b_{i}_{j}'] = w,b
    np.savez_compressed(out/'weights.npz', **weights)
    write_json(out/'model.json', {'source_model_sha256': expected,
        'weights_sha256': hashlib.sha256((out/'weights.npz').read_bytes()).hexdigest(),
        'feature_names': names, 'fit_rows':7702, 'ensemble_members':3,
        'output_frame':'training_vive_world', 'output_unit':'mm',
        'historical_segment_loso_rmse_mm':14.262783, 'independent_control_validation':False})
    portable = Model(out)
    actual = np.array([portable.predict(row) for row in fixture[names].to_dict('records')])
    expected_fixture = predictions[[f'pred_vive_world_{a}_mm' for a in 'xyz']].to_numpy()
    error = float(np.max(np.abs(actual-expected_fixture)))
    assert error < 1e-7, error
    # Select ONLY by geometry, joint margins, and data availability, never by model error.
    with ZipFile(base/'raw_sessions_archive.zip') as z:
        plan_payload = z.read('sessions/real_motion_demo/target_plan_segment036.json')
        plan = json.loads(plan_payload)
        # Freeze a real STATE sample for parser/rig checks.
        with gzip.GzipFile(fileobj=z.open('sessions/real_motion_demo/encoder_segment036.jsonl.gz')) as f:
            state = json.loads(f.readline())['raw_line']
        (out/'historical_state.txt').write_text(state, encoding='utf-8')
    geom = {'link_lengths_mm':[233.,215.], 'marker_offset_tip_mm':[0,0,0]}
    limits = np.array([[1160,3083],[1202,3186],[1937,3968]])
    counts = eligible.groupby('target_id').size()
    candidates = []
    for t in plan:
        raw = np.array(t['raw_ticks'])
        sv = np.linalg.svd(jacobian(raw,geom), compute_uv=False)
        if (raw>limits[:,0]+100).all() and (raw<limits[:,1]-100).all() and sv[-1]>.035 and sv[0]/sv[-1]<20 and counts.get(t['target_id'],0)>=50:
            candidates.append(t)
    assert len(candidates)>=5
    points = np.array([t['p_xyz_mm'] for t in candidates])
    chosen = [int(np.argmin(np.linalg.norm(points-points.mean(axis=0),axis=1)))]
    while len(chosen)<5:
        distances = np.min(np.linalg.norm(points[:,None]-points[chosen],axis=2),axis=1)
        chosen.append(int(np.argmax(distances)))
    targets = [candidates[i] for i in chosen]
    for t in targets:
        t['eligible_trial_count'] = int(counts[t['target_id']])
        t['selection_reason'] = 'geometry-only spread; >100 ticks from limits; nonsingular rigid FK'
    write_json(out/'targets.json', targets)
    replay=[]
    for t in targets:
        rows=eligible[eligible.target_id==t['target_id']].sort_values(['segment_id','t_start_ns'])
        row=rows.iloc[0]
        replay.append({'target_id':t['target_id'],'trial_uid':row.trial_uid,'segment_id':int(row.segment_id),
                       'raw_ticks':[float(row[f'encoder_tick_j{j}']) for j in range(3)],
                       'features':{n:float(row[n]) for n in names},
                       'measured_world_mm':[float(row[f'vive_pos_raw_{a}_mm']) for a in 'xyz']})
    write_json(out/'replay.json',replay)
    write_json(out/'build_report.json', {'fixture_rows':len(fixture),'max_fixture_difference_mm':error,
        'eligible_rows':len(eligible),'target_ids':[t['target_id'] for t in targets],
        'plan_member':'sessions/real_motion_demo/target_plan_segment036.json',
        'plan_sha256':hashlib.sha256(plan_payload).hexdigest(),
        'source_model_sha256':expected,'hardware_commands_sent':0,
        'scope':'model export parity, not live control validation'})
    print(json.dumps(read_report(out/'build_report.json'),indent=2))

def read_report(path):
    return json.loads(path.read_text(encoding='utf-8'))

if __name__=='__main__':
    main()
