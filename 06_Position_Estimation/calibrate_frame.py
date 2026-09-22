"""Fit a rigid transform from independently measured rigid fiducials, NOT sagged trials."""
import argparse
import json
from pathlib import Path
import numpy as np
from demo_core import DemoError, read_json

def fit_frame(model_mm, world_mm, max_residual_mm=2.0):
    a,b=np.asarray(model_mm,float),np.asarray(world_mm,float)
    if a.ndim!=2 or a.shape[1]!=3 or a.shape!=b.shape or len(a)<4:
        raise DemoError('At least four matching independently measured XYZ fiducials required')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise DemoError('Calibration contains NaN/Inf')
    aa,bb=a-a.mean(axis=0),b-b.mean(axis=0)
    if np.linalg.svd(aa,compute_uv=False)[1]<1 or np.linalg.svd(bb,compute_uv=False)[1]<1:
        raise DemoError('Fiducials are collinear or too tightly clustered')
    u,_,vt=np.linalg.svd(aa.T@bb)
    sign=np.diag([1.,1.,np.linalg.det(vt.T@u.T)])
    rotation=vt.T@sign@u.T
    translation=b.mean(axis=0)-rotation@a.mean(axis=0)
    errors=np.linalg.norm(a@rotation.T+translation-b,axis=1)
    if not np.isfinite(max_residual_mm) or max_residual_mm<=0 or np.max(errors)>max_residual_mm:
        raise DemoError(f'Independent fiducial residual exceeds limit: {errors.tolist()} mm')
    return {'world_from_model_rotation':rotation.tolist(),
            'world_from_model_translation_mm':translation.tolist(),
            'fiducial_residuals_mm':errors.tolist(),
            'max_residual_mm':float(np.max(errors)),
            'note':'Numerical registration only. Does not verify historical frame equivalence or marker mounting.'}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path)
    p.add_argument('--output',type=Path,default=Path('frame_fit.json'))
    p.add_argument('--max-residual-mm',type=float,default=2.)
    args=p.parse_args();data=read_json(args.input)
    if data.get('independent_rigid_fiducials') is not True or not data.get('measurement_record'):
        raise DemoError('Use independent rigid fiducials and record how they were measured')
    result=fit_frame(data['model_frame_points_mm'],data['training_world_points_mm'],args.max_residual_mm)
    result['measurement_record']=data['measurement_record']
    with args.output.open('x',encoding='utf-8') as f:
        json.dump(result,f,indent=2,allow_nan=False)
    print(f'Written {args.output}. Configuration flags are NOT enabled automatically.')

if __name__=='__main__':
    try:main()
    except DemoError as e:raise SystemExit(str(e))
