"""Package source and data only, excluding environments and generated run logs."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile, ZIP_DEFLATED

ROOT=Path(__file__).resolve().parent

def main():
    tests=subprocess.run([sys.executable,'-m','unittest','-v','test_demo','test_coordinate_frame'],cwd=ROOT,capture_output=True,text=True)
    if tests.returncode:
        print(tests.stdout,tests.stderr)
        raise SystemExit(tests.returncode)
    smoke=subprocess.run([sys.executable,'run_demo.py','--dry-run'],cwd=ROOT,capture_output=True,text=True)
    synthetic=subprocess.run([sys.executable,'run_demo.py','--simulate'],cwd=ROOT,capture_output=True,text=True)
    # The stock config MUST refuse before asking for input or opening ports.
    blocked=subprocess.run([sys.executable,'run_demo.py','--live'],cwd=ROOT,capture_output=True,text=True,input='')
    approximate=subprocess.run([sys.executable,'run_demo.py','--preview-correction','--config','config.approximate.json'],cwd=ROOT,capture_output=True,text=True)
    approximate_blocked=subprocess.run([sys.executable,'run_demo.py','--live','--config','config.approximate.json'],cwd=ROOT,capture_output=True,text=True,input='')
    assert smoke.returncode==0 and synthetic.returncode==0
    assert blocked.returncode==2 and 'PRECHECK BLOCKED (no ports opened)' in blocked.stdout
    assert approximate.returncode==0
    assert approximate_blocked.returncode==2 and 'PRECHECK BLOCKED (no ports opened)' in approximate_blocked.stdout
    report={'tests_passed':True,'hardware_executed':False,
            'default_live_mode':'blocked_pending_actual_calibration_and_rig_checks',
            'test_output':tests.stdout+tests.stderr,'offline_replay':smoke.stdout,
            'synthetic_controller':synthetic.stdout,'live_precheck':blocked.stdout,
            'approximate_correction_preview':approximate.stdout,
            'approximate_live_precheck':approximate_blocked.stdout,
            'model_export':json.loads((ROOT/'data/build_report.json').read_text(encoding='utf-8'))}
    (ROOT/'verification_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    files=sorted(p for p in ROOT.iterdir() if p.is_file() and p.suffix in ('.py','.json','.md','.txt','.cmd','.docx'))
    files+=sorted(p for p in (ROOT/'data').rglob('*') if p.is_file())
    files+=sorted(p for p in (ROOT/'coordinate_analysis').glob('*') if p.is_file() and p.suffix in ('.json','.md'))
    manifest={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    target=ROOT.parent/(ROOT.name+'.zip')
    with ZipFile(target,'w',ZIP_DEFLATED) as z:
        for p in files:z.write(p,ROOT.name+'/'+p.relative_to(ROOT).as_posix())
        z.writestr(ROOT.name+'/checksums.json',json.dumps(manifest,indent=2))
    with ZipFile(target) as z:
        assert z.testzip() is None
        for name,digest in manifest.items():
            assert hashlib.sha256(z.read(ROOT.name+'/'+name)).hexdigest()==digest
    print(json.dumps({'zip':str(target),'bytes':target.stat().st_size,'members':len(files)+1,
                      'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'tests':'passed'},indent=2))

if __name__=='__main__':main()
