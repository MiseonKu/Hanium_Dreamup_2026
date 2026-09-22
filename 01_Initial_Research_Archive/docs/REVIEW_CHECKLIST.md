# Review Checklist

Use this checklist for AI or human code review.

## Contract

- [ ] CSV columns remain compatible or `DATA_CONTRACT.md` and tests changed.
- [ ] Metadata JSON required keys remain present.
- [ ] Units and frame suffixes are explicit.

## Research Validity

- [ ] Tracker is not used for feedback.
- [ ] IMU-only remains ShapeFK-based.
- [ ] Condition comparison does not change downstream controller logic.
- [ ] Randomization seed and plan digest are recorded for collected trials.
- [ ] Optimization uses validation split only, not final test trials.
- [ ] Exclusions are represented by `quality_flags`, not silent row deletion.

## Code Health

- [ ] Hardware IO stays in adapters.
- [ ] Quaternion/vector math uses `math3d.py`.
- [ ] Generated data is ignored except tiny sample fixtures.
- [ ] New behavior has a synthetic test.
- [ ] New parameter search code records candidate ids and objective settings.
- [ ] New IK code logs convergence, residual error, and joint limit status.
- [ ] Simulator support is export-only unless a separate integration package is added.
- [ ] URDF/MJCF exporters preserve meter/radian units and base/tool frame names.
- [ ] Calibration validators block placeholder dates and missing calibration types.
- [ ] Reports state they are descriptive summaries only.

## Verification

- [ ] `python -m research_host.cli.agent_doctor`
- [ ] `python -m pytest tests`
- [ ] README quickstart still works when touched.
- [ ] Optional simulator replay paths are tested without requiring installed simulators.
