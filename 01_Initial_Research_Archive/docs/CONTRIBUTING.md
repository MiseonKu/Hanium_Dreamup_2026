# Contributing

## Ground Rules

- Keep realtime hardware IO behind adapters.
- Keep generated real data out of git.
- Include frame suffixes in position, attitude, error, and deflection names.
- Update tests when changing a public field, config key, or data column.
- Do not use Tracker as feedback in a condition estimator.

## Before Opening A PR

- Run `python -m pytest tests`.
- Run the sample generation and validation commands from `README.md`.
- Check that no real trial data is staged.
- Update `docs/DATA_CONTRACT.md` if any CSV or metadata field changed.
- Update `docs/GLOSSARY.md` if a new term appears in code or config.

## Adding A Sensor

Add a reader in `research_host/adapters/` that emits normalized dataclasses. Do not let serial parser details leak into ShapeFK, conditions, logging, or analysis.

## Adding A Metric

Add the metric to `analysis.py`, expose it in the relevant CLI only if needed, and add a synthetic test with known expected values.
