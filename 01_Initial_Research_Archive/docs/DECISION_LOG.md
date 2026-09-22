# Decision Log

This log records decisions agents should not reopen without an explicit user
request.

## 2026-05-01: Offline-First Baseline

Decision: Build the first research foundation as an offline package with
synthetic data and CSV readers.

Reason: The research contracts must be stable before realtime multi-device IO
is added.

## 2026-05-01: CSV + JSON Required Contract

Decision: Required storage is processed/raw CSV plus metadata JSON. Parquet can
be added later only as an optional export.

Reason: CSV + JSON is easy for new collaborators and AI agents to inspect.

## 2026-05-01: Tracker Is Evaluation Only

Decision: Tracker measurements must not be used as controller feedback.

Reason: The research design evaluates encoder-only, IMU-only, and fusion
conditions against Tracker ground truth.

## 2026-05-01: IMU-Only Means ShapeFK

Decision: IMU-only uses three-IMU attitude arrays and arc-length ShapeFK. It is
not acceleration double integration.

Reason: This avoids strapdown INS drift and matches the research methodology.

## 2026-05-01: Realtime IO Behind Adapters

Decision: ST3215, MPU6050, WT901C, and Tracker IO must enter through adapter
interfaces.

Reason: Core math, logging, and analysis must remain testable without hardware.
