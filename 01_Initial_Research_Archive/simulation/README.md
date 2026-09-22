# Simulation Assets

This folder contains generated baseline assets for simulator integration.

Regenerate from the project root:

```powershell
$env:PYTHONPATH = ".\host"
python -m research_host.cli.export_sim_assets --robot-model long --output .\simulation\generated
python -m research_host.cli.export_sim_assets --robot-model short --output .\simulation\generated
```

The generated models are scaffolds for ROS 2/RViz, Isaac Sim, and MuJoCo. They
are not final physics-validated robot models.

Tracked generated examples include both `so101_long` and `so101_short`. Local
simulator project files, caches, or imported USD outputs should stay outside the
repository unless a later task explicitly defines a reviewable artifact.
