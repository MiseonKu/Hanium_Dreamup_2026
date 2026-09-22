"""Hardware-motion double-arm gate (master spec section 22).

`experiment` mode is the only mode that would eventually move hardware
(this task never actually moves any). Before anything resembling a
"dispatch next target" action is allowed, BOTH the config flag
(`hardware_motion_enabled: true`) AND the CLI flag
(`--enable-hardware-motion`) must be set simultaneously -- either alone
blocks it. ponytail: this is a pure two-argument AND; no config-loading
machinery lives here, the caller (session_controller.py, built separately)
is responsible for reading the config file and argparse and handing both
booleans in.
"""
from __future__ import annotations


def hardware_motion_allowed(config_flag: bool, cli_flag: bool) -> bool:
    """True only when both flags explicitly enable hardware motion."""
    return bool(config_flag) and bool(cli_flag)


def assert_hardware_motion_allowed(config_flag: bool, cli_flag: bool) -> None:
    """Raise `PermissionError` unless hardware motion is double-armed. Call
    this immediately before any action that would dispatch a target to real
    hardware."""
    if not hardware_motion_allowed(config_flag, cli_flag):
        raise PermissionError(
            "hardware motion blocked: requires BOTH config hardware_motion_enabled=true "
            "AND --enable-hardware-motion (master spec section 22)"
        )


def _self_check() -> None:
    assert hardware_motion_allowed(True, True) is True
    assert hardware_motion_allowed(True, False) is False
    assert hardware_motion_allowed(False, True) is False
    assert hardware_motion_allowed(False, False) is False

    assert_hardware_motion_allowed(True, True)  # must not raise
    for config_flag, cli_flag in ((True, False), (False, True), (False, False)):
        try:
            assert_hardware_motion_allowed(config_flag, cli_flag)
        except PermissionError:
            continue
        raise AssertionError(f"expected PermissionError for ({config_flag}, {cli_flag})")

    print("safety_gate.py self-check OK")


if __name__ == "__main__":
    _self_check()
