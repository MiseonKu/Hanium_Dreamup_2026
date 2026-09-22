"""CLI: python tools/run_mock_session.py --duration-s 60

Runs `experiment/session_controller.py`'s `SessionController` in `mock`
mode for a fixed duration, wired to `mocks/*.py` sources, then a clean
shutdown (master spec section 16). Main smoke-test entry point for the
whole system.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment.session_controller import SessionController


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-s", type=float, default=60.0)
    parser.add_argument("--sessions-root", default=None, help="override where session_YYYYMMDD_NNN/ is created")
    args = parser.parse_args(argv)

    controller = SessionController("mock", sessions_root=args.sessions_root)
    print(f"session: {controller.session_id}")
    print(f"dir:     {controller.session_dir}")
    controller.start()
    print(f"collector_status: {controller._collector_status}")
    controller.run_for(args.duration_s)

    manifest = json.loads(controller.manifest_path.read_text(encoding="utf-8"))
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0 if manifest["session_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
