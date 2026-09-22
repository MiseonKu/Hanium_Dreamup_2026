"""CLI: python tools/run_soak_test.py --duration-s 1800

Extended-duration stability run (master spec section 22 `soak` mode -- 30+
minutes, no motor commands ever). Thin wrapper around the same mock-backed
`SessionController` `run_mock_session.py` uses, just under
`workspace_mode="soak"` and defaulting to a long duration.
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
    parser.add_argument("--duration-s", type=float, default=1800.0)
    parser.add_argument("--sessions-root", default=None)
    args = parser.parse_args(argv)

    controller = SessionController("soak", sessions_root=args.sessions_root)
    print(f"soak session: {controller.session_id}  dir: {controller.session_dir}  duration_s={args.duration_s}")
    controller.start()
    controller.run_for(args.duration_s)

    manifest = json.loads(controller.manifest_path.read_text(encoding="utf-8"))
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0 if manifest["session_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
