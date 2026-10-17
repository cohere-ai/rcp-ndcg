"""Run every reproduction check and print one summary line per script.

Usage::

    python experiments/run_all.py

Exits 1 if any script reports a value outside its tolerance.
"""

from __future__ import annotations

import external_judges
import human_study
import leaderboards


def main() -> int:
    results = {}
    for name, run in (
        ("leaderboards", leaderboards.main),
        ("human_study", human_study.main),
        ("external_judges", external_judges.main),
    ):
        print(f"\n######## {name}")
        results[name] = run([])
    print("\n######## summary")
    for name, code in results.items():
        print(f"  {name:<16} {'ok' if code == 0 else 'FAILED'}")
    return 1 if any(results.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
