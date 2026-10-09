"""Phase 5 check: verify the temporal gate on synthetic score streams.

Confirms the 'done when' criterion without needing a model or video: the gate fires
once per real event, ignores lone spikes, and respects the cooldown.

    python scripts/test_gate.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.gate import TemporalGate  # noqa: E402


def run(scores, timestamps=None, **kw):
    g = TemporalGate(**kw)
    fires = []
    for i, s in enumerate(scores):
        ts = timestamps[i] if timestamps else None
        st = g.update(s, ts)
        if st["fired"]:
            fires.append(round(st["time"], 2))
    return fires


def check(name, ok):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    return ok


def main():
    base = dict(window_size=8, smoothing="ema", confirm_threshold=0.8,
                consecutive_required=5, cooldown_seconds=30.0, seconds_per_step=8 / 30)
    ok = True

    # 1. lone spikes in a low baseline -> never fire
    s = [0.1] * 30
    s[7], s[18] = 0.95, 0.97
    f = run(s, **base)
    ok &= check("lone spikes rejected (0 fires)", len(f) == 0)

    # 2. one sustained event -> exactly one fire
    s = [0.1] * 10 + [0.9] * 30 + [0.1] * 10
    f = run(s, **base)
    ok &= check("sustained event fires exactly once", len(f) == 1)

    # 3. second event arriving within the cooldown -> suppressed
    s = [0.1] * 5 + [0.9] * 20 + [0.1] * 10 + [0.9] * 20 + [0.1] * 5
    f = run(s, **base)
    ok &= check("second event within cooldown suppressed (1 fire)", len(f) == 1)

    # 4. two events more than cooldown apart -> both fire
    #    explicit timestamps with a 35s jump placed just before the second event
    s = [0.1] * 5 + [0.9] * 20 + [0.1] * 10 + [0.9] * 20
    ts, t, step = [], 0.0, 8 / 30
    for i in range(len(s)):
        ts.append(t)
        t += 35.0 if i == (5 + 20 + 10 - 1) else step
    f = run(s, timestamps=ts, **base)
    ok &= check("two events > cooldown apart both fire (2 fires)", len(f) == 2)

    print("\nGATE TESTS PASS" if ok else "\nSOME GATE TESTS FAILED")


if __name__ == "__main__":
    main()
