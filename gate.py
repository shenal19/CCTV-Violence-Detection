"""Phase 5: temporal confidence gate (the project's core novelty).

The model emits a violence probability per sliding window. Those per-window scores
are noisy: a single window can spike high. Firing an alert on any one high score is
exactly what produces the 17% per-clip false-positive rate. The gate turns that noisy
per-window signal into stable, confirmed events using three filters in series:

  1. SMOOTH the incoming scores (EMA or moving average) to absorb single-window spikes.
  2. CONFIRM only after the smoothed score stays above `confirm_threshold` for
     `consecutive_required` windows in a row.
  3. COOLDOWN: after firing, stay quiet for `cooldown_seconds` so one event = one alert.

Real violence persists across many windows and survives all three filters; random
spikes do not. That is what converts a per-clip classifier into a low-false-alarm
event detector.

Pure Python — no torch — so it is trivial to unit-test and reuse in both the offline
inference path and the live webcam demo.
"""
from __future__ import annotations

from collections import deque


class TemporalGate:
    def __init__(self, window_size=8, smoothing="ema", confirm_threshold=0.8,
                 consecutive_required=5, cooldown_seconds=30.0, seconds_per_step=1.0):
        if smoothing not in ("ema", "mean"):
            raise ValueError("smoothing must be 'ema' or 'mean'")
        self.window_size = int(window_size)
        self.smoothing = smoothing
        self.confirm_threshold = float(confirm_threshold)
        self.consecutive_required = int(consecutive_required)
        self.cooldown_seconds = float(cooldown_seconds)
        self.seconds_per_step = float(seconds_per_step)
        self.alpha = 2.0 / (self.window_size + 1.0)  # EMA span == window_size
        self.reset()

    @classmethod
    def from_config(cls, cfg, seconds_per_step=1.0):
        g = cfg["gate"]
        return cls(window_size=g["window_size"], smoothing=g["smoothing"],
                   confirm_threshold=g["confirm_threshold"],
                   consecutive_required=g["consecutive_required"],
                   cooldown_seconds=g["cooldown_seconds"],
                   seconds_per_step=seconds_per_step)

    def reset(self):
        self._buf = deque(maxlen=self.window_size)
        self._ema = None
        self.consecutive = 0
        self.alarm_active = False
        self.last_fire_time = None
        self._t = 0.0

    def _smooth(self, score):
        if self.smoothing == "mean":
            self._buf.append(score)
            return sum(self._buf) / len(self._buf)
        self._ema = score if self._ema is None else \
            self.alpha * score + (1.0 - self.alpha) * self._ema
        return self._ema

    def update(self, score, timestamp=None):
        """Feed one per-window violence probability; returns the gate state.

        `fired` is True only on the single step a NEW event is confirmed — that is the
        edge the alert module (Phase 7) acts on. `alarm_active` and `smoothed` drive the
        on-screen overlay (Phase 6). Pass `timestamp` (seconds) on a live stream; if
        omitted, the gate advances its own clock by `seconds_per_step`.
        """
        if timestamp is None:
            t = self._t
            self._t += self.seconds_per_step
        else:
            t = float(timestamp)

        s = self._smooth(float(score))
        fired = False

        if s >= self.confirm_threshold:
            self.consecutive += 1
            if self.consecutive >= self.consecutive_required and not self.alarm_active:
                if self.last_fire_time is None or \
                        (t - self.last_fire_time) >= self.cooldown_seconds:
                    fired = True
                    self.alarm_active = True
                    self.last_fire_time = t
        else:
            self.consecutive = 0
            self.alarm_active = False

        return {
            "time": round(t, 3),
            "raw": round(float(score), 4),
            "smoothed": round(s, 4),
            "consecutive": self.consecutive,
            "alarm_active": self.alarm_active,
            "fired": fired,
        }
