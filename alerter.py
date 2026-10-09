"""Phase 7: alert dispatch on a confirmed violent event.

When the temporal gate confirms an event (its `fired` edge), this sends a
proof-of-concept email with the timestamp and a snapshot of the moment, saves the
snapshot locally, writes an audit-log row, and retries on failure so a confirmed
event is never silently lost. This is an academic stand-in for a real dispatch to
emergency services -- the report notes a real deployment would keep a human in the loop.

Email uses Python's stdlib smtplib over TLS -> NO extra packages to install.

Credentials live in .env (never the repo); src/config.py loads them into cfg["secrets"]:
  EMAIL_USER            your Gmail address
  EMAIL_APP_PASSWORD    a Google *App Password* (requires 2-Step Verification on the account;
                        your normal password will NOT work)
  ALERT_RECIPIENT       where alerts go (can be the same as EMAIL_USER)

Quick credential test (sends one real email, no detection needed):
    python -m src.alerter --test
"""
from __future__ import annotations

import csv
import smtplib
import ssl
import threading
import time
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path

from src.config import ROOT, load_config

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587


class Alerter:
    def __init__(self, cfg: dict):
        acfg = cfg.get("alert", {})
        self.enabled = bool(acfg.get("enabled", False))
        self.channels = [c.lower() for c in acfg.get("channels", ["email"])]
        self.retries = int(acfg.get("retries", 3))
        self.retry_wait = float(acfg.get("retry_wait_seconds", 2.0))

        sec = cfg.get("secrets", {})
        self.user = sec.get("email_user")
        self.app_password = sec.get("email_app_password")
        self.recipient = sec.get("alert_recipient") or self.user

        logs = ROOT / cfg["paths"]["logs_root"]
        logs.mkdir(parents=True, exist_ok=True)
        self.audit_path = logs / "alerts_audit.csv"
        self.snap_dir = ROOT / cfg["paths"]["outputs_root"] / "snapshots"
        self.snap_dir.mkdir(parents=True, exist_ok=True)
        self._audit_lock = threading.Lock()   # audit CSV is written from worker threads
        self._threads = []                     # background send threads (joined at shutdown)

        if not self.audit_path.exists():
            with open(self.audit_path, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["wall_clock", "source", "event_time_s", "raw", "smoothed",
                     "channel", "status", "attempts", "snapshot"])

    # ------------------------------------------------------------------ #
    def _audit(self, source, event, channel, status, attempts, snap):
        with self._audit_lock:
            with open(self.audit_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    datetime.now().isoformat(timespec="seconds"), source,
                    round(float(event.get("time", 0.0)), 2),
                    round(float(event.get("raw", 0.0)), 4),
                    round(float(event.get("smoothed", 0.0)), 4),
                    channel, status, attempts, snap or "",
                ])

    def _save_snapshot(self, snapshot_bgr, event):
        """Save the moment to disk (report-useful + a fallback if email fails)."""
        if snapshot_bgr is None:
            return None, None
        import cv2  # lazy: only needed when there's a frame
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.snap_dir / f"alert_{stamp}_t{event.get('time', 0):.0f}s.jpg"
        cv2.imwrite(str(path), snapshot_bgr)
        ok, buf = cv2.imencode(".jpg", snapshot_bgr)
        return str(path), (buf.tobytes() if ok else None)

    def _build_email(self, event, source, jpeg_bytes, snap_path):
        subject = "[VIOLENCE ALERT] Confirmed event detected"
        body = (
            "Automated proof-of-concept alert (academic project).\n\n"
            f"Detected at:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Source:        {source}\n"
            f"Event time:    {event.get('time', 0.0):.1f}s into the stream\n"
            f"Confidence:    raw {event.get('raw', 0):.2f} / "
            f"smoothed {event.get('smoothed', 0):.2f}\n\n"
            "This message stands in for a dispatch to emergency services. "
            "A real deployment would route this through a human verifier.\n"
        )
        msg = EmailMessage()
        msg["From"] = self.user
        msg["To"] = self.recipient
        msg["Subject"] = subject
        msg.set_content(body)
        if jpeg_bytes:
            msg.add_attachment(jpeg_bytes, maintype="image", subtype="jpeg",
                               filename=Path(snap_path).name if snap_path else "snapshot.jpg")
        return msg

    def _send_email(self, msg):
        ctx = ssl.create_default_context()
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as s:
            s.starttls(context=ctx)
            s.login(self.user, self.app_password)
            s.send_message(msg)

    # ------------------------------------------------------------------ #
    def send_alert(self, event: dict, snapshot_bgr=None, source: str = "?") -> bool:
        """Dispatch an alert for a confirmed event. Returns True on success.

        Safe to call unconditionally from the inference loop: if alerts are disabled
        or credentials are missing it logs and returns False rather than raising.
        """
        if not self.enabled:
            return False

        snap_path, jpeg_bytes = self._save_snapshot(snapshot_bgr, event)

        if "email" not in self.channels:
            self._audit(source, event, "none", "skipped:no_email_channel", 0, snap_path)
            return False
        if not (self.user and self.app_password and self.recipient):
            print("  [alert] email skipped: missing EMAIL_USER / EMAIL_APP_PASSWORD / "
                  "ALERT_RECIPIENT in .env")
            self._audit(source, event, "email", "skipped:no_credentials", 0, snap_path)
            return False

        msg = self._build_email(event, source, jpeg_bytes, snap_path)
        last_err = ""
        for attempt in range(1, self.retries + 1):
            try:
                self._send_email(msg)
                print(f"  [alert] email sent to {self.recipient} "
                      f"(attempt {attempt}){' + snapshot' if jpeg_bytes else ''}")
                self._audit(source, event, "email", "sent", attempt, snap_path)
                return True
            except Exception as e:  # noqa: BLE001
                last_err = type(e).__name__
                print(f"  [alert] email attempt {attempt}/{self.retries} failed: {last_err}")
                if attempt < self.retries:
                    time.sleep(self.retry_wait)
        print(f"  [alert] all {self.retries} attempts failed; event kept locally "
              f"({snap_path or 'no snapshot'})")
        self._audit(source, event, "email", f"failed:{last_err}", self.retries, snap_path)
        return False

    def send_alert_async(self, event: dict, snapshot_bgr=None, source: str = "?"):
        """Fire the alert on a background thread so the caller (the video loop) never
        blocks on the network. The snapshot is copied here so the loop can reuse its
        frame immediately. Threads are joined by flush() at shutdown."""
        if not self.enabled:
            return None
        snap = snapshot_bgr.copy() if snapshot_bgr is not None else None
        t = threading.Thread(target=self.send_alert, args=(event, snap, source), daemon=True)
        t.start()
        self._threads = [x for x in self._threads if x.is_alive()] + [t]
        return t

    def flush(self, timeout: float = 15.0):
        """Wait for any in-flight alert sends to finish (call at shutdown)."""
        for t in self._threads:
            t.join(timeout=timeout)


if __name__ == "__main__":
    # Credential smoke test: forces alerts on and sends one real email.
    cfg = load_config()
    cfg.setdefault("alert", {})["enabled"] = True
    if "email" not in [c.lower() for c in cfg["alert"].get("channels", ["email"])]:
        cfg["alert"]["channels"] = ["email"]
    a = Alerter(cfg)
    print(f"sending test alert to {a.recipient} ...")
    fake = {"time": 0.0, "raw": 0.99, "smoothed": 0.95}
    ok = a.send_alert(fake, snapshot_bgr=None, source="alerter self-test")
    print("OK: test email sent." if ok else
          "FAILED: check .env (EMAIL_USER / EMAIL_APP_PASSWORD / ALERT_RECIPIENT) "
          "and that the App Password is correct.")
