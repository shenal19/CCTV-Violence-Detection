"""Phase 6: real-time inference pipeline (clips + webcam) with the temporal gate.

Runs the trained R(2+1)D-18 (outputs/checkpoints/best.pt) over a video source using a
rolling window, feeds each window's P(Violence) into the temporal gate (src/gate.py),
and draws an OpenCV overlay (raw prob, smoothed score, consecutive count, alarm state).
On a confirmed event the gate's `fired` flag goes True for one step -- that is the edge
the Phase 7 alert module will hook into; here we just log it and flash the banner.

Reuses the EXACT eval preprocessing from src/dataset.py (resize short side ~128 ->
center-crop 112 -> /255 -> Kinetics normalize -> (C,T,H,W)). OpenCV frames are BGR;
the model was trained on RGB (decord), so every frame is converted to RGB before
inference and kept BGR only for display.

Examples (run from the project root VD, with .venv active):
    python scripts/infer.py --source RWF-2000/val/Fight/<some_clip>.avi
    python scripts/infer.py --source webcam
    python scripts/infer.py --source <clip> --no-display --save outputs/demo_annotated.mp4

The "done when": running on a known violent clip shows the overlay and the gate confirms.

Temporal-window note: training samples 16 frames across the whole ~5s clip, so the model
saw motion at ~0.3s frame spacing. A live buffer of 16 *consecutive* frames spans only
~0.5s and under-represents that motion. So we hold a window of `--window-seconds` of
video and uniformly sample 16 frames across it. If the gate does NOT fire on a clip you
know is violent, RAISE --window-seconds (toward ~3-4) so the motion matches training; if
it fires too eagerly on calm footage, LOWER it. This is the main knob for this phase.
"""
import argparse
import csv
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2                                                            # noqa: E402
import numpy as np                                                   # noqa: E402
import torch                                                         # noqa: E402

from src.config import ROOT, load_config                            # noqa: E402
from src.dataset import _resize_short_side, _crop, MEAN, STD        # noqa: E402
from src.gate import TemporalGate                                   # noqa: E402
from src.model import build_model                                   # noqa: E402
from src.alerter import Alerter                                     # noqa: E402

POS = 1  # positive class = Violence (class_names = ["NonViolence", "Violence"])

# =========================================================================== #
#  EDIT THIS: paste the path to the video you want to analyze.
#  - Use a raw string (r"...") so Windows backslashes are safe.
#  - Set it to "webcam" to use the live camera instead of a file.
#  A --source flag on the command line still overrides this if you pass one.
# =========================================================================== #
VIDEO_PATH = r"C:\Users\tabri\OneDrive\Desktop\VIT\SUMM\VD\RWF-2000\val\Fight\0Ow4cotKOuw_3.avi"

#  ---- tuning knobs (None = use config.yaml / fps defaults) ----
#  WINDOW_SECONDS: seconds of video each prediction looks at. Your model was
#    trained on frames spanning the whole ~5s clip, so a TOO-SHORT window makes
#    it under-react. If violence isn't detected, RAISE this (try 3.0-4.0).
WINDOW_SECONDS = 3.0
#  PLAYBACK_SPEED: 1.0 = real time. Use 0.5 for slow-mo analysis, 2.0 for fast.
PLAYBACK_SPEED = 1.0
#  Gate overrides for the demo. The config defaults (0.8 / 5) are tuned for LOW
#  false alarms. If the gate is too strict to fire on clips you know are violent,
#  lower these (e.g. 0.6 / 3). Higher = stricter/fewer alarms, lower = more
#  sensitive. None keeps the config value.
CONFIRM_THRESHOLD = None     # e.g. 0.6 ; None -> cfg.gate.confirm_threshold (0.8)
CONSECUTIVE_REQUIRED = None  # e.g. 3   ; None -> cfg.gate.consecutive_required (5)

#  ---- input-quality gate (rejects blank / blocked / static screens) ----
#  The model only knows "violence vs non-violence" on REAL footage. A blocked
#  camera (dark noise) or a solid/blank end-card is out-of-distribution and can
#  score as violent. These floors suppress windows that aren't valid footage, so
#  the alarm cannot fire on them. Calibrated: real footage measures ~82 / ~55 / ~9.
QUALITY_CHECK = True
QUALITY_DARK_LUMA = 20.0    # mean brightness below this -> too dark (blocked camera)
QUALITY_UNIFORM_STD = 15.0  # spatial detail below this  -> too uniform (blank/solid screen)
QUALITY_STATIC_DIFF = 1.0   # frame-to-frame motion below this -> frozen image / static card
# =========================================================================== #


# --------------------------------------------------------------------------- #
# setup helpers
# --------------------------------------------------------------------------- #
def resolve_device(cfg):
    want = str(cfg.get("project", {}).get("device", "auto")).lower()
    if want == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def open_source(source):
    """Return (cv2.VideoCapture, is_webcam, fps). Accepts 'webcam', a device index, or a path."""
    s = str(source).strip()
    if s.lower() in ("webcam", "cam", "camera") or s.isdigit():
        idx = 0 if s.lower() in ("webcam", "cam", "camera") else int(s)
        cap = cv2.VideoCapture(idx)
        is_webcam = True
    else:
        if not Path(s).exists():
            raise SystemExit(f"Source not found: {s}")
        cap = cv2.VideoCapture(s)
        is_webcam = False
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps <= 1:  # webcams often report 0 / NaN
        fps = 30.0
    return cap, is_webcam, float(fps)


def load_model(cfg, device):
    ckpt_path = ROOT / cfg["paths"]["checkpoints_root"] / "best.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}\nTrain first (scripts/train.py).")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"loaded {ckpt_path.name} (epoch {ckpt.get('epoch', '?')}, "
          f"val F1 {ckpt.get('val_metrics', {}).get('f1', '?')})")
    return model


# --------------------------------------------------------------------------- #
# preprocessing + inference (mirrors src/dataset.py eval path exactly)
# --------------------------------------------------------------------------- #
def sample_window(buffer, num_frames):
    """Uniformly sample `num_frames` frames (T,H,W,C) from the rolling buffer."""
    n = len(buffer)
    idx = np.linspace(0, n - 1, num_frames).astype(int)
    return np.stack([buffer[i] for i in idx])  # (T, H, W, C) uint8 RGB


def preprocess(frames_rgb, img_size):
    """(T,H,W,C) uint8 RGB -> (1,C,T,H,W) float tensor, identical to dataset eval path."""
    short = int(round(img_size * 1.14))                     # ~128 for 112
    f = _resize_short_side(frames_rgb, short)
    f = _crop(f, img_size, training=False)                  # center crop
    x = f.astype(np.float32) / 255.0
    x = (x - MEAN) / STD                                    # Kinetics normalize
    x = torch.from_numpy(np.ascontiguousarray(x)).permute(3, 0, 1, 2)  # (C,T,H,W)
    return x.unsqueeze(0).contiguous()                      # (1,C,T,H,W)


@torch.no_grad()
def violence_prob(model, frames_rgb, cfg, device, use_amp):
    x = preprocess(frames_rgb, cfg["data"]["img_size"]).to(device)
    with torch.amp.autocast("cuda", enabled=use_amp):
        logits = model(x)
    return torch.softmax(logits.float(), dim=1)[0, POS].item()


def frame_quality(window_rgb):
    """Is this 16-frame window valid footage, or a blank/blocked/static screen?

    Returns {valid, reason, mean_luma, spatial_std, temporal}. Cheap: grayscale
    brightness, average per-frame spatial std, and mean frame-to-frame motion.
    The model is only trusted on valid windows; everything else is suppressed
    (fed 0 to the gate) because violence is a high-motion event and a blank or
    frozen screen cannot contain one.
    """
    g = window_rgb[:, ::8, ::8, :].astype(np.float32)  # subsample: ~30x cheaper, same stats
    gray = 0.299 * g[..., 0] + 0.587 * g[..., 1] + 0.114 * g[..., 2]  # (T,h,w)
    mean_luma = float(gray.mean())
    spatial_std = float(gray.reshape(gray.shape[0], -1).std(axis=1).mean())
    temporal = float(np.mean(np.abs(np.diff(gray, axis=0)))) if gray.shape[0] > 1 else 0.0

    reasons = []
    if mean_luma < QUALITY_DARK_LUMA:
        reasons.append("too dark")
    if spatial_std < QUALITY_UNIFORM_STD:
        reasons.append("too uniform")
    if temporal < QUALITY_STATIC_DIFF:
        reasons.append("no motion")
    valid = (not QUALITY_CHECK) or (len(reasons) == 0)
    return {"valid": valid, "reason": ", ".join(reasons) if reasons else "ok",
            "mean_luma": mean_luma, "spatial_std": spatial_std, "temporal": temporal}


@torch.no_grad()
def whole_clip_verdict(frames_rgb_all, model, cfg, device, use_amp):
    """Accurate clip-level verdict via multi-clip averaging (same as evaluate.py).

    Samples cfg.eval.num_clips evenly-spaced 16-frame views spanning the WHOLE clip
    and averages P(Violence). This mirrors the validated test-time procedure, so it's
    the trustworthy "is this clip violent?" answer -- separate from the streaming gate.
    """
    nf = cfg["data"]["num_frames"]
    n_views = cfg["eval"]["num_clips"]
    total = len(frames_rgb_all)
    if total < nf:
        return None
    phases = (np.arange(n_views) + 0.5) / n_views     # 5 views -> 0.1,0.3,...,0.9
    seg = total / nf
    probs = []
    for ph in phases:
        idx = np.minimum(((np.arange(nf) + ph) * seg).astype(int), total - 1)
        view = np.stack([frames_rgb_all[i] for i in idx])   # (T,H,W,C)
        if not frame_quality(view)["valid"]:
            continue  # don't let blank/static views skew the verdict
        probs.append(violence_prob(model, view, cfg, device, use_amp))
    if not probs:
        return None  # whole clip was low-signal
    return float(np.mean(probs))


# --------------------------------------------------------------------------- #
# overlay
# --------------------------------------------------------------------------- #
def draw_overlay(frame_bgr, state, threshold, fps_disp=None):
    """Draw label / probability / smoothed score / alarm banner onto a BGR frame."""
    h, w = frame_bgr.shape[:2]
    raw, sm = state["raw"], state["smoothed"]
    alarm = state["alarm_active"]
    violent = raw >= threshold

    green, red, white, grey = (80, 200, 80), (40, 40, 230), (255, 255, 255), (160, 160, 160)
    accent = red if (alarm or violent) else green

    # translucent header panel
    panel_h = 86
    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame_bgr, 0.45, 0, frame_bgr)

    label = "VIOLENCE" if violent else "non-violent"
    if not state.get("signal_ok", True):
        cv2.putText(frame_bgr, "LOW SIGNAL", (14, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, grey, 2, cv2.LINE_AA)
        cv2.putText(frame_bgr, f"({state.get('quality_reason', '')}) - not analysed",
                    (210, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, grey, 1, cv2.LINE_AA)
        cv2.putText(frame_bgr, "no valid video to analyse", (14, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, grey, 1, cv2.LINE_AA)
        if fps_disp is not None:
            cv2.putText(frame_bgr, f"{fps_disp:4.1f} fps", (w - 110, h - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, grey, 1, cv2.LINE_AA)
        return frame_bgr

    cv2.putText(frame_bgr, f"{label}", (14, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, accent, 2, cv2.LINE_AA)
    cv2.putText(frame_bgr, f"p={raw:.2f}", (210, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, white, 2, cv2.LINE_AA)
    cv2.putText(frame_bgr,
                f"smoothed {sm:.2f}   consec {state['consecutive']}",
                (14, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, grey, 1, cv2.LINE_AA)

    # probability bar (smoothed)
    bx, by, bw = 14, 70, w - 28
    cv2.rectangle(frame_bgr, (bx, by), (bx + bw, by + 8), (70, 70, 70), -1)
    cv2.rectangle(frame_bgr, (bx, by), (bx + int(bw * sm), by + 8), accent, -1)
    tx = bx + int(bw * threshold)
    cv2.line(frame_bgr, (tx, by - 3), (tx, by + 11), white, 1)  # threshold tick

    if alarm:
        cv2.rectangle(frame_bgr, (0, panel_h), (w, panel_h + 34), red, -1)
        cv2.putText(frame_bgr, "ALARM - VIOLENCE CONFIRMED", (14, panel_h + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, white, 2, cv2.LINE_AA)

    if fps_disp is not None:
        cv2.putText(frame_bgr, f"{fps_disp:4.1f} fps", (w - 110, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, grey, 1, cv2.LINE_AA)
    return frame_bgr


# --------------------------------------------------------------------------- #
# main loop
# --------------------------------------------------------------------------- #
def main():
    cfg = load_config()
    demo = cfg.get("demo", {})

    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=VIDEO_PATH,
                    help="'webcam', a device index, or a path to a video file "
                         "(defaults to the VIDEO_PATH set at the top of this file)")
    ap.add_argument("--stride", type=int, default=int(demo.get("stride", 8)),
                    help="run inference every N frames (rolling step)")
    ap.add_argument("--window-seconds", type=float,
                    default=(WINDOW_SECONDS if WINDOW_SECONDS is not None
                             else float(demo.get("window_seconds", 3.0))),
                    help="seconds of video each prediction looks at (motion-span knob)")
    ap.add_argument("--speed", type=float, default=PLAYBACK_SPEED,
                    help="playback speed for video files (1.0 = real time, 0.5 = slow-mo)")
    ap.add_argument("--threshold", type=float, default=float(cfg["eval"].get("threshold", 0.5)),
                    help="per-window prob for the VIOLENCE *label* (gate has its own confirm_threshold)")
    ap.add_argument("--no-display", action="store_true", help="run headless (no window)")
    ap.add_argument("--quiet", action="store_true", help="don't print every per-window score")
    ap.add_argument("--save", default=None, help="path to write an annotated output video")
    ap.add_argument("--max-frames", type=int, default=None, help="stop after N frames (testing)")
    args = ap.parse_args()

    device = resolve_device(cfg)
    use_amp = bool(cfg["train"]["amp"]) and device == "cuda"
    model = load_model(cfg, device)

    cap, is_webcam, fps = open_source(args.source)
    num_frames = cfg["data"]["num_frames"]
    buf_len = max(num_frames, int(round(args.window_seconds * fps)))
    buffer = deque(maxlen=buf_len)
    collected = [] if not is_webcam else None  # full-clip frames for the eval-style verdict

    # gate: one update per inference, i.e. every `stride` frames -> stride/fps seconds apart
    seconds_per_step = args.stride / fps
    gate = TemporalGate.from_config(cfg, seconds_per_step=seconds_per_step)
    if CONFIRM_THRESHOLD is not None:
        gate.confirm_threshold = float(CONFIRM_THRESHOLD)
    if CONSECUTIVE_REQUIRED is not None:
        gate.consecutive_required = int(CONSECUTIVE_REQUIRED)
    gate.reset()

    alerter = Alerter(cfg)
    src_label = "webcam" if is_webcam else args.source

    print(f"device {device} | amp {use_amp} | source {'webcam' if is_webcam else args.source}")
    print(f"fps {fps:.1f} | window {args.window_seconds:.1f}s ({buf_len} frames) | "
          f"infer every {args.stride} frames | label threshold {args.threshold}")
    print(f"gate: confirm>={gate.confirm_threshold} for {gate.consecutive_required} windows, "
          f"cooldown {gate.cooldown_seconds:.0f}s, seconds/step {seconds_per_step:.3f}")
    print("press 'q' in the window to quit\n" if not args.no_display else "")

    writer = None
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)

    log_dir = ROOT / cfg["paths"]["logs_root"]
    log_dir.mkdir(parents=True, exist_ok=True)
    events_path = log_dir / "inference_events.csv"
    new_log = not events_path.exists()
    events_f = open(events_path, "a", newline="")
    events_w = csv.writer(events_f)
    if new_log:
        events_w.writerow(["wall_clock", "source", "event_time_s", "raw", "smoothed"])

    frame_i = 0
    last_state = {"raw": 0.0, "smoothed": 0.0, "consecutive": 0, "alarm_active": False}
    fires = 0
    t_start = time.time()
    last_tick = t_start
    fps_disp = None
    # real-time pacing target (files only); webcam is paced by the camera itself
    frame_period = (1.0 / fps) / max(args.speed, 1e-6)

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_i += 1
            if args.max_frames and frame_i > args.max_frames:
                break

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            buffer.append(rgb)
            if collected is not None and len(collected) < 1200:  # cap ~40s @30fps
                collected.append(rgb)

            # run inference on a rolling step, once we have at least num_frames
            if len(buffer) >= num_frames and frame_i % args.stride == 0:
                window = sample_window(buffer, num_frames)
                q = frame_quality(window)
                if q["valid"]:
                    p = violence_prob(model, window, cfg, device, use_amp)
                else:
                    p = 0.0  # out-of-distribution input -> suppress, never trust it
                ts = (time.time() - t_start) if is_webcam else (frame_i / fps)
                last_state = gate.update(p, timestamp=ts)
                last_state["signal_ok"] = q["valid"]
                last_state["quality_reason"] = q["reason"]

                now = time.time()
                fps_disp = args.stride / max(now - last_tick, 1e-6)
                last_tick = now

                if not args.quiet:
                    if not q["valid"]:
                        print(f"  t={ts:5.1f}s  LOW SIGNAL ({q['reason']}) "
                              f"-- suppressed  [luma {q['mean_luma']:.0f} "
                              f"std {q['spatial_std']:.0f} mot {q['temporal']:.1f}]")
                    else:
                        bar = "#" * int(last_state["smoothed"] * 20)
                        print(f"  t={ts:5.1f}s  raw={last_state['raw']:.2f}  "
                              f"smoothed={last_state['smoothed']:.2f} |{bar:<20}|  "
                              f"consec {last_state['consecutive']}/{gate.consecutive_required}"
                              f"{'  <ALARM>' if last_state['alarm_active'] else ''}")

                if last_state["fired"]:
                    fires += 1
                    stamp = datetime.now().strftime("%H:%M:%S")
                    print(f"  [{stamp}] *** ALERT FIRED *** event_t={last_state['time']:.1f}s "
                          f"raw={last_state['raw']:.2f} smoothed={last_state['smoothed']:.2f}")
                    events_w.writerow([datetime.now().isoformat(timespec="seconds"),
                                       src_label, last_state["time"], last_state["raw"],
                                       last_state["smoothed"]])
                    events_f.flush()
                    # Phase 7: dispatch the alert on a background thread so the video
                    # loop never blocks on the network. No-op unless alert.enabled.
                    alerter.send_alert_async(last_state, snapshot_bgr=frame_bgr,
                                             source=src_label)

            frame_out = draw_overlay(frame_bgr, last_state, args.threshold, fps_disp)

            if writer is None and args.save:
                h, w = frame_out.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(args.save, fourcc, fps, (w, h))
            if writer is not None:
                writer.write(frame_out)

            if not args.no_display:
                cv2.imshow("Violence Detection (Phase 6)", frame_out)
                # pace playback to real time (files): wait the remaining frame period
                if is_webcam:
                    wait_ms = 1
                else:
                    spent = time.time() - (t_start + (frame_i - 1) * frame_period)
                    wait_ms = max(1, int((frame_period - spent) * 1000))
                if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
                    print("quit by user")
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"saved annotated video -> {args.save}")
        if not args.no_display:
            cv2.destroyAllWindows()
        events_f.close()
        alerter.flush()  # let any in-flight alert emails finish before exit

    print(f"\ndone. processed {frame_i} frames | gate fired {fires} time(s) | "
          f"events log -> {events_path}")

    # accurate clip-level verdict (files only) -- matches the validated evaluate.py path
    if collected is not None and len(collected) >= num_frames:
        if len(collected) >= 1200:
            print("(verdict uses first ~40s of the file)")
        clip_p = whole_clip_verdict(collected, model, cfg, device, use_amp)
        print("=" * 56)
        if clip_p is None:
            print("CLIP VERDICT: no valid footage found (all views were "
                  "blank/static/too dark).")
        else:
            label = "VIOLENT" if clip_p >= args.threshold else "NON-VIOLENT"
            print(f"CLIP VERDICT (multi-clip avg, {cfg['eval']['num_clips']} views): "
                  f"P(Violence) = {clip_p:.3f}  ->  {label}")
            print("This is the trustworthy clip-level answer (same method as the")
            print("RWF-2000 test eval). The streaming gate above is the live-demo path.")
        print("=" * 56)


if __name__ == "__main__":
    main()
