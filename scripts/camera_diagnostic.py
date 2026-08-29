"""Standalone camera diagnostic: verify the negotiated capture mode without MediaPipe.

Opens the camera through the same ``OpenCVVideoSource`` used by the application,
prints the negotiated backend/resolution/FPS/pixel format, measures effective
capture FPS and read failures, optionally shows the feed and saves frames, then
exits cleanly. Use this to determine whether Linux/Windows itself is producing a
good camera stream before the vision pipeline is involved.

Usage:
    python scripts/camera_diagnostic.py                       # configs/default.yaml
    python scripts/camera_diagnostic.py --device 1 --no-display --frames 200
    python scripts/camera_diagnostic.py --save-dir /tmp/cam --no-display
    python scripts/camera_diagnostic.py --width 640 --height 480 --format YUYV

Example output:
    Camera backend: V4L2
    Requested: 1280x720 @ 30 FPS MJPG
    Actual:    1280x720 @ 30 FPS MJPG
    Capture FPS: 29.8
    Frame drops: 0
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from bas_assistant.config.settings import load_settings
from bas_assistant.utils.logging import setup_logging
from bas_assistant.video.source import OpenCVVideoSource, VideoSourceError

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=None, help="Path to a configs/*.yaml file")
    parser.add_argument("--device", default=None, help="Override camera device/index")
    parser.add_argument("--width", type=int, default=None, help="Override requested width")
    parser.add_argument("--height", type=int, default=None, help="Override requested height")
    parser.add_argument("--fps", type=int, default=None, help="Override requested FPS")
    parser.add_argument(
        "--format", default=None, help="Override pixel format / FourCC (MJPG, YUYV, ...)"
    )
    parser.add_argument(
        "--disable-dynamic-framerate",
        action="store_true",
        help="Linux/V4L2: set exposure_dynamic_framerate=0 via v4l2-ctl (fixes ~17fps drops)",
    )
    parser.add_argument(
        "--frames", type=int, default=300, help="Number of frames to capture (0 = until closed)"
    )
    parser.add_argument("--no-display", action="store_true", help="Run headless (no GUI window)")
    parser.add_argument("--save-dir", type=Path, default=None, help="Save sample frames to a dir")
    parser.add_argument("--save-every", type=int, default=25, help="Save every Nth frame")
    return parser.parse_args()


def _device_arg(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def main() -> int:
    args = _parse_args()
    setup_logging(logging.INFO)

    settings = load_settings(args.config)
    cam = settings.camera
    if args.device is not None:
        cam.device = _device_arg(args.device)
    if args.width is not None:
        cam.width = args.width
    if args.height is not None:
        cam.height = args.height
    if args.fps is not None:
        cam.fps = args.fps
    if args.format is not None:
        cam.format = args.format
    if args.disable_dynamic_framerate:
        cam.disable_dynamic_framerate = True

    source = OpenCVVideoSource(
        cam.device,
        width=cam.width,
        height=cam.height,
        fps=cam.fps,
        format=cam.format,
        backend=cam.backend,
        disable_dynamic_framerate=cam.disable_dynamic_framerate,
    )
    try:
        source.start()
    except VideoSourceError as exc:
        logger.error("%s", exc)
        return 1

    diagnostics = source.diagnostics()
    print(f"Camera backend: {diagnostics['backend']}")
    print("Requested: {}x{} @ {} FPS {}".format(cam.width, cam.height, cam.fps, cam.format or "?"))
    actual = diagnostics["actual"] or {}
    print(
        "Actual:    {width}x{height} @ {fps:.0f} FPS {format}".format(
            width=actual.get("width", "?"),
            height=actual.get("height", "?"),
            fps=actual.get("fps", 0.0),
            format=actual.get("format") or "?",
        )
    )

    save_dir = args.save_dir
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving sample frames to {save_dir}")

    window = None
    if not args.no_display:
        window = "Camera diagnostic (press q to quit)"
        cv2.namedWindow(window)

    start = time.perf_counter()
    frames_ok = 0
    frames_fail = 0
    consecutive_failures = 0
    try:
        while True:
            if args.frames and frames_ok + frames_fail >= args.frames:
                break
            frame = source.read()
            if frame is None:
                frames_fail += 1
                consecutive_failures += 1
                if consecutive_failures >= 30:  # dead source guard
                    break
                continue
            consecutive_failures = 0
            frames_ok += 1
            if window is not None:
                fps_live = frames_ok / max(time.perf_counter() - start, 1e-9)
                overlay = frame.copy()
                cv2.putText(
                    overlay,
                    f"fps {fps_live:.1f}  frame {frames_ok}  {actual.get('format','')}",
                    (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow(window, overlay)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            if save_dir is not None and frames_ok % args.save_every == 0:
                out = save_dir / f"frame_{frames_ok:04d}.jpg"
                cv2.imwrite(str(out), frame)
    finally:
        source.stop()
        if window is not None:
            cv2.destroyAllWindows()

    elapsed = max(time.perf_counter() - start, 1e-9)
    diagnostics = source.diagnostics()
    capture_fps = diagnostics.get("acquisition_fps", 0.0)
    print()
    print(f"Frames: {frames_ok} read, {frames_fail} failed (over {elapsed:.1f} s)")
    print(f"Capture FPS (measured): {capture_fps:.1f}")
    print(f"Mean capture time: {diagnostics.get('capture_ms_mean', 0.0):.1f} ms")
    print(f"Mean frame gap: {diagnostics.get('frame_gap_ms_mean', 0.0):.1f} ms")
    print(f"Frame size: {actual.get('width', '?')}x{actual.get('height', '?')}")
    if frames_fail:
        print(f"Frame drops: {frames_fail}")
    if capture_fps and capture_fps < cam.fps * 0.8:
        print()
        print(
            "NOTE: measured capture FPS is below the requested FPS. The negotiated mode "
            "still reports full FPS, so the driver is silently throttling delivery."
        )
        if sys.platform != "win32":
            print(
                "  On Linux, this is usually the V4L2 exposure_dynamic_framerate control: "
                "re-run with --disable-dynamic-framerate or set "
                "camera.disable_dynamic_framerate: true in configs/default.yaml."
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
