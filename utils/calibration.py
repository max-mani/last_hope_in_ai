"""
UYIR Camera Calibration — pixel <-> real-world metric conversion.

WHY THIS EXISTS
----------------
Phase A's proximity/TTC gate and several Phase B thresholds are computed
directly in image pixels. Two cars 150px apart on a camera mounted low
and close to the road can be meters apart in reality compared to the
same two cars 150px apart on a camera mounted high and far back. A
single set of pixel constants (PROXIMITY_THRESHOLD, TTC_MAX_FRAMES) is
therefore not physically meaningful across more than one specific
camera setup — this is the #1 item in the deployment readiness report.

WHAT THIS MODULE DOES
----------------------
Nothing here fabricates real-world measurements. It provides:

1. `CameraCalibration` — loads a per-camera planar homography (a flat
   ground-plane assumption, standard for this kind of road-surface
   traffic-conflict analysis) from a JSON file, and converts pixel
   points / distances / speeds to meters and m/s.
2. `get_calibration(camera_id)` — used by the detection phases. If no
   calibration file exists for a camera, it returns None and every
   caller falls back to the existing pixel-based behavior. Nothing
   breaks and nothing is silently wrong — it's just uncalibrated,
   exactly like today.
3. A small interactive CLI (`python -m utils.calibration ...`) for a
   field team to actually calibrate a camera: click 4+ points on a
   saved frame that correspond to points whose real-world positions
   you have physically measured, enter those coordinates, and it
   saves the resulting homography.

IMPORTANT — READ THIS BEFORE ENABLING CALIBRATION FOR ANY CAMERA
------------------------------------------------------------------
No physical site survey of any camera location has been performed as
part of this change. I (the assistant) have no way to measure a real
camera's mounting height, angle, or the physical dimensions of the
road it's pointed at from here. A calibration file MUST be produced by
someone physically measuring reference points at the actual camera
site (tape measure, a surveyed site plan, or documented lane-width
standards for the road markings visible in frame — NOT distances eyeballed
from satellite imagery, which isn't precise enough at this scale).

A fabricated/guessed calibration would produce confidently WRONG
metric thresholds — worse than the current, honestly-approximate pixel
thresholds, especially for a system whose output may inform a police
response. CALIBRATION_ENABLED defaults to False in config.py for
exactly this reason, and stays False for a given camera until a real
`calibration/<camera_id>.json` file exists.
"""

import json
import os

import cv2
import numpy as np

import config


def _calibration_path(camera_id):
    os.makedirs(config.CALIBRATION_DIR, exist_ok=True)
    return os.path.join(config.CALIBRATION_DIR, f"{camera_id}.json")


class CameraCalibration:
    """
    Homography-based pixel <-> real-world-meter converter for one camera,
    assuming a flat ground plane (the standard assumption for road-surface
    traffic-conflict analysis from a fixed CCTV angle).
    """

    def __init__(self, camera_id, pixel_points, world_points_m):
        """
        pixel_points:   list of >=4 [x, y] pixel coordinates from the camera frame
        world_points_m: list of >=4 [x, y] real-world coordinates in meters,
                         in the same order, measured on-site on the ground plane
        """
        if len(pixel_points) < 4 or len(pixel_points) != len(world_points_m):
            raise ValueError("Need >= 4 matching pixel/world point pairs.")

        self.camera_id = camera_id
        self.pixel_points = np.array(pixel_points, dtype=np.float64)
        self.world_points_m = np.array(world_points_m, dtype=np.float64)

        self.H, _ = cv2.findHomography(self.pixel_points, self.world_points_m, method=0)
        if self.H is None:
            raise ValueError("Homography computation failed — check point correspondences.")

    @property
    def is_calibrated(self):
        return self.H is not None

    def pixel_to_meters(self, point_px):
        """Transform a single (x, y) pixel point to (x, y) meters on the ground plane."""
        pt = np.array([[point_px]], dtype=np.float64)  # shape (1, 1, 2)
        out = cv2.perspectiveTransform(pt, self.H)
        return float(out[0, 0, 0]), float(out[0, 0, 1])

    def distance_m(self, point_a_px, point_b_px):
        """Real-world ground-plane distance in meters between two pixel points."""
        ax, ay = self.pixel_to_meters(point_a_px)
        bx, by = self.pixel_to_meters(point_b_px)
        return float(np.hypot(ax - bx, ay - by))

    def speed_mps(self, prev_point_px, curr_point_px, dt_seconds):
        """Real-world speed in m/s between two pixel positions dt_seconds apart."""
        if dt_seconds <= 0:
            return 0.0
        return self.distance_m(prev_point_px, curr_point_px) / dt_seconds

    def time_to_collision_seconds(self, track_a, track_b, effective_fps):
        """
        Metric TTC in seconds, using ground-plane positions and closing
        speed instead of raw pixel velocity. Mirrors
        utils.geometry.time_to_collision but calibrated.
        Returns float('inf') if not converging or fps is invalid.
        """
        if effective_fps is None or effective_fps <= 0:
            return float("inf")
        dt = 1.0 / effective_fps

        c1 = track_a.get_centroid()
        c2 = track_b.get_centroid()
        ax, ay = self.pixel_to_meters(c1)
        bx, by = self.pixel_to_meters(c2)
        distance = float(np.hypot(ax - bx, ay - by))

        if len(track_a.history) < 2 or len(track_b.history) < 2:
            return float("inf")

        a_prev_x, a_prev_y = self.pixel_to_meters(track_a.history[-2])
        b_prev_x, b_prev_y = self.pixel_to_meters(track_b.history[-2])

        va = ((ax - a_prev_x) / dt, (ay - a_prev_y) / dt)
        vb = ((bx - b_prev_x) / dt, (by - b_prev_y) / dt)
        rel_vx = va[0] - vb[0]
        rel_vy = va[1] - vb[1]
        closing_speed = float(np.hypot(rel_vx, rel_vy))

        if closing_speed < config.TTC_MIN_CLOSING_SPEED_MPS:
            return float("inf")

        return distance / closing_speed

    def to_dict(self):
        return {
            "camera_id": self.camera_id,
            "pixel_points": self.pixel_points.tolist(),
            "world_points_m": self.world_points_m.tolist(),
        }

    def save(self):
        path = _calibration_path(self.camera_id)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def load(cls, camera_id):
        path = _calibration_path(camera_id)
        if not os.path.exists(path):
            return None
        with open(path, "r") as f:
            data = json.load(f)
        try:
            return cls(data["camera_id"], data["pixel_points"], data["world_points_m"])
        except (KeyError, ValueError) as e:
            print(f"[Calibration] Invalid calibration file for {camera_id}: {e}")
            return None


_cache = {}


def get_calibration(camera_id):
    """
    Returns a CameraCalibration for camera_id, or None if uncalibrated.
    Cached per process — calibration files don't change while running.
    """
    if camera_id in _cache:
        return _cache[camera_id]
    calib = CameraCalibration.load(camera_id)
    _cache[camera_id] = calib
    return calib


# ================= INTERACTIVE CALIBRATION CLI =================
def _run_cli():
    """
    Field calibration tool. Run:
        python -m utils.calibration --camera CAM_001 --frame snapshot.jpg

    Click 4+ points on the displayed frame that correspond to points you
    have physically measured on the ground (e.g. lane-marking corners,
    a surveyed reference rectangle, a marked pedestrian crossing). Press
    'c' when done clicking, 'q' to abort. You'll then be prompted in the
    terminal for each point's real-world (x, y) in meters, using any
    consistent local origin (e.g. one corner of the junction as (0, 0)).

    Use points you can actually measure with a tape or that are already
    documented (a site survey / architectural drawing / known IRC lane-
    width standard) — do not estimate distances from satellite imagery
    alone; it is not precise enough for TTC-scale thresholds.
    """
    import argparse

    parser = argparse.ArgumentParser(description="UYIR Camera Calibration Tool")
    parser.add_argument("--camera", required=True, help="Camera ID, e.g. CAM_001")
    parser.add_argument("--frame", required=True, help="Path to a saved frame/snapshot from that camera")
    args = parser.parse_args()

    img = cv2.imread(args.frame)
    if img is None:
        print(f"[Calibration] Could not read frame: {args.frame}")
        return

    window = "UYIR Calibration - click reference points, then press 'c'"
    clicked = []

    def _on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked.append((x, y))
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(img, str(len(clicked)), (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow(window, img)

    cv2.imshow(window, img)
    cv2.setMouseCallback(window, _on_click)
    print("Click 4+ points with known real-world coordinates. Press 'c' to continue, 'q' to abort.")

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord("c") and len(clicked) >= 4:
            break
        if key == ord("q"):
            cv2.destroyAllWindows()
            print("Aborted.")
            return

    cv2.destroyAllWindows()

    world_points = []
    print("\nEnter the real-world (x, y) in METERS for each clicked point,")
    print("using a consistent local origin (e.g. one known corner = 0,0):\n")
    for i, (px, py) in enumerate(clicked, start=1):
        raw = input(f"  Point {i} (pixel {px},{py}) - world x,y in meters (e.g. '0,0'): ")
        try:
            wx, wy = [float(v.strip()) for v in raw.split(",")]
        except ValueError:
            print("  Invalid input, aborting.")
            return
        world_points.append([wx, wy])

    try:
        calib = CameraCalibration(args.camera, clicked, world_points)
    except ValueError as e:
        print(f"[Calibration] Failed: {e}")
        return

    path = calib.save()
    print(f"\nSaved calibration for {args.camera} -> {path}")
    print("Set UYIR_CALIBRATION_ENABLED=true (and make sure this camera_id matches "
          "config.CAMERA_ID / UYIR_CAMERA_ID) to activate it.")


if __name__ == "__main__":
    _run_cli()
