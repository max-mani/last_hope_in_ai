"""
UYIR Data Logger — logs raw factor values for threshold tuning.

Usage:
  python data_logger.py --video clip.mp4 --label accident
  python data_logger.py --video normal.mp4 --label normal
"""

import argparse
import csv                   # FIX: was "import csvzzzzz" (typo)
import os
from itertools import combinations

import cv2
import numpy as np

import config
from tracking.vehicle_tracker import VehicleTracker

_prev_gray_log = None
_flow_hist_log = []


def _get_flow_ratio(gray: np.ndarray) -> float:
    global _prev_gray_log, _flow_hist_log
    magnitude = 0.0
    if _prev_gray_log is not None:
        try:
            flow = cv2.calcOpticalFlowFarneback(
                _prev_gray_log, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            magnitude = float(np.mean(mag))
        except Exception:
            pass
    _prev_gray_log = gray.copy()
    _flow_hist_log.append(magnitude)
    if len(_flow_hist_log) > 10:
        _flow_hist_log.pop(0)
    avg = max(0.1, float(np.mean(_flow_hist_log)))
    return magnitude / avg


def _iou(b1, b2):
    xi1 = max(b1[0], b2[0]); yi1 = max(b1[1], b2[1])
    xi2 = min(b1[2], b2[2]); yi2 = min(b1[3], b2[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    if inter == 0:
        return 0.0
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def run_logger(video_path, label: str, output_csv: str):
    global _prev_gray_log, _flow_hist_log
    _prev_gray_log = None
    _flow_hist_log = []

    tracker = VehicleTracker()
    accident_model = None
    if os.path.exists(config.STAGE1_YOLO_GATE_PATH):
        from ultralytics import YOLO
        accident_model = YOLO(config.STAGE1_YOLO_GATE_PATH)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Logger] Cannot open: {video_path}")
        return

    file_exists = os.path.exists(output_csv)
    csv_file = open(output_csv, "a", newline="")
    writer = csv.writer(csv_file)

    if not file_exists:
        writer.writerow([
            "video_file", "label", "frame_num", "vehicle_id", "class_name",
            "centroid_x", "centroid_y", "bbox_width", "bbox_height",
            "speed_px_per_frame", "avg_speed_5f", "speed_drop_percent",
            "direction_degrees", "nearest_euclidean_dist", "iou_with_nearest",
            "trajectory_deviation_px", "bbox_area_change_ratio",
            "optical_flow_ratio", "accident_model_confidence",
        ])

    video_name = os.path.basename(str(video_path))
    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        vehicles = tracker.process_frame(frame)
        vlist = list(vehicles.values())
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow_ratio = _get_flow_ratio(gray)

        acc_conf = 0.0
        if accident_model is not None:
            acc_results = accident_model.predict(frame, conf=0.1, verbose=False)
            if acc_results and acc_results[0].boxes is not None:
                confs = acc_results[0].boxes.conf.cpu().numpy()
                if len(confs) > 0:
                    acc_conf = float(confs.max())

        euclid_map = {}
        iou_map = {}
        if len(vlist) >= 2:
            for va, vb in combinations(vlist, 2):
                cx1, cy1 = va.centroid; cx2, cy2 = vb.centroid
                d = float(np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2))
                euclid_map[va.id] = min(euclid_map.get(va.id, 9999), d)
                euclid_map[vb.id] = min(euclid_map.get(vb.id, 9999), d)
                val = _iou(va.bbox, vb.bbox)
                iou_map[va.id] = max(iou_map.get(va.id, 0.0), val)
                iou_map[vb.id] = max(iou_map.get(vb.id, 0.0), val)

        for vid, v in vehicles.items():
            w = int(v.bbox[2] - v.bbox[0])
            h = int(v.bbox[3] - v.bbox[1])
            writer.writerow([
                video_name, label, frame_num, vid, v.class_name,
                v.centroid[0], v.centroid[1], w, h,
                round(v.speed, 3), round(v.get_avg_speed(5), 3),
                round(v.get_speed_drop_percent(), 3), round(v.direction, 2),
                round(euclid_map.get(vid, 9999), 2), round(iou_map.get(vid, 0.0), 4),
                round(v.get_trajectory_deviation(), 3), round(v.get_bbox_area_change(), 4),
                round(flow_ratio, 4), round(acc_conf, 4),
            ])

        display = tracker.draw_tracks(frame.copy(), vehicles)
        cv2.putText(display, f"Frame:{frame_num} Vehicles:{len(vehicles)} [{label.upper()}]",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)
        cv2.imshow("UYIR Data Logger", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    csv_file.close()
    cv2.destroyAllWindows()
    print(f"\n[Logger] Done — {frame_num} frames logged to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UYIR Data Logger")
    parser.add_argument("--video", required=True, help="Video file, RTSP URL, or 0 for webcam")
    parser.add_argument("--label", required=True, choices=["accident", "normal"])
    parser.add_argument("--output", default=config.DATA_LOG_CSV)
    args = parser.parse_args()
    src = int(args.video) if args.video.isdigit() else args.video
    run_logger(src, args.label, args.output)
