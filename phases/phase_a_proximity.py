import config
from utils.geometry import euclidean_distance, time_to_collision, ttc_score


def _is_person(label):
    return label == config.PERSON_CLASS


def _pair_threshold(track_a, track_b, calibrated=False):
    if calibrated:
        if _is_person(track_a.label) or _is_person(track_b.label):
            return config.PROXIMITY_PERSON_THRESHOLD_M
        return config.PROXIMITY_THRESHOLD_M
    if _is_person(track_a.label) or _is_person(track_b.label):
        return config.PROXIMITY_PERSON_THRESHOLD
    return config.PROXIMITY_THRESHOLD


def proximity_filter(tracks, threshold=None, calibration=None, effective_fps=None):
    """
    Phase A gate: proximity + Time-To-Collision.
    Returns list of (track1, track2, distance, ttc_score).

    If `calibration` is a valid CameraCalibration (utils/calibration.py),
    distance/TTC are computed in real-world meters/seconds against the
    metric thresholds (config.PROXIMITY_THRESHOLD_M / TTC_MAX_SECONDS).
    Otherwise this falls back to the original pixel-based thresholds —
    behavior is completely unchanged unless a camera has been explicitly
    calibrated (config.CALIBRATION_ENABLED + a real calibration file).
    """
    use_calibration = calibration is not None and getattr(calibration, "is_calibrated", False)
    candidate_pairs = []
    num_tracks = len(tracks)

    for i in range(num_tracks):
        for j in range(i + 1, num_tracks):
            t1 = tracks[i]
            t2 = tracks[j]

            if use_calibration:
                pair_threshold = threshold or _pair_threshold(t1, t2, calibrated=True)
                p1 = t1.get_centroid()
                p2 = t2.get_centroid()
                dist = calibration.distance_m(p1, p2)

                if dist > pair_threshold:
                    continue

                ttc = calibration.time_to_collision_seconds(t1, t2, effective_fps or 15.0)
                has_velocity = len(t1.velocities) > 1 and len(t2.velocities) > 1

                if has_velocity:
                    if ttc >= config.TTC_MAX_SECONDS:
                        continue
                    score = max(0.0, 1.0 - (ttc / config.TTC_MAX_SECONDS))
                else:
                    # Static image / first frame — distance-only gate
                    score = max(0.0, 1.0 - (dist / pair_threshold))

                candidate_pairs.append((t1, t2, dist, score))
                continue

            # ── Uncalibrated fallback: original pixel-based logic ──────
            pair_threshold = threshold or _pair_threshold(t1, t2)
            p1 = t1.get_centroid()
            p2 = t2.get_centroid()
            dist = euclidean_distance(p1, p2)

            if dist > pair_threshold:
                continue

            ttc = time_to_collision(t1, t2)
            has_velocity = len(t1.velocities) > 1 and len(t2.velocities) > 1

            if has_velocity:
                if ttc >= config.TTC_MAX_FRAMES:
                    continue
                score = ttc_score(ttc)
            else:
                # Static image / first frame — distance-only gate
                score = max(0.0, 1.0 - (dist / pair_threshold))

            candidate_pairs.append((t1, t2, dist, score))

    return candidate_pairs
