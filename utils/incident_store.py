"""Local incident persistence and index for dashboard API."""

import json
import os
import threading
import uuid
from datetime import datetime, timezone

import config

_lock = threading.Lock()

# ── Operator workflow (Batch 3) ──────────────────────────────
# Every incident moves through: new -> acknowledged -> resolved.
# "status" (confirmed/suspicious/suppressed) is the PIPELINE's classification
# of the detection itself and never changes after the fact. "workflow_status"
# is separate: it's what a human operator has done about it. Keeping these
# two concepts distinct means an operator can resolve a "confirmed" incident
# as a false positive without the system pretending the pipeline classified
# it differently — the resolution_reason is the operator's correction, not
# a rewrite of the detector's output.
RESOLUTION_REASONS = {
    "confirmed_accident",  # operator verified a real accident occurred
    "false_positive",      # pipeline triggered but there was no accident
    "duplicate",            # same event already recorded as another incident
    "test_footage",         # known test/demo clip, not a real camera event
    "other",                # anything else — pair with resolution_note
}


def _ensure_workflow_defaults(record):
    """
    Backfill workflow fields on a record so both newly-created incidents and
    ones saved before this feature existed have a consistent shape when
    returned from the API. Mutates and returns the same dict.
    """
    record.setdefault("workflow_status", "new")
    record.setdefault("acknowledged_at", None)
    record.setdefault("resolved_at", None)
    record.setdefault("resolution_reason", None)
    record.setdefault("resolution_note", None)
    return record


def _base_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def incidents_dir():
    path = os.path.join(_base_dir(), config.INCIDENTS_DIR)
    os.makedirs(path, exist_ok=True)
    return path


def index_path():
    path = os.path.join(_base_dir(), config.INCIDENTS_INDEX)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def _load_index():
    path = index_path()
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _save_index(records):
    with open(index_path(), "w") as f:
        json.dump(records, f, indent=2)


def _public_url(relative_path):
    rel = relative_path.replace("\\", "/")
    if not rel.startswith("/"):
        rel = "/" + rel
    return rel


def save_incident(record):
    """Persist an incident record and return it with id/urls set."""
    with _lock:
        incident_id = record.get("id") or str(uuid.uuid4())
        record["id"] = incident_id

        if "timestamp" not in record:
            record["timestamp"] = datetime.now(timezone.utc).isoformat()

        _ensure_workflow_defaults(record)

        records = _load_index()
        records = [r for r in records if r.get("id") != incident_id]
        records.insert(0, record)
        _save_index(records)
        return record


def update_incident(incident_id, updates: dict) -> bool:
    """
    Partially update an existing incident record in the index.
    Used to set clip_url and llm_analysis after async background processing.
    Returns True if the record was found and updated.
    """
    with _lock:
        records = _load_index()
        found = False
        for record in records:
            if record.get("id") == incident_id:
                record.update(updates)
                found = True
                break
        if found:
            _save_index(records)
    return found


def clear_all_incidents() -> bool:
    """
    Delete every incident from the index and remove their media files from disk.
    Used by the dashboard 'Clear All' button.
    """
    with _lock:
        records = _load_index()
        for record in records:
            iid = record.get("id")
            if not iid:
                continue
            clip_fs, snap_fs, _, _ = build_incident_paths(iid)
            for path in (clip_fs, snap_fs):
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except OSError:
                    pass
        _save_index([])
    return True


def list_incidents(limit=50):
    with _lock:
        records = _load_index()
    return [_ensure_workflow_defaults(r) for r in records[:limit]]


def get_incident(incident_id):
    with _lock:
        for record in _load_index():
            if record.get("id") == incident_id:
                return _ensure_workflow_defaults(record)
    return None


def delete_incident(incident_id):
    """Remove an incident from the index and delete its media files."""
    with _lock:
        records = _load_index()
        target = next((r for r in records if r.get("id") == incident_id), None)
        if target is None:
            return False

        records = [r for r in records if r.get("id") != incident_id]
        _save_index(records)

    clip_fs, snap_fs, _, _ = build_incident_paths(incident_id)
    for path in (clip_fs, snap_fs):
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass
    return True


def build_incident_paths(incident_id):
    """Return filesystem paths and public URLs for clip/snapshot assets."""
    clip_name = f"clip_{incident_id}.mp4"
    snap_name = f"snap_{incident_id}.jpg"
    clip_fs = os.path.join(incidents_dir(), clip_name)
    snap_fs = os.path.join(incidents_dir(), snap_name)
    clip_url = _public_url(f"{config.INCIDENTS_DIR}/{clip_name}")
    snap_url = _public_url(f"{config.INCIDENTS_DIR}/{snap_name}")
    return clip_fs, snap_fs, clip_url, snap_url


def acknowledge_incident(incident_id):
    """
    Mark an incident as seen/being-handled by an operator. Only moves
    "new" -> "acknowledged" — calling this on an already-acknowledged or
    already-resolved incident is a harmless no-op (idempotent), so a
    double-click or a second operator opening the same card can't corrupt
    the timestamp.
    Returns the updated record, or None if the incident doesn't exist.
    """
    with _lock:
        records = _load_index()
        target = next((r for r in records if r.get("id") == incident_id), None)
        if target is None:
            return None
        _ensure_workflow_defaults(target)
        if target["workflow_status"] == "new":
            target["workflow_status"] = "acknowledged"
            target["acknowledged_at"] = datetime.now(timezone.utc).isoformat()
        _save_index(records)
        return target


def resolve_incident(incident_id, reason, note=None):
    """
    Close out an incident with a reason code (see RESOLUTION_REASONS).
    These reason codes are exactly the labeled feedback the deployment
    report calls for: a growing set of operator-confirmed true/false
    positives that can eventually inform threshold and model tuning.
    Returns the updated record, or None if the incident doesn't exist.
    Raises ValueError if `reason` isn't a recognized code.
    """
    if reason not in RESOLUTION_REASONS:
        raise ValueError(f"Unknown resolution reason: {reason!r}")

    with _lock:
        records = _load_index()
        target = next((r for r in records if r.get("id") == incident_id), None)
        if target is None:
            return None
        _ensure_workflow_defaults(target)

        now = datetime.now(timezone.utc).isoformat()
        target["workflow_status"] = "resolved"
        target["resolved_at"] = now
        target["resolution_reason"] = reason
        target["resolution_note"] = note
        # Resolving directly from "new" (operator skipped the ack step)
        # still implies it was seen and handled — backfill acknowledged_at
        # so the audit trail never shows a resolution with no acknowledgment.
        if target.get("acknowledged_at") is None:
            target["acknowledged_at"] = now

        _save_index(records)
        return target


def reopen_incident(incident_id):
    """
    Revert an incident back to "new" — for correcting an accidental
    acknowledge/resolve. Clears all workflow timestamps and the
    resolution reason/note.
    Returns the updated record, or None if the incident doesn't exist.
    """
    with _lock:
        records = _load_index()
        target = next((r for r in records if r.get("id") == incident_id), None)
        if target is None:
            return None
        target["workflow_status"] = "new"
        target["acknowledged_at"] = None
        target["resolved_at"] = None
        target["resolution_reason"] = None
        target["resolution_note"] = None
        _save_index(records)
        return target
