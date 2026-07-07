"""
UYIR LLM Verifier — sends a confirmed-accident snapshot to an external
Hugging Face Space (kishore-28k50/accident-call) for independent
verification, using the gradio_client API.

This is a SECOND OPINION layered on top of the detector's own decision,
not a replacement for it. The detector has already confirmed an accident
(DL gate + phase votes + consecutive frames + cooldown) before this ever
runs — this module's job is only to produce a labeled training row for
the XGBoost refinement model, and to attach a note to the incident
record. If the external space is unreachable, slow, or returns something
unexpected, that's treated as "no verdict available" and the pipeline's
own detection stands on its own; nothing here should ever cause the
detection pipeline itself to fail or block.

Requires the `gradio_client` package:
    pip install gradio_client
"""

import json
import logging

logger = logging.getLogger("LLMVerifier")

SPACE_NAME = "kishore-28k50/accident-call"


def verify_accident_frame(image_path, num_frames=2, timeout=None):
    """
    Calls the external Hugging Face Space with a single confirmed-accident
    snapshot. Returns a dict:
        {
            "accident_detected": bool | None,   # None if no usable verdict
            "confidence_percent": float | None,
            "report": str | None,
            "raw": dict | None,                 # full parsed response
            "error": str | None,
        }
    Never raises.
    """
    try:
        from gradio_client import Client, handle_file
    except ImportError:
        return {
            "accident_detected": None, "confidence_percent": None,
            "report": None, "raw": None,
            "error": "gradio_client not installed — run: pip install gradio_client",
        }

    try:
        client = Client(SPACE_NAME)
        result = client.predict(
            image=handle_file(image_path),
            video=None,
            num_frames=num_frames,
            api_name="/analyze_media",
        )
    except Exception as e:
        logger.warning(f"LLM verification call to {SPACE_NAME} failed: {e}")
        return {
            "accident_detected": None, "confidence_percent": None,
            "report": None, "raw": None, "error": str(e),
        }

    parsed = result
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
        except Exception:
            parsed = None

    if not isinstance(parsed, dict):
        logger.warning(f"LLM verification returned an unrecognized response shape: {type(result)}")
        return {
            "accident_detected": None, "confidence_percent": None,
            "report": None, "raw": result if isinstance(result, (dict, list, str)) else None,
            "error": "Unrecognized response format from LLM space.",
        }

    detection = parsed.get("detection", {}) if isinstance(parsed.get("detection"), dict) else {}

    return {
        "accident_detected": detection.get("accident_detected"),
        "confidence_percent": detection.get("confidence_percent"),
        "report": detection.get("incident_report"),
        # frame_inference/per_frame_analysis can carry embedded base64 image
        # data — dropped here so we don't duplicate-store large blobs inside
        # every incident record's JSON.
        "raw": {k: v for k, v in parsed.items() if k not in ("frame_inference",)},
        "error": None,
    }
