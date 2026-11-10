"""Optional source-time center smoothing; no fitting or new visibility claims."""

import csv
import json
import math
import os
from pathlib import Path

import numpy as np

from .track_pose import flag, WARP_FIELDS

FIELDS = ("raw_x", "raw_y", "center_smoothed")


def validate_window(seconds):
    if (
        isinstance(seconds, bool)
        or not isinstance(seconds, (int, float))
        or not math.isfinite(seconds)
        or not 0 <= seconds <= 10
    ):
        raise ValueError(
            "Smoothing window must be source seconds in [0,10]; zero disables it"
        )
    return float(seconds)


def smooth_rows(rows, metadata, seconds):
    """Degree-2 SG within contiguous visible runs; never fill or extend a track.

    Raw position columns retain the measurements. Translating the whole H keeps
    its query mapping and deformation coherent; its fit score still describes
    the raw measurement, not a newly fitted smoothed pose.
    """
    seconds = validate_window(seconds)
    if not seconds:
        return 0
    from scipy.signal import savgol_filter

    fps = float(metadata["source_metadata_fps"])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Smoothing requires positive source FPS")
    window = max(5, int(round(seconds * fps)) | 1)
    size = np.asarray(metadata["source_size_px"], float)
    anchors = metadata.get("manual_edits", {}).get("anchors", {})
    grouped = {}
    for row in rows:
        grouped.setdefault(int(row["point_id"]), []).append(row)
    changed = 0
    for identity, values in grouped.items():
        values.sort(key=lambda row: int(row["frame"]))
        frames = np.asarray([int(row["frame"]) for row in values])
        observed = np.asarray([[float(row["x"]), float(row["y"])] for row in values])
        raw = np.asarray(
            [
                [
                    float(
                        row["raw_x"] if row.get("raw_x") not in ("", None) else row["x"]
                    ),
                    float(
                        row["raw_y"] if row.get("raw_y") not in ("", None) else row["y"]
                    ),
                ]
                for row in values
            ]
        )
        valid = np.asarray([flag(row["visible"]) for row in values]) & np.isfinite(
            raw
        ).all(axis=1)
        valid &= ((raw >= 0) & (raw < size)).all(axis=1)
        cuts = np.flatnonzero(
            np.r_[True, (np.diff(frames) != 1) | ~valid[1:] | ~valid[:-1], True]
        )
        positions = raw.copy()
        for start, stop in zip(cuts[:-1], cuts[1:]):
            count = stop - start
            if count < 5 or not valid[start:stop].all():
                continue
            width = min(window, count if count % 2 else count - 1)
            positions[start:stop] = savgol_filter(
                raw[start:stop], width, 2, axis=0, mode="interp"
            )
        pinned = {int(metadata["query_frame"])}
        pinned.update(int(row["frame"]) for row in anchors.get(str(identity), []))
        for index, row in enumerate(values):
            row.update(
                raw_x=float(raw[index, 0]),
                raw_y=float(raw[index, 1]),
                center_smoothed=0,
            )
            if not valid[index] or frames[index] in pinned:
                continue
            xy = np.clip(positions[index], 0, size - 1)
            delta = xy - observed[index]
            filtered = bool(np.linalg.norm(xy - raw[index]) >= 1e-9)
            row["center_smoothed"] = int(filtered)
            changed += int(filtered)
            if np.linalg.norm(delta) < 1e-9:
                continue
            for fields in (
                WARP_FIELDS,
                tuple(f"h{y}{x}" for y in range(3) for x in range(3)),
            ):
                if not all(row.get(key) not in ("", None) for key in fields):
                    continue
                matrix = np.asarray([float(row[key]) for key in fields]).reshape(3, 3)
                matrix[:2] += delta[:, None] * matrix[2]
                row.update(zip(fields, map(float, matrix.ravel())))
            row.update(x=float(xy[0]), y=float(xy[1]))
            for xkey, ykey in (
                ("circle_x", "circle_y"),
                ("bbox_x", "bbox_y"),
                ("point_prediction_x", "point_prediction_y"),
            ):
                if row.get(xkey) not in ("", None) and row.get(ykey) not in ("", None):
                    row[xkey], row[ykey] = (
                        float(row[xkey]) + delta[0],
                        float(row[ykey]) + delta[1],
                    )
    metadata["trajectory_smoothing"] = dict(
        method="savitzky_golay",
        polynomial_order=2,
        window_seconds=seconds,
        nominal_window_frames=window,
        changed_centers=changed,
        raw_columns=list(FIELDS[:2]),
        fit_scores_refer_to="raw measurements",
        policy="visible contiguous source-frame runs only; no gap filling; query/manual anchors fixed; full H translated",
    )
    return changed


def smooth_csv(path, metadata, seconds):
    """Used on an unpublished exporter temporary, after lifetime cleanup."""
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields, rows = list(reader.fieldnames), list(reader)
    changed = smooth_rows(rows, metadata, seconds)
    fields += [key for key in FIELDS if key not in fields]
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    metadata["tracks_bytes"] = Path(path).stat().st_size
    return changed


def smooth_tracks(tracks, output, *, seconds=0.1):
    """Save a derivative of existing/corrected tracks, retaining source masks."""
    from .tapnextpp import contained_path, write_json
    from .track_pose import read_point_pose_tracks, write_point_pose_tracks
    from ..common.validation import require_no_output_references

    tracks, output = contained_path(tracks, "data"), contained_path(output, "data")
    if tracks == output or output.exists() or output.with_suffix(".json").exists():
        raise ValueError(
            "Smoothing output must be a new CSV; original tracks are preserved"
        )
    seconds = validate_window(seconds)
    if not seconds:
        raise ValueError("Specify a positive smoothing window")
    data = read_point_pose_tracks(tracks)
    rows = [
        row for frame in data["frames"] for row in data["rows_by_frame"][frame].values()
    ]
    smooth_rows(rows, data["metadata"], seconds)
    write_point_pose_tracks(output, data, extra_fields=FIELDS)
    metadata = json.loads(output.with_suffix(".json").read_text())
    metadata.update(
        tracker="smoothed-points",
        trajectory_smoothing=data["metadata"]["trajectory_smoothing"],
    )
    manual = data["metadata"].get("manual_edits")
    if manual:
        manual = dict(manual)
        if manual.get("base_tracks"):
            manual["base_tracks"] = os.path.relpath(
                tracks.parent / manual["base_tracks"], output.parent
            )
        metadata["manual_edits"] = manual
    require_no_output_references(metadata)
    write_json(output.with_suffix(".json"), metadata)
    return dict(output=str(output), **metadata["trajectory_smoothing"])
