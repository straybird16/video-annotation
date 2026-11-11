"""Stream saved point trajectories into portable H.264 MP4 sanity previews."""
import csv
import hashlib
import itertools
from collections import deque
import json
from pathlib import Path
import warnings
from uuid import uuid4

import cv2
import numpy as np

from video_annotation.common.video import H264Writer
from video_annotation.common.video import probe_video
from .tapnextpp import contained_path, video_frames, write_json
from .workflow import archive_files, check_cancel
from .video_geometry import DISPLAY_COORDINATE_SPACE, VIDEO_GEOMETRY_VERSION, display_rotation


def _stored_point_groups(tracks):
    with Path(tracks).open(newline="") as f:
        reader = csv.DictReader(f)
        for frame, group in itertools.groupby(reader, lambda r: int(r["frame"])):
            points = {}
            for row in group:
                identity = int(row["point_id"])
                column = "quality_score" if "quality_score" in row else "visibility_logit"
                values = np.array([float(row["x"]), float(row["y"]), float(row[column])])
                if identity in points or not np.isfinite(values).all() or row["visible"] not in ("0", "1"):
                    raise ValueError(f"invalid or duplicate point row at frame {frame}")
                if column == "quality_score" and (values[2] > 0) != (row["visible"] == "1"):
                    raise ValueError(f"point quality sign disagrees with validity at frame {frame}")
                points[identity] = (values[:2], row["visible"] == "1")
            yield frame, points


def point_groups(tracks, ids, metadata=None):
    """Read dense files strictly; expand only explicitly declared terminated tails."""
    rows = iter(_stored_point_groups(tracks))
    policy = (metadata or {}).get('track_policy', {})
    if policy.get('version') != 1 or policy.get('trimmed_terminal_rows') is not True:
        for frame, points in rows:
            if set(points) != set(ids):
                raise ValueError(f'missing or unexpected point IDs at frame {frame}')
            yield frame, points
        return
    start, count = metadata['query_frame'], metadata['frames']
    ends = policy.get('point_end_frames', {})
    if (set(ends) != set(map(str, ids)) or
            any(end is not None and (isinstance(end, bool) or not isinstance(end, int) or
                                     not start <= end < start + count) for end in ends.values())):
        raise ValueError('Invalid declared point termination bounds')
    queries = np.asarray(metadata.get('query_points_xy'), float)
    if queries.shape != (len(ids), 2) or not np.isfinite(queries).all():
        raise ValueError('Terminated point tracks need finite saved query coordinates')
    held = {identity: xy.copy() for identity, xy in zip(ids, queries)}
    pending = next(rows, None)
    for frame in range(start, start + count):
        if pending is not None and pending[0] < frame:
            raise ValueError('Point rows must follow consecutive source frame order')
        points = pending[1] if pending is not None and pending[0] == frame else {}
        active = {identity for identity in ids if ends[str(identity)] is not None and frame <= ends[str(identity)]}
        if set(points) != active:
            raise ValueError(f'Point rows disagree with declared termination bounds at frame {frame}')
        if any(frame == ends[str(identity)] and not value[1] for identity, value in points.items()):
            raise ValueError('Declared point termination endpoint must be visible')
        held.update({identity: value[0] for identity, value in points.items()})
        yield frame, {identity: points.get(identity, (held[identity].copy(), False)) for identity in ids}
        if pending is not None and pending[0] == frame:
            pending = next(rows, None)
    if pending is not None:
        raise ValueError('Point rows extend beyond the declared source interval')


def draw_points(frame, points, source_frame, status_label=None, *, show_invalid=False):
    """Draw native coordinates; invalid markers are diagnostic-only opt-in."""
    result = frame.copy()
    palette = [(0, 220, 255), (255, 180, 0), (80, 255, 80), (255, 80, 255), (80, 150, 255)]
    h, w = result.shape[:2]
    for identity, (xy, visible) in sorted(points.items()):
        if not visible and not show_invalid:
            continue
        x, y = np.rint(xy).astype(int)
        if not (0 <= x < w and 0 <= y < h):
            continue
        color = palette[identity % len(palette)] if visible else (150, 150, 150)
        cv2.circle(result, (x, y), 5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.circle(result, (x, y), 4, color, -1 if visible else 1, cv2.LINE_AA)
        label = str(identity) + (" ?" if not visible else "")
        cv2.putText(result, label, (x + 7, max(14, y - 7)), cv2.FONT_HERSHEY_SIMPLEX,
                    .5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(result, label, (x + 7, max(14, y - 7)), cv2.FONT_HERSHEY_SIMPLEX,
                    .5, color, 1, cv2.LINE_AA)
    cv2.putText(result, f"Source frame {source_frame}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                .55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(result, f"Source frame {source_frame}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                .55, (255, 255, 255), 1, cv2.LINE_AA)
    if status_label:
        cv2.putText(result, status_label, (8, 45), cv2.FONT_HERSHEY_SIMPLEX,
                    .5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(result, status_label, (8, 45), cv2.FONT_HERSHEY_SIMPLEX,
                    .5, (255, 255, 255), 1, cv2.LINE_AA)
    return result


def render_tracks(tracks, output, *, fps=None, max_width=1280, force=False,
                  cancel=None, progress=None, render_options=None, reacquire_seconds=None):
    from .lifecycle import TrackLifecycle
    from .render_options import normalize_render_options
    options = normalize_render_options(render_options)
    tracks = contained_path(tracks, "data")
    output = contained_path(output, "outputs")
    if output.suffix.lower() != ".mp4":
        raise ValueError("overlay output must be a portable .mp4 file")
    metadata = json.loads(tracks.with_suffix(".json").read_text())
    video = contained_path(tracks.parent / metadata["source_video"], "data")
    source = probe_video(video)[0]
    motion_model = metadata.get("algorithm", {}).get("config", {}).get("motion_model", "similarity")
    rotation = display_rotation(source)
    if rotation and metadata.get("coordinate_space") != DISPLAY_COORDINATE_SPACE:
        raise ValueError("legacy tracks for a rotated video lack displayed-coordinate provenance; regenerate them from reviewed point references")
    if metadata.get("coordinate_space") == DISPLAY_COORDINATE_SPACE and display_rotation(metadata) != rotation:
        raise ValueError("video display rotation changed since tracking; regenerate the tracks")
    if "source_size_px" in metadata and metadata["source_size_px"] != [source["width"], source["height"]]:
        raise ValueError("source dimensions changed since tracking; use the tracked original video")
    if "source_file_stat" in metadata and metadata["source_file_stat"] != [video.stat().st_size, video.stat().st_mtime_ns]:
        raise ValueError("source file changed since tracking; restore the original file or regenerate tracks")
    identity_verified = "source_size_px" in metadata and "source_file_stat" in metadata
    if not identity_verified:
        warnings.warn("legacy track metadata lacks source size/file identity; verify that this is the original tracked video", stacklevel=2)
    fps = source["fps"] if fps is None else fps
    if not np.isfinite(fps) or fps <= 0 or max_width < 1:
        raise ValueError("overlay FPS and maximum width must be positive")
    if reacquire_seconds is None:
        reacquire_seconds = metadata.get('track_policy', {}).get('reacquire_seconds', .5)
    lifecycle = TrackLifecycle(metadata['point_ids'], source['fps'], reacquire_seconds)
    manual_anchors = metadata.get('manual_edits', {}).get('anchors', {})
    resets = {}
    for identity, anchors in manual_anchors.items():
        for anchor in anchors:
            if anchor.get('visible') is True:
                resets.setdefault(anchor['frame'], []).append(int(identity))
    signature = hashlib.sha256(json.dumps({"version": VIDEO_GEOMETRY_VERSION, "render_version": 5,
        "tracks": [tracks.stat().st_size, tracks.stat().st_mtime_ns],
        "metadata": metadata, "source": [video.stat().st_size, video.stat().st_mtime_ns],
        "fps": fps, "max_width": max_width, 'render_options': options,
        'reacquire_seconds': lifecycle.seconds}, sort_keys=True).encode()).hexdigest()
    sidecar = output.with_suffix(".json")
    if not force and output.is_file() and sidecar.is_file():
        previous = json.loads(sidecar.read_text())
        if (previous.get("input_signature") == signature and
                previous.get("output_bytes") == output.stat().st_size):
            return {"output": str(output), "reused": True}
    check_cancel(cancel)
    start, expected, ids = metadata["query_frame"], metadata["frames"], metadata["point_ids"]
    if (not isinstance(start, int) or not isinstance(expected, int) or start < 0 or expected < 1
            or start + expected > source["number_of_frames"] or not ids or len(set(ids)) != len(ids)):
        raise ValueError("track metadata has an invalid source frame range or point IDs")
    total_frames = expected
    if options['max_frames'] is not None:
        expected = min(expected, options['max_frames'])
    w, h = source["width"], source["height"]
    ratio = min(1., max_width / w)
    dw, dh = max(1, round(w * ratio)), max(1, round(h * ratio))
    size = (dw + dw % 2, dh + dh % 2)
    pose_data = None
    if options['show_pose'] and metadata.get('coordinate_space') == DISPLAY_COORDINATE_SPACE:
        with tracks.open(newline='') as stream:
            columns = set(csv.DictReader(stream).fieldnames or ())
        if columns & {'warp_h00', 'h00', 'scale', 'similarity_scale', 'rotation_degrees'}:
            from .track_pose import read_point_pose_tracks
            pose_data = read_point_pose_tracks(tracks, video=video, source=source,
                                               reacquire_seconds=lifecycle.seconds)
    if pose_data is not None:
        from ..pose.overlay import draw_point_pose_rows
        # Review uses geometry plus existing point IDs, avoiding duplicate labels.
        pose_options = dict(options, pose_labels=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.{uuid4().hex}.mp4")
    writer, processed = None, 0
    try:
        writer = H264Writer(temporary, size, fps, crf=options['crf'], preset=options['preset'])
        groups = iter(point_groups(tracks, ids, metadata))
        trails = {identity: deque(maxlen=options['trail_length']) for identity in ids} if options['trails'] else {}
        for offset, frame in enumerate(video_frames(video, start, expected, source=source)):
            check_cancel(cancel)
            row = next(groups, None)
            if row is None or row[0] != start + offset:
                raise ValueError("point frames must be consecutive and match source frame IDs")
            if frame.shape[:2] != (h, w):
                raise ValueError("decoded source dimensions differ from track metadata")
            accepted = lifecycle.step(row[0], [row[1][identity][1] and
                0 <= row[1][identity][0][0] < w and 0 <= row[1][identity][0][1] < h
                for identity in ids], resets.get(row[0], ()))
            row = (row[0], {identity: (row[1][identity][0], bool(valid))
                            for identity, valid in zip(ids, accepted)})
            if trails:
                frame = frame.copy()
                palette = [(0, 220, 255), (255, 180, 0), (80, 255, 80), (255, 80, 255), (80, 150, 255)]
                for identity, (xy, visible) in row[1].items():
                    trail = trails[identity]
                    if not visible or not (0 <= xy[0] < w and 0 <= xy[1] < h):
                        trail.clear()
                        continue
                    trail.append(np.rint(xy).astype(np.int32))
                    if len(trail) > 1:
                        cv2.polylines(frame, [np.array(trail)], False, palette[identity % len(palette)], 1, cv2.LINE_AA)
            if metadata.get("tracker", "tapnextpp") != "tapnextpp":
                valid = sum(point[1] for point in row[1].values())
                label = metadata["tracker"] + (f"/{motion_model}" if motion_model != "similarity" else "")
                image = draw_points(frame, row[1], row[0],
                    f"{label}: {valid}/{len(row[1])} valid")
            else:
                image = draw_points(frame, row[1], row[0])
            if (dw, dh) != (w, h):
                image = cv2.resize(image, (dw, dh), interpolation=cv2.INTER_AREA)
            if pose_data is not None:
                poses = {identity: pose for identity, pose in pose_data['rows_by_frame'][row[0]].items()
                         if row[1][identity][1]}
                draw_point_pose_rows(image, poses, row[0], supports=pose_data['supports'],
                                     source_size=(w, h), render_options=pose_options)
            image = cv2.copyMakeBorder(image, 0, size[1] - dh, 0, size[0] - dw,
                                       cv2.BORDER_CONSTANT, value=(0, 0, 0))
            writer.write(image)
            processed += 1
            if progress and (processed % 30 == 0 or processed == expected):
                progress(f"Rendering {video.name}: {processed}/{expected} frames")
        if processed != expected or (expected == total_frames and next(groups, None) is not None):
            raise ValueError("point/source frame counts do not match track metadata")
        writer.close()
        check_cancel(cancel)
        archive_files([output, sidecar], output.parent.name)
        temporary.replace(output)
        result = {"output": str(output), "reused": False, "frames": processed,
                  "source_frame_start": start, "fps": fps, "source": str(video),
                  "tracks": str(tracks), "width": size[0], "height": size[1],
                  "source_identity_verified": identity_verified,
                  "image_size_before_even_padding": [dw, dh],
                  "timing": "CFR sanity preview: one source frame per output frame; FPS override changes preview speed, not labels or alignment; VFR intervals are not preserved",
                  "codec": "H.264 libx264", "pixel_format": "yuv420p", "faststart": True,
                  'render_options': options, 'track_frames': total_frames,
                  'pose_overlay': pose_data is not None, 'render_version': 5,
                  'reacquire_seconds': lifecycle.seconds,
                  "output_bytes": output.stat().st_size,
                  "input_signature": signature}
        write_json(sidecar, result)
        return result
    finally:
        if writer:
            writer.close()
        temporary.unlink(missing_ok=True)
