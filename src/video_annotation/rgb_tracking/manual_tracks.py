"""GUI-independent sparse corrections of saved source-frame point tracks."""
from copy import deepcopy
import csv
import json
import os
from pathlib import Path
from uuid import uuid4

import numpy as np

from video_annotation.common.validation import require_no_output_references
from .annotations import sequence_id
from .render import point_groups
from .lifecycle import apply_lifecycle, validate_reacquire_seconds
from .tapnextpp import contained_path, write_json
from .video_geometry import DISPLAY_COORDINATE_SPACE, VIDEO_GEOMETRY_VERSION, display_rotation
from .workflow import archive_files, file_digest


class TrackEdits:
    """Absolute manual anchors applied to an immutable original CSV.

    Interpolation uses actual source frame numbers. Invisible endpoints create
    invisible held-coordinate gaps, never a fabricated visible trajectory.
    """

    def __init__(self, tracks, reacquire_seconds=None):
        self.input_tracks = contained_path(tracks, "data")
        opened = self._metadata(self.input_tracks)
        self.opened_metadata = deepcopy(opened)
        self.has_saved_policy = 'reacquire_seconds' in opened.get('track_policy', {})
        stored_seconds = opened.get('track_policy', {}).get('reacquire_seconds', .5)
        self._saved_reacquire_seconds = validate_reacquire_seconds(stored_seconds)
        self.reacquire_seconds = validate_reacquire_seconds(stored_seconds if reacquire_seconds is None else reacquire_seconds)
        manual = opened.get("manual_edits")
        if manual is not None and (not isinstance(manual, dict) or not isinstance(manual.get("base_tracks"), str)):
            raise ValueError("manual export lacks original-track provenance")
        self.output = self.input_tracks if manual is not None else None
        self.tracks = (contained_path(self.input_tracks.parent/manual["base_tracks"], "data")
                       if manual is not None else self.input_tracks)
        if self.tracks.suffix.lower() != ".csv" or not self.tracks.is_file():
            raise ValueError("manual correction needs an existing point CSV")
        self.metadata = self._metadata(self.tracks)
        if self.metadata.get("manual_edits") is not None:
            raise ValueError("manual provenance must refer directly to the original tracks")
        self._base_digest = file_digest(self.tracks)
        self._metadata_digest = file_digest(self.tracks.with_suffix(".json"))
        if manual is not None and (manual.get("base_tracks_sha256") != self._base_digest or
                manual.get("base_metadata_sha256") != self._metadata_digest):
            raise ValueError("original tracks or their metadata changed since the manual export")
        if manual is not None and opened.get("tracks_sha256") != file_digest(self.input_tracks):
            raise ValueError("manual CSV changed outside the saved anchor edits")
        self.video = contained_path(self.tracks.parent/self.metadata["source_video"], "data")
        if not self.video.is_file():
            raise ValueError("the original source video is unavailable")
        self._source_stat = [self.video.stat().st_size, self.video.stat().st_mtime_ns]
        if self.metadata.get("source_file_stat", self._source_stat) != self._source_stat:
            raise ValueError("source video changed since tracking; restore the tracked original")
        size = self.metadata.get("source_size_px")
        if size is None:
            from video_annotation.common.video import probe_video
            source = probe_video(self.video)[0]
            size = [source["width"], source["height"]]
        if (len(size) != 2 or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in size)):
            raise ValueError("point metadata needs valid source pixel dimensions")
        self.width, self.height = size
        self.point_ids = list(self.metadata["point_ids"])
        if (not self.point_ids or len(set(self.point_ids)) != len(self.point_ids) or
                any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in self.point_ids)):
            raise ValueError("point IDs must be unique nonnegative integers")
        groups = list(point_groups(self.tracks, self.point_ids, self.metadata))
        self.frames = np.array([frame for frame, _ in groups], np.int64)
        if (not len(self.frames) or self.frames[0] < 0 or np.any(np.diff(self.frames) != 1) or
                self.metadata.get("query_frame", int(self.frames[0])) != int(self.frames[0]) or
                self.metadata.get("frames", len(self.frames)) != len(self.frames)):
            raise ValueError("track rows must have consecutive original source frame IDs matching metadata")
        self.original_xy = np.array([[row[p][0] for p in self.point_ids] for _, row in groups], np.float64)
        self.original_visible = np.array([[row[p][1] for p in self.point_ids] for _, row in groups], bool)
        self.frames.setflags(write=False)
        self.original_xy.setflags(write=False)
        self.original_visible.setflags(write=False)
        self.source_fps = opened.get('track_policy', {}).get('source_fps') or self.metadata.get('source_metadata_fps')
        if self.source_fps is None:
            from video_annotation.common.video import probe_video
            self.source_fps = probe_video(self.video)[0]['fps']
        self._refresh_original_lifecycle()
        self._point_index = {point: i for i, point in enumerate(self.point_ids)}
        self.anchors = {point: {} for point in self.point_ids}
        self._undo = []
        self.tail = manual.get("tail", "keep") if manual is not None else "keep"
        self.join_frame = manual.get("join_frame") if manual is not None else None
        if manual is not None:
            for point, rows in manual.get("anchors", {}).items():
                for row in rows:
                    self.set_anchor(int(point), row["frame"], row["x"], row["y"], row["visible"])
        self._undo.clear()
        self._saved_anchors = deepcopy(self.anchors)
        self.preview(self.tail, self.join_frame)  # Validate restored tail settings.

    @staticmethod
    def _metadata(path):
        value = json.loads(path.with_suffix(".json").read_text())
        if not isinstance(value, dict) or value.get("kind") != "rgb_point_tracks":
            raise ValueError("manual correction needs rgb_point_tracks metadata")
        if value.get("coordinate_space") != DISPLAY_COORDINATE_SPACE:
            raise ValueError("manual correction requires displayed-video pixel coordinates")
        return value

    @property
    def dirty(self):
        return self.anchors != self._saved_anchors or self.reacquire_seconds != self._saved_reacquire_seconds

    def _refresh_original_lifecycle(self):
        inside = np.all((self.original_xy >= 0) & (self.original_xy < [self.width, self.height]), axis=-1)
        self.original_lifecycle_visible, _ = apply_lifecycle(self.frames, self.point_ids, self.original_xy,
            self.original_visible & inside, self.source_fps, self.reacquire_seconds)
        self.original_lifecycle_visible.setflags(write=False)

    def set_reacquire_seconds(self, seconds):
        seconds = validate_reacquire_seconds(seconds)
        if seconds != self.reacquire_seconds:
            self.reacquire_seconds = seconds
            self._refresh_original_lifecycle()

    def _location(self, point_id, frame):
        if isinstance(point_id, bool) or not isinstance(point_id, (int, np.integer)) or point_id not in self._point_index:
            raise ValueError(f"unknown point ID: {point_id}")
        if (isinstance(frame, bool) or not isinstance(frame, (int, np.integer)) or
                frame < self.frames[0] or frame > self.frames[-1]):
            raise ValueError("anchor frame must be represented in the original track")
        return int(point_id), int(frame)

    def set_anchor(self, point_id, frame, x, y, visible=True):
        point_id, frame = self._location(point_id, frame)
        if not isinstance(visible, (bool, np.bool_)):
            raise ValueError("anchor visibility must be boolean")
        xy = np.asarray([x, y], float)
        if not np.isfinite(xy).all() or not (0 <= xy[0] <= self.width-1 and 0 <= xy[1] <= self.height-1):
            raise ValueError(f"manual point must lie inside displayed frame {self.width}x{self.height}")
        value = float(xy[0]), float(xy[1]), bool(visible)
        if self.anchors[point_id].get(frame) != value:
            self._undo.append(deepcopy(self.anchors))
            self.anchors[point_id][frame] = value

    def remove_anchor(self, point_id, frame):
        return bool(self.remove_anchors(point_id, [frame]))

    def remove_anchors(self, point_id, frames):
        """Remove existing anchors as one undoable edit; validate before changing."""
        locations = {self._location(point_id, frame) for frame in frames}
        selected = [(point, frame) for point, frame in locations if frame in self.anchors[point]]
        if selected:
            self._undo.append(deepcopy(self.anchors))
            for point, frame in selected:
                del self.anchors[point][frame]
        return len(selected)

    def set_anchor_visibility(self, point_id, frames, visible):
        """Keep selected anchor coordinates and change visibility in one undo step."""
        if not isinstance(visible, (bool, np.bool_)):
            raise ValueError("anchor visibility must be boolean")
        locations = {self._location(point_id, frame) for frame in frames}
        if any(frame not in self.anchors[point] for point, frame in locations):
            raise ValueError("select existing manual anchors to change their visibility")
        selected = [(point, frame) for point, frame in locations if self.anchors[point][frame][2] != visible]
        if selected:
            self._undo.append(deepcopy(self.anchors))
            for point, frame in selected:
                x, y, _ = self.anchors[point][frame]
                self.anchors[point][frame] = x, y, bool(visible)
        return len(selected)

    def undo(self):
        if not self._undo:
            return False
        self.anchors = self._undo.pop()
        return True

    def preview(self, tail="keep", join_frame=None):
        if tail not in ("keep", "stop", "next"):
            raise ValueError("tail must be keep, stop or next")
        if join_frame is not None and (isinstance(join_frame, bool) or not isinstance(join_frame, (int, np.integer))):
            raise ValueError("join_frame must be an original source frame number")
        xy, visible = self.original_xy.copy(), self.original_visible.copy()
        labels = np.full(visible.shape, "original", dtype="<U12")
        notices = []

        def interval(point, first, start, last, end):
            selected = (self.frames > first) & (self.frames < last)
            if start[2] and end[2]:
                weight = (self.frames[selected]-first)/(last-first)
                xy[selected, point] = np.asarray(start[:2]) + weight[:, None]*(np.asarray(end[:2])-start[:2])
                visible[selected, point], labels[selected, point] = True, "interpolated"
            else:
                xy[selected, point] = start[:2]
                visible[selected, point], labels[selected, point] = False, "terminated"

        for identity, anchors in self.anchors.items():
            if not anchors:
                continue
            point = self._point_index[identity]
            ordered = sorted(anchors.items())
            for (first, start), (last, end) in zip(ordered, ordered[1:]):
                interval(point, first, start, last, end)
            last, value = ordered[-1]
            if tail == "stop":
                selected = self.frames > last
                xy[selected, point], visible[selected, point], labels[selected, point] = value[:2], False, "terminated"
            elif tail == "next":
                candidates = np.flatnonzero((self.frames > last) & self.original_visible[:, point])
                if join_frame is not None:
                    candidates = candidates[self.frames[candidates] == join_frame]
                    if not len(candidates):
                        raise ValueError(f"join frame must be later than point {identity}'s last anchor and originally valid")
                if len(candidates):
                    index = candidates[0]
                    interval(point, last, value, int(self.frames[index]), (*self.original_xy[index, point], True))
                elif last < self.frames[-1]:
                    notices.append(f"Point {identity}: no valid original join; later original rows retained")
            for frame, value in ordered:
                index = int(frame-self.frames[0])
                xy[index, point], visible[index, point], labels[index, point] = value[:2], value[2], "manual"
        inside = np.all((xy >= 0) & (xy < [self.width, self.height]), axis=-1)
        accepted, self.track_policy = apply_lifecycle(self.frames, self.point_ids, xy,
            visible & inside, self.source_fps, self.reacquire_seconds, self.anchors)
        labels[visible & ~accepted] = 'terminated'
        self.status = "; ".join(notices)
        return xy, accepted, labels

    def save(self, output=None, *, tail="keep", join_frame=None):
        xy, visible, labels = self.preview(tail, join_frame)
        output = contained_path(output or self.output or self.tracks.parent/"manual/points.csv", "data")
        sidecar = output.with_suffix(".json")
        protected = {self.tracks, self.tracks.with_suffix(".json"), self.video}
        if output.suffix.lower() != ".csv" or output in protected or sidecar in protected:
            raise ValueError("manual edits need a separate CSV; original tracks/video are immutable")
        if file_digest(self.tracks) != self._base_digest or file_digest(self.tracks.with_suffix(".json")) != self._metadata_digest:
            raise ValueError("original tracks changed while editing; reload before saving")
        if [self.video.stat().st_size, self.video.stat().st_mtime_ns] != self._source_stat:
            raise ValueError("source video changed while editing; restore the tracked original")
        if output.exists() or sidecar.exists():
            existing = self._metadata(output)
            base = existing.get("manual_edits", {}).get("base_tracks")
            if base is None or (output.parent/base).resolve() != self.tracks:
                raise ValueError("output belongs to another track; choose a fresh manual CSV")
        output.parent.mkdir(parents=True, exist_ok=True)
        token = uuid4().hex
        pending = output.with_name(f".{output.name}.{token}.partial")
        pending_json = pending.with_name(pending.name+".json")
        backups = {}
        try:
            with pending.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["frame", "point_id", "x", "y", "visible", "quality_score", "manual_origin"])
                for index, frame in enumerate(self.frames):
                    for point, identity in enumerate(self.point_ids):
                        end = self.track_policy['point_end_frames'][str(identity)]
                        if end is None or frame > end:
                            continue
                        valid = bool(visible[index, point])
                        writer.writerow([int(frame), identity, *xy[index, point], int(valid), 1 if valid else -1, labels[index, point]])
            metadata = {"schema_version": 1, "kind": "rgb_point_tracks", "tracker": "manual",
                "tracks": output.name, "tracks_bytes": pending.stat().st_size, "tracks_sha256": file_digest(pending),
                "source_video": os.path.relpath(self.video, output.parent), "source_size_px": [self.width, self.height],
                "source_file_stat": self._source_stat, "source_metadata_fps": self.metadata.get("source_metadata_fps"),
                "coordinate_space": DISPLAY_COORDINATE_SPACE, "video_geometry_version": VIDEO_GEOMETRY_VERSION,
                "display_rotation_degrees": display_rotation(self.metadata), "point_ids": self.point_ids,
                "query_frame": int(self.frames[0]), "frames": len(self.frames),
                "query_points_xy": xy[0].tolist(), "track_policy": deepcopy(self.track_policy),
                "point_rows": sum((end-int(self.frames[0])+1) if end is not None else 0
                                  for end in self.track_policy['point_end_frames'].values()),
                "score_column": "quality_score", "score_kind": "signed visibility flag only; no tracker fit confidence",
                "visibility": "manual/interpolated validity or unchanged original validity; not measured physical visibility",
                "timing": "original zero-based source frame IDs; linear positions use source frame numbers",
                "manual_edits": {"schema_version": 1, "base_tracks": os.path.relpath(self.tracks, output.parent),
                    "base_tracks_sha256": self._base_digest, "base_metadata_sha256": self._metadata_digest,
                    "tail": tail, "join_frame": int(join_frame) if join_frame is not None else None,
                    "anchors": {str(identity): [{"frame": frame, "x": value[0], "y": value[1], "visible": value[2]}
                                for frame, value in sorted(anchors.items())] for identity, anchors in self.anchors.items()}}}
            require_no_output_references(metadata)
            write_json(pending_json, metadata)
            archive_files([output, sidecar], sequence_id(self.video))
            for path in (output, sidecar):
                if path.is_file():
                    backup = path.with_name(f".{path.name}.{token}.previous")
                    os.link(path, backup)
                    backups[path] = backup
            try:
                pending.replace(output)
                pending_json.replace(sidecar)
            except BaseException:
                for path in (output, sidecar):
                    if path in backups:
                        backups[path].replace(path)
                    else:
                        path.unlink(missing_ok=True)
                raise
        finally:
            pending.unlink(missing_ok=True)
            pending_json.unlink(missing_ok=True)
            for backup in backups.values():
                backup.unlink(missing_ok=True)
        self.output, self.tail, self.join_frame = output, tail, join_frame
        self._saved_anchors = deepcopy(self.anchors)
        self._saved_reacquire_seconds = self.reacquire_seconds
        self.opened_metadata, self.has_saved_policy = deepcopy(metadata), True
        return output
