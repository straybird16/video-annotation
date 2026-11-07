"""Small shared visibility gate for bounded point reacquisition in source time."""
import math
import csv
from collections import Counter
from pathlib import Path

import numpy as np

DEFAULT_REACQUIRE_SECONDS = 0.5


def validate_reacquire_seconds(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError('Reacquisition seconds must be a finite nonnegative number')
    return float(value)


class TrackLifecycle:
    """Accept native recovery only within the window since the first missing frame."""

    def __init__(self, point_ids, fps, reacquire_seconds=DEFAULT_REACQUIRE_SECONDS):
        if isinstance(fps, bool) or not math.isfinite(fps) or fps <= 0:
            raise ValueError('Source FPS must be positive for track loss timing')
        self.ids = list(point_ids)
        self.seconds = validate_reacquire_seconds(reacquire_seconds)
        self.fps = float(fps)
        self.missing_since = [None] * len(self.ids)
        self.last_valid = [None] * len(self.ids)
        self.terminated = np.zeros(len(self.ids), bool)
        self.previous_frame = None

    def step(self, frame, visible, reset_ids=()):
        if self.previous_frame is not None and frame != self.previous_frame + 1:
            raise ValueError('Track loss policy requires consecutive source frames')
        native = np.asarray(visible, bool)
        if native.shape != (len(self.ids),):
            raise ValueError('Track loss policy needs one visibility value per point')
        accepted = native.copy()
        reset_ids = set(reset_ids)
        for index, good in enumerate(native):
            if self.ids[index] in reset_ids and good:
                self.terminated[index] = False
                self.missing_since[index] = None
            missing = self.missing_since[index]
            if missing is not None and (frame - missing) / self.fps > self.seconds + 1e-12:
                self.terminated[index] = True
            if self.terminated[index]:
                accepted[index] = False
            elif good:
                self.last_valid[index] = int(frame)
                self.missing_since[index] = None
            else:
                if missing is None:
                    self.missing_since[index] = int(frame)
                if self.seconds == 0:
                    self.terminated[index] = True
        self.previous_frame = int(frame)
        return accepted

    def metadata(self):
        return {'version': 1, 'reacquire_seconds': self.seconds, 'source_fps': self.fps,
                'trimmed_terminal_rows': True,
                'point_end_frames': {str(identity): frame for identity, frame in zip(self.ids, self.last_valid)}}


def trim_terminal_rows(path, ends):
    """Keep recoverable internal gaps, dropping each point's unaccepted tail once."""
    path = Path(path)
    temporary = path.with_name(path.name + '.retained')
    count = 0
    registrations, models = Counter(), Counter()
    try:
        with path.open(newline='') as source, temporary.open('w', newline='') as target:
            reader = csv.DictReader(source)
            writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                end = ends[str(row['point_id'])]
                if end is None or int(row['frame']) > end:
                    continue
                writer.writerow(row)
                count += 1
                if 'registration_status' in row:
                    registrations[row['registration_status'] or 'not_applicable'] += 1
                    if row['valid'] == '1' and row.get('motion_model'):
                        models[row['motion_model']] += 1
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return count, registrations, models
