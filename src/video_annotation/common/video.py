"""Video probing and H.264 writing.
Sources: Event/scripts/temporal_mapping/{video,array_video}.py.

Implementations extracted from the Event repository; source files are retained there.
"""
from fractions import Fraction
import json
from pathlib import Path
import subprocess

import numpy as np


def probe_video(path, *, timing=False, timeout=30.):
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError(f'RGB video does not exist: {path}')
    entries = 'stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration:stream_side_data=rotation:format=duration'
    if timing:
        entries += ':packet=pts_time,duration_time'
    result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                             '-show_entries', entries, '-of', 'json', str(path)],
                            capture_output=True, text=True, check=True, timeout=timeout)
    data = json.loads(result.stdout)
    if not data.get('streams'):
        raise ValueError('RGB input has no video stream.')
    stream = data['streams'][0]
    rate = stream.get('avg_frame_rate', '0/0')
    if rate in ('0/0', '0/1'):
        rate = stream['r_frame_rate']
    fps = float(Fraction(rate))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('Original video needs a positive presentation frame rate.')
    rotation = float(next((row['rotation'] for row in stream.get('side_data_list', []) if 'rotation' in row), 0))
    if not np.isfinite(rotation) or rotation % 90:
        raise ValueError('Only right-angle video display rotations are supported.')
    width, height = stream['width'], stream['height']
    if rotation % 180:
        width, height = height, width
    duration = float(stream.get('duration', data.get('format', {}).get('duration', 0)))
    count = int(stream['nb_frames']) if stream.get('nb_frames', '').isdigit() else None
    if not timing and count is None:
        return probe_video(path, timing=True, timeout=timeout)
    boundaries = None
    if timing:
        packets = data.get('packets', [])
        if not packets or any('pts_time' not in row for row in packets):
            raise ValueError('Original video needs presentation timestamps for every video packet.')
        pts = np.array([float(row['pts_time']) for row in packets])
        durations = np.array([float(row.get('duration_time', 0)) for row in packets])
        order = np.argsort(pts)
        pts, durations = pts[order], durations[order]
        if count is not None and len(pts) != count:
            raise ValueError('Video packet/frame count differs; use a file with one timestamped packet per frame.')
        if not np.isfinite(pts).all() or np.any(np.diff(pts) <= 0):
            raise ValueError('Original video presentation timestamps must be finite and unique.')
        count = len(pts)
        pts = pts-pts[0]
        final_duration = durations[-1] if durations[-1] > 0 else 1/fps
        duration = float(pts[-1]+final_duration)
        boundaries = np.r_[pts, duration]*1e6
    if count < 2 or not np.isfinite([fps, duration]).all() or min(fps, duration, stream['width'], stream['height']) <= 0:
        raise ValueError('Need a finite-FPS RGB video with at least two frames.')
    metadata = {'schema_version': 1, 'kind': 'unmarked_rgb_video', 'source_video': str(path),
                'width': width, 'height': height, 'fps': fps,
                'encoded_size_px': [stream['width'], stream['height']], 'display_rotation_degrees': rotation,
                'number_of_frames': count, 'source_frames': count,
                'encoded_duration_seconds': duration, 'initial_frame': 0,
                'source_size_px': [width, height],
                'source_to_monitor': np.eye(3).tolist(),
                'timing_method': 'original presentation timestamps; no playback-rate fitting'}
    return metadata, boundaries


def h264_encoding_args(lossless=False, *, crf=20, preset='veryfast'):
    """Portable YUV420 by default; explicit exact RGB avoids chroma subsampling."""
    return ['-c:v','libx264rgb' if lossless else 'libx264', '-preset',preset,
            '-qp' if lossless else '-crf','0' if lossless else str(crf),
            '-pix_fmt','bgr24' if lossless else 'yuv420p','-threads','2','-movflags','+faststart']


class H264Writer:
    """Stream BGR frames to MP4. lossless=True preserves decoded BGR pixels exactly."""
    def __init__(self, path, size, fps, *, lossless=False, crf=20, preset='veryfast'):
        width,height=size
        if width <= 0 or height <= 0 or width % 2 or height % 2 or not np.isfinite(fps) or fps <= 0:
            raise ValueError('H.264 output needs positive even dimensions and positive FPS.')
        command=['ffmpeg','-v','error','-nostdin','-n','-f','rawvideo','-pix_fmt','bgr24',
                 '-s',f'{width}x{height}','-r',str(fps),'-i','-','-an',
                 *h264_encoding_args(lossless, crf=crf, preset=preset),str(path)]
        self.command=command
        self.process=subprocess.Popen(command,stdin=subprocess.PIPE)

    def write(self, frame):
        self.process.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self):
        if self.process.stdin.closed:
            return
        self.process.stdin.close()
        if self.process.wait():
            raise OSError('FFmpeg H.264 encoding failed.')

    release = close

    def isOpened(self):
        return self.process.poll() is None
