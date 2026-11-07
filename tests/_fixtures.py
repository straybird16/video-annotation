"""Synthetic video fixture copied from Event/tests/test_point_labeler.py."""
import subprocess

import numpy as np


def make_video(path, *, width=64, height=48, count=7):
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = np.zeros((count, height, width, 3), dtype=np.uint8)
    for i, frame in enumerate(frames):
        frame[:] = (20 + i * 3, 50, 80)
        frame[height//3:height//3+5, width//3:width//3+5] = (255, 220, 20)
    # Odd source geometry is intentional; preview encoder must still be YUV420.
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", "30", "-i", "-", "-c:v", "libx264rgb", "-crf", "0",
        "-threads", "1", "-movflags", "+faststart", str(path)], input=frames.tobytes(), check=True)
    return path
