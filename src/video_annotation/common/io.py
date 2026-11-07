"""Small artifact IO. Source: Event/scripts/temporal_mapping/common.py.

Implementations extracted from the Event repository; source files are retained there.
"""
import hashlib
import json
from pathlib import Path


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False, default=str)+'\n')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()
