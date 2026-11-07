"""Focused regressions in an isolated workspace; native models are mocked."""
import atexit
import os
from pathlib import Path
import tempfile

_temporary = tempfile.TemporaryDirectory(prefix="video-annotation-tests-")
atexit.register(_temporary.cleanup)
_workspace = Path(_temporary.name)
(_workspace / "data").mkdir()
(_workspace / "outputs").mkdir()
os.environ["VIDEO_ANNOTATION_WORKSPACE"] = str(_workspace)
os.environ["VIDEO_ANNOTATION_MODELS"] = str(_workspace / "models")
